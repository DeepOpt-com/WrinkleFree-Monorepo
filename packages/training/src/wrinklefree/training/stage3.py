"""Stage 3: Distillation Fine-Tuning - Train with teacher guidance."""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from wrinklefree.distillation import BitDistillLoss
from wrinklefree.training.fsdp_wrapper import setup_distributed, wrap_model_fsdp
from wrinklefree.training.trainer import Trainer, create_optimizer, create_scheduler

logger = logging.getLogger(__name__)


class TeacherWrapper(nn.Module):
    """
    Wrapper for frozen teacher model.

    Handles teacher model loading, freezing, and optional CPU offloading.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        load_in_fp16: bool = True,
        offload_to_cpu: bool = False,
        use_eager_attention: bool = True,
    ):
        """
        Initialize teacher wrapper.

        Args:
            model_name_or_path: HuggingFace model name or path
            device: Device to load model on
            load_in_fp16: Load in FP16 for memory efficiency
            offload_to_cpu: Offload to CPU between forward passes
            use_eager_attention: Use eager attention instead of SDPA/Flash
                (required for attention distillation, as SDPA doesn't return attention weights)
        """
        super().__init__()

        logger.info(f"Loading teacher model: {model_name_or_path}")

        dtype = torch.bfloat16 if load_in_fp16 else torch.float32

        # Use eager attention if attention distillation is needed
        # SDPA/Flash attention doesn't return attention weights
        attn_impl = "eager" if use_eager_attention else None
        if use_eager_attention:
            logger.info("Using eager attention for teacher (required for attention distillation)")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        # Freeze teacher
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.offload_to_cpu = offload_to_cpu

        if not offload_to_cpu:
            self.model = self.model.to(device)

        logger.info(f"Teacher model loaded (frozen, {'BF16' if load_in_fp16 else 'FP32'})")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """
        Forward pass through teacher model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            output_attentions: Whether to return attention weights

        Returns:
            Dictionary with logits and optionally attentions
        """
        if self.offload_to_cpu:
            self.model = self.model.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        result = {
            "logits": outputs.logits,
            "attentions": outputs.attentions if output_attentions else None,
        }

        if self.offload_to_cpu:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

        return result


class Stage3Trainer(Trainer):
    """
    Trainer for Stage 3 distillation fine-tuning.

    Extends base Trainer with:
    - Teacher model handling
    - BitDistill loss computation
    """

    def __init__(
        self,
        model: nn.Module,
        teacher: TeacherWrapper,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        config: DictConfig,
        distill_config: DictConfig,
        **kwargs,
    ):
        # Create BitDistill loss
        loss_fn = BitDistillLoss(
            lambda_logits=distill_config.lambda_logits,
            gamma_attention=distill_config.gamma_attention,
            temperature=distill_config.temperature,
        )

        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            config=config,
            **kwargs,
        )

        self.teacher = teacher
        self.output_attentions = distill_config.gamma_attention > 0

    def _forward_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward step with teacher distillation."""
        # Get teacher outputs
        teacher_outputs = self.teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_attentions=self.output_attentions,
        )

        # Get student outputs
        student_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_attentions=self.output_attentions,
        )

        # Compute distillation loss
        loss_dict = self.loss_fn(
            student_logits=student_outputs["logits"],
            teacher_logits=teacher_outputs["logits"],
            student_attentions=student_outputs.get("attentions"),
            teacher_attentions=teacher_outputs.get("attentions"),
            labels=batch["labels"],
            attention_mask=batch.get("attention_mask"),
        )

        return loss_dict


def run_stage3(
    student_model: nn.Module,
    teacher_model_name: str,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    config: DictConfig,
    distill_config: DictConfig,
    output_dir: Path,
    resume_from: Optional[Path] = None,
    run_manager: Optional[Any] = None,
    experiment_name: Optional[str] = None,
) -> nn.Module:
    """
    Run Stage 3: Distillation fine-tuning.

    Args:
        student_model: BitNet student model from Stage 2
        teacher_model_name: HuggingFace name for teacher model
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        config: Training configuration
        distill_config: Distillation configuration
        output_dir: Output directory
        resume_from: Optional checkpoint to resume from
        run_manager: Optional RunManager for GCS checkpoint uploads
        experiment_name: Name for GCS checkpoint path

    Returns:
        Fine-tuned student model
    """
    logger.info("Stage 3: Starting distillation fine-tuning")

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Check if vLLM teacher is configured
    vllm_config = distill_config.get("vllm", None)
    use_vllm = vllm_config is not None and vllm_config.get("enabled", False)

    if use_vllm:
        from wrinklefree.distillation.vllm_teacher import VLLMTeacherWrapper, VLLMTeacherWithPrefetch

        vllm_url = vllm_config.get("base_url", "http://localhost:8000")
        use_prefetch = vllm_config.get("prefetch", False)
        top_k_logprobs = vllm_config.get("top_k_logprobs", 100)

        logger.info(f"Using vLLM teacher at {vllm_url} (prefetch={use_prefetch})")

        # Note: vLLM doesn't support attention distillation
        if distill_config.gamma_attention > 0:
            logger.warning(
                "vLLM does not support attention distillation. "
                "Setting gamma_attention=0 for logits-only distillation."
            )
            # Create a mutable copy if needed
            distill_config = dict(distill_config)
            distill_config["gamma_attention"] = 0

        TeacherClass = VLLMTeacherWithPrefetch if use_prefetch else VLLMTeacherWrapper
        teacher = TeacherClass(
            model_name=teacher_model_name,
            base_url=vllm_url,
            top_k_logprobs=top_k_logprobs,
        )
    else:
        # Load teacher model in-process
        load_in_4bit = distill_config.get("teacher_4bit", False)

        # Use eager attention if attention distillation is enabled
        # SDPA/Flash attention doesn't return attention weights
        needs_attention_weights = distill_config.gamma_attention > 0
        use_eager_attention = needs_attention_weights

        teacher = TeacherWrapper(
            model_name_or_path=teacher_model_name,
            device=device,
            load_in_fp16=True,
            offload_to_cpu=False,  # Keep on GPU for speed
            use_eager_attention=use_eager_attention,
        )

    # Move student to device
    student_model = student_model.to(device)

    # Wrap with FSDP if multi-GPU
    if world_size > 1:
        from wrinklefree.models import BitNetDecoderLayer
        student_model = wrap_model_fsdp(
            student_model,
            transformer_layer_cls=BitNetDecoderLayer,
            sharding_strategy=config.distributed.fsdp.sharding_strategy,
            mixed_precision=config.distributed.fsdp.mixed_precision.enabled,
            activation_checkpointing=config.distributed.fsdp.activation_checkpointing.enabled,
        )

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        student_model,
        learning_rate=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        use_8bit=config.training.optimizer.type == "adamw_8bit",
    )

    # Initialize influence components if configured
    influence_config = getattr(config, "influence", None)
    if influence_config and influence_config.get("enabled", False):
        try:
            # Lazy import - only when influence is enabled
            from cheapertraining import (
                DataInfCalculator,
                MixtureWeightCalculator,
                InfluenceAwareOptimizer,
                MixedDataset,
                InfluenceConfig,
                DiscriminativeGradientExtractor,
            )
        except ImportError:
            logger.error(
                "CheaperTraining library not found. "
                "Install it with: pip install -e ../WrinkleFree-CheaperTraining"
            )
            raise
            
        logger.info("Initializing influence-aware training components")
        
        # Create DataInf calculator
        inf_config = InfluenceConfig(**influence_config.get("config", {}))
        # Note: We need a gradient extractor that works with our model
        # For now assuming compatible interface or using library's extractor if model compatible
        # In practice might need an adapter
        grad_extractor = DiscriminativeGradientExtractor(student_model, inf_config)
        datainf = DataInfCalculator(grad_extractor, inf_config)
        
        # Create mixture calculator
        mixture_calc = MixtureWeightCalculator(datainf)
        
        # Get mixed dataset from dataloader if available
        # This assumes train_dataloader.dataset is or contains MixedDataset
        dataset = train_dataloader.dataset
        if hasattr(dataset, "dataset") and isinstance(dataset.dataset, MixedDataset):
            mixed_dataset = dataset.dataset
        elif isinstance(dataset, MixedDataset):
            mixed_dataset = dataset
        else:
            logger.warning("Could not find MixedDataset in dataloader, influence updates disabled")
            mixed_dataset = None
            
        if mixed_dataset:
            # Wrap optimizer
            optimizer = InfluenceAwareOptimizer(
                optimizer=optimizer,
                mixture_calculator=mixture_calc,
                mixed_dataset=mixed_dataset,
                update_interval=influence_config.get("update_interval", 1000),
                learning_rate=influence_config.get("learning_rate", 0.2),
                rank=rank,
            )
            logger.info("Optimizer wrapped with InfluenceAwareOptimizer")

    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.training.scheduler.type,
        num_training_steps=config.training.max_steps,
        num_warmup_steps=config.training.scheduler.warmup_steps,
    )

    # Create trainer
    trainer = Stage3Trainer(
        model=student_model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config.training,
        distill_config=distill_config,
        device=device,
        rank=rank,
        world_size=world_size,
        run_manager=run_manager,
        experiment_name=experiment_name,
        stage="stage3",
    )
    trainer.output_dir = output_dir

    # Resume if specified
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)

    # Train
    metrics = trainer.train()

    logger.info(f"Stage 3: Complete. Final metrics: {metrics}")

    return student_model
