"""Stage 1.9: Layer-wise Distillation - Align BitNet hidden states with teacher.

This stage runs after Stage 1 (SubLN insertion) but before Stage 2 (pre-training).
It performs lightweight layer-wise distillation to align BitNet hidden states
with the original full-precision teacher model.

Research basis:
- OneBit (arxiv.org/abs/2402.11295): L2-normalized MSE for scale-invariance
- BitDistill (arxiv.org/abs/2510.13998): Layer-wise alignment, later layers more important
- HBLLM: Saliency-based mixed-precision curriculum for stable quantization
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from wrinklefree.distillation.layerwise_loss import (
    LayerwiseDistillationLoss,
    LayerwiseLossType,
)
from wrinklefree.models.bitlinear import SaliencyAwareBitLinear
from wrinklefree.quantization.saliency_curriculum import SaliencyCurriculum
from wrinklefree.training.fsdp_wrapper import setup_distributed, wrap_model_fsdp
from wrinklefree.training.trainer import Trainer, create_optimizer, create_scheduler

logger = logging.getLogger(__name__)


class HiddenStateTeacherWrapper(nn.Module):
    """
    Teacher wrapper that extracts per-layer hidden states.

    Extends the basic teacher wrapper to return hidden states from all
    transformer layers for layer-wise distillation.

    Args:
        model_name_or_path: HuggingFace model name or local path
        device: Device to load model on
        load_in_fp16: Load model in FP16 for memory efficiency
        offload_to_cpu: Offload model to CPU between forward passes
        load_in_4bit: Load model in 4-bit NF4 quantization for 3x memory reduction
        use_flash_attention: Use Flash Attention 2 for faster attention
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        load_in_fp16: bool = True,
        offload_to_cpu: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        logger.info(f"Loading teacher model for hidden state extraction: {model_name_or_path}")

        # Use bfloat16 for better training stability (matches student dtype)
        dtype = torch.bfloat16 if load_in_fp16 else torch.float32

        # Build model loading kwargs
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        # Add 4-bit quantization if requested (3x memory reduction)
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Using 4-bit NF4 quantization for teacher (3x memory reduction)")
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to BF16")

        # Add Flash Attention 2 if requested (15-25% speedup)
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2 for teacher")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        # Freeze teacher
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = device
        self.offload_to_cpu = offload_to_cpu

        if not offload_to_cpu:
            self.model = self.model.to(device)

        logger.info(
            f"Teacher model loaded (frozen, {'BF16' if load_in_fp16 else 'FP32'}, "
            f"offload={'enabled' if offload_to_cpu else 'disabled'})"
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass returning hidden states from all layers.

        Args:
            input_ids: Input token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)

        Returns:
            Dictionary with:
                - hidden_states: Tuple of hidden states per layer
                - logits: Output logits
        """
        if self.offload_to_cpu:
            self.model = self.model.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Always extract hidden states
        )

        result = {
            "hidden_states": outputs.hidden_states,
            "logits": outputs.logits,
        }

        if self.offload_to_cpu:
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

        return result


class Stage19Trainer(Trainer):
    """
    Trainer for Stage 1.9 layer-wise distillation.

    Extends base Trainer with:
    - Teacher model handling for hidden state extraction
    - Layer-wise distillation loss computation
    - Token-based training limit support

    Args:
        model: BitNet student model (from Stage 1)
        teacher: HiddenStateTeacherWrapper with frozen teacher
        optimizer: Optimizer instance
        train_dataloader: Training data loader
        config: Training configuration (DictConfig)
        layerwise_config: Layer-wise distillation configuration
        **kwargs: Additional arguments for base Trainer
    """

    def __init__(
        self,
        model: nn.Module,
        teacher: HiddenStateTeacherWrapper,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        config: DictConfig,
        layerwise_config: DictConfig,
        **kwargs,
    ):
        # Parse loss type from config
        loss_type_str = layerwise_config.get("loss_type", "mse_normalized")
        loss_type = LayerwiseLossType(loss_type_str)

        # Get layer weights configuration
        layer_weights = layerwise_config.get("layer_weights", None)

        # Create layerwise loss
        loss_fn = LayerwiseDistillationLoss(
            loss_type=loss_type,
            layer_weights=layer_weights,
            hidden_size=layerwise_config.get("hidden_size"),
            vocab_size=layerwise_config.get("vocab_size"),
            temperature=layerwise_config.get("temperature", 1.0),
            normalize=layerwise_config.get("normalize", True),
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

        # Token counting for this stage
        self.total_tokens = getattr(config, "total_tokens", 100_000_000)  # Default 100M
        self.tokens_processed = 0
        self.seq_length = getattr(config, "max_seq_length", 512)

        # LM loss weight for combined training (maintains language modeling capability)
        # 0.0 = pure hidden state distillation (original behavior)
        # 0.5 = equal weight to distillation and cross-entropy
        # 1.0 = pure cross-entropy (no distillation)
        self.lm_loss_weight = layerwise_config.get("lm_loss_weight", 0.0)

        # Distillation weight schedule (ramps down distillation over training)
        schedule_config = layerwise_config.get("distill_schedule", None)
        if schedule_config is not None and getattr(schedule_config, "enabled", False):
            self.distill_schedule_enabled = True
            self.distill_schedule_type = getattr(schedule_config, "type", "cosine")
            self.distill_initial_weight = getattr(schedule_config, "initial_weight", 0.5)
            self.distill_final_weight = getattr(schedule_config, "final_weight", 0.0)
            self.distill_warmup_steps = getattr(schedule_config, "warmup_steps", 0)
            logger.info(
                f"Distillation schedule enabled: type={self.distill_schedule_type}, "
                f"initial={self.distill_initial_weight}, final={self.distill_final_weight}, "
                f"warmup={self.distill_warmup_steps}"
            )
        else:
            self.distill_schedule_enabled = False

        logger.info(f"Stage 1.9 initialized with loss_type={loss_type_str}, layer_weights={layer_weights}")
        if self.lm_loss_weight > 0:
            logger.info(f"LM loss enabled with weight={self.lm_loss_weight}")

        # Saliency curriculum setup
        curriculum_config = getattr(layerwise_config, "saliency_curriculum", None)
        if curriculum_config is not None and getattr(curriculum_config, "enabled", False):
            self.saliency_curriculum = SaliencyCurriculum(
                initial_saliency_k=getattr(curriculum_config, "initial_k", 0.1),
                final_saliency_k=getattr(curriculum_config, "final_k", 0.0),
                ema_decay=getattr(curriculum_config, "ema_decay", 0.99),
                schedule_type=getattr(curriculum_config, "schedule_type", "cosine"),
                warmup_steps=getattr(curriculum_config, "warmup_steps", 0),
                update_interval=getattr(curriculum_config, "update_interval", 10),
            )

            # Attach curriculum to all SaliencyAwareBitLinear layers
            self._attach_saliency_curriculum(model)
            logger.info(
                f"Saliency curriculum enabled: initial_k={self.saliency_curriculum.initial_k}, "
                f"final_k={self.saliency_curriculum.final_k}, "
                f"schedule={curriculum_config.schedule_type}, "
                f"update_interval={self.saliency_curriculum.update_interval}"
            )
        else:
            self.saliency_curriculum = None

    def _attach_saliency_curriculum(self, model: nn.Module) -> None:
        """
        Attach saliency curriculum to all SaliencyAwareBitLinear layers in the model.

        Args:
            model: The model containing SaliencyAwareBitLinear layers
        """
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, SaliencyAwareBitLinear):
                module.set_saliency_curriculum(self.saliency_curriculum, name)
                count += 1
                logger.debug(f"Attached saliency curriculum to layer: {name}")

        logger.info(f"Attached saliency curriculum to {count} SaliencyAwareBitLinear layers")

    def _get_current_distill_weight(self) -> float:
        """
        Get current distillation weight based on schedule.

        Returns a value between 0.0 and 1.0 representing the distillation weight.
        This weight is applied as: loss = distill_weight * distill_loss + (1 - distill_weight) * lm_loss
        """
        import math

        if not self.distill_schedule_enabled:
            # Use fixed weight from lm_loss_weight config
            return 1.0 - self.lm_loss_weight

        # Handle warmup: keep initial weight constant
        if self.global_step < self.distill_warmup_steps:
            return self.distill_initial_weight

        # Calculate progress (0 to 1) after warmup
        effective_step = self.global_step - self.distill_warmup_steps
        total_decay_steps = max(self.max_steps - self.distill_warmup_steps, 1)
        progress = min(effective_step / total_decay_steps, 1.0)

        # Apply schedule
        if self.distill_schedule_type == "linear":
            # Linear decay from initial to final
            weight = self.distill_initial_weight + progress * (
                self.distill_final_weight - self.distill_initial_weight
            )
        elif self.distill_schedule_type == "cosine":
            # Cosine decay: smooth transition, slower at endpoints
            weight = self.distill_final_weight + 0.5 * (
                self.distill_initial_weight - self.distill_final_weight
            ) * (1 + math.cos(math.pi * progress))
        else:
            # Fallback to initial weight
            weight = self.distill_initial_weight

        return weight

    def train(self) -> dict[str, float]:
        """Training loop with token counting and early stopping.

        Note: No separate eval pass - uses training mean_layer_loss for early stopping
        since all data is unseen in streaming pre-training.
        """
        import time
        from tqdm import tqdm

        self.model.train()

        # Calculate max_steps from token count, but respect explicit max_steps override
        # Priority: explicit max_steps > calculated from total_tokens
        tokens_per_step = (
            self.config.batch_size
            * self.seq_length
            * self.gradient_accumulation_steps
            * self.world_size
        )
        calculated_max_steps = self.total_tokens // tokens_per_step

        # Check if max_steps was explicitly set in config (not default 10000)
        explicit_max_steps = getattr(self.config, "max_steps", None)
        if explicit_max_steps is not None:
            # Explicit max_steps takes priority
            self.max_steps = explicit_max_steps
            logger.info(f"Using explicit max_steps={explicit_max_steps} (calculated would be {calculated_max_steps})")
        elif hasattr(self.config, "total_tokens"):
            # Fall back to calculated from total_tokens
            self.max_steps = calculated_max_steps

        logger.info(
            f"Stage 1.9: Training for {self.max_steps} steps (~{self.total_tokens:,} tokens)"
        )

        accumulated_loss = 0.0
        num_accumulated = 0
        start_time = time.time()
        last_loss_dict = {}

        pbar = tqdm(
            total=self.max_steps,
            desc="Training",
            disable=self.rank != 0,
            initial=self.global_step,
        )

        data_iter = iter(self.train_dataloader)

        while self.global_step < self.max_steps:
            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Move to device
            batch = self._move_to_device(batch)

            # Forward pass
            loss_dict = self._forward_step(batch)
            last_loss_dict = loss_dict
            loss = loss_dict["loss"] / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            accumulated_loss += loss_dict["loss"].item()
            num_accumulated += 1

            # Optimizer step
            if num_accumulated >= self.gradient_accumulation_steps:
                # Gradient clipping
                if self.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clipping,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                # Step saliency curriculum
                if self.saliency_curriculum is not None:
                    self.saliency_curriculum.step()

                self.global_step += 1
                avg_loss = accumulated_loss / num_accumulated
                self.train_losses.append(avg_loss)

                # Get lr for logging
                lr = self.optimizer.param_groups[0]["lr"]

                # Logging
                if self.global_step % self.log_interval == 0 and self.rank == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0
                    mean_layer_loss = loss_dict["mean_layer_loss"].item()

                    log_msg = (
                        f"Step {self.global_step}/{self.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"MeanLayerLoss: {mean_layer_loss:.6f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f}"
                    )
                    logger.info(log_msg)

                    # Log curriculum progress if enabled (outside wandb block)
                    if self.saliency_curriculum is not None:
                        current_k = self.saliency_curriculum.get_current_k()
                        logger.info(f"Saliency curriculum: k={current_k:.4f}")

                    # Log distillation schedule progress
                    if self.distill_schedule_enabled:
                        current_distill = self._get_current_distill_weight()
                        logger.info(f"Distill schedule: weight={current_distill:.4f}")

                    # WandB logging
                    if self.wandb_enabled:
                        wandb_log = {
                            "train/loss": avg_loss,
                            "train/mean_layer_loss": mean_layer_loss,
                            "train/lr": lr,
                            "train/steps_per_sec": steps_per_sec,
                            "train/step": self.global_step,
                            "train/tokens_processed": self.tokens_processed,
                        }

                        # Add separate LM and distillation loss if using combined loss
                        if "lm_loss" in loss_dict:
                            wandb_log["train/lm_loss"] = loss_dict["lm_loss"].item()
                        if "distill_loss" in loss_dict:
                            wandb_log["train/distill_loss"] = loss_dict["distill_loss"].item()

                        # Add curriculum metrics to wandb
                        if self.saliency_curriculum is not None:
                            wandb_log["curriculum/saliency_k"] = self.saliency_curriculum.get_current_k()

                        # Add distillation schedule metrics
                        if self.distill_schedule_enabled:
                            current_distill_weight = self._get_current_distill_weight()
                            wandb_log["schedule/distill_weight"] = current_distill_weight
                            wandb_log["schedule/lm_weight"] = 1.0 - current_distill_weight

                        self.wandb.log(wandb_log, step=self.global_step)

                # Checkpointing (keep this, just skip eval)
                # NOTE: All ranks must call save_checkpoint for FSDP state dict gathering
                # (collective operation), but only rank 0 writes to disk
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                pbar.update(1)
                pbar.set_postfix({"loss": avg_loss, "lr": lr})

                accumulated_loss = 0.0
                num_accumulated = 0

        pbar.close()

        # Final metrics
        final_metrics = {
            "train_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "mean_layer_loss": last_loss_dict.get("mean_layer_loss", torch.tensor(0.0)).item(),
            "tokens_processed": self.tokens_processed,
            "stopped_early": self.es_stopped_early,
        }

        # Save final checkpoint (all ranks must participate for FSDP)
        self.save_checkpoint("final")
        if self.rank == 0:
            logger.info(f"Training complete! Final metrics: {final_metrics}")

        return final_metrics

    def save_checkpoint(self, name: str) -> None:
        """Save checkpoint including saliency curriculum state."""
        # Call parent save_checkpoint first
        super().save_checkpoint(name)

        # Save curriculum state if enabled
        if self.saliency_curriculum is not None and self.rank == 0:
            checkpoint_dir = self.output_dir / "checkpoints" / name
            curriculum_state = self.saliency_curriculum.state_dict()
            torch.save(curriculum_state, checkpoint_dir / "curriculum.pt")
            logger.debug(f"Saved curriculum state to {checkpoint_dir / 'curriculum.pt'}")

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint including saliency curriculum state."""
        # Call parent load_checkpoint first
        super().load_checkpoint(path)

        # Load curriculum state if it exists
        if self.saliency_curriculum is not None:
            if path.is_dir():
                curriculum_path = path / "curriculum.pt"
            else:
                curriculum_path = path.parent / "curriculum.pt"

            if curriculum_path.exists():
                curriculum_state = torch.load(curriculum_path, map_location=self.device)
                self.saliency_curriculum.load_state_dict(curriculum_state, device=self.device)
                logger.info(
                    f"Loaded curriculum state: step={self.saliency_curriculum._current_step}, "
                    f"k={self.saliency_curriculum.get_current_k():.4f}"
                )
            else:
                logger.warning(
                    f"Curriculum state not found at {curriculum_path}, "
                    "starting fresh curriculum tracking"
                )

    def _forward_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Forward step with hidden state distillation + optional cross-entropy.

        Extracts hidden states from both teacher and student models,
        then computes layer-wise distillation loss.

        If lm_loss_weight > 0, also computes cross-entropy loss to maintain
        language modeling capability during hidden state alignment.
        """
        import torch.nn.functional as F

        # Track tokens
        batch_tokens = batch["input_ids"].numel()
        self.tokens_processed += batch_tokens

        # Get teacher hidden states (and logits for LM loss if enabled)
        # Note: HiddenStateTeacherWrapper already enables output_hidden_states internally
        teacher_outputs = self.teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

        # Get student hidden states (and logits for LM loss)
        student_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
        )

        # Extract hidden states (skip embedding layer at index 0)
        # Both should be tuples of length (num_layers + 1)
        student_hidden = list(student_outputs["hidden_states"])[1:]  # Skip embedding
        teacher_hidden = list(teacher_outputs["hidden_states"])[1:]  # Skip embedding

        # Compute layerwise distillation loss
        loss_dict = self.loss_fn(
            student_hidden_states=student_hidden,
            teacher_hidden_states=teacher_hidden,
            attention_mask=batch.get("attention_mask"),
        )

        # Optionally add cross-entropy / logits distillation to maintain LM capability
        # Use scheduled distill weight if enabled, otherwise use fixed lm_loss_weight
        distill_weight = self._get_current_distill_weight()
        lm_weight = 1.0 - distill_weight

        if lm_weight > 0 and "labels" in batch:
            # Handle both dict-style and ModelOutput-style access
            if isinstance(student_outputs, dict):
                student_logits = student_outputs["logits"]
            elif hasattr(student_outputs, "logits"):
                student_logits = student_outputs.logits
            else:
                student_logits = student_outputs[0]
            if isinstance(teacher_outputs, dict):
                teacher_logits = teacher_outputs["logits"]
            elif hasattr(teacher_outputs, "logits"):
                teacher_logits = teacher_outputs.logits
            else:
                teacher_logits = teacher_outputs[0]
            labels = batch["labels"]

            # Shift for next-token prediction
            shift_student = student_logits[..., :-1, :].contiguous()
            shift_teacher = teacher_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 1. Cross-entropy loss against ground truth
            ce_loss = F.cross_entropy(
                shift_student.view(-1, shift_student.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # 2. KL divergence from teacher logits (soft distillation)
            # Use temperature scaling for smoother distributions
            temperature = 2.0  # Standard distillation temperature
            kl_loss = F.kl_div(
                F.log_softmax(shift_student / temperature, dim=-1).view(-1, shift_student.size(-1)),
                F.softmax(shift_teacher.detach() / temperature, dim=-1).view(-1, shift_teacher.size(-1)),
                reduction="batchmean",
            ) * (temperature ** 2)

            # Combined LM loss: 0.5 CE + 0.5 KL (balanced)
            lm_loss = 0.5 * ce_loss + 0.5 * kl_loss

            # Combine hidden state distill + LM losses using scheduled weights
            distill_loss = loss_dict["loss"]
            combined_loss = distill_weight * distill_loss + lm_weight * lm_loss
            loss_dict["loss"] = combined_loss
            loss_dict["distill_loss"] = distill_loss.detach()
            loss_dict["lm_loss"] = lm_loss.detach()
            loss_dict["ce_loss"] = ce_loss.detach()
            loss_dict["kl_loss"] = kl_loss.detach()
            loss_dict["distill_weight"] = torch.tensor(distill_weight, device=batch["input_ids"].device)

        # Add tracking metrics
        loss_dict["tokens_processed"] = torch.tensor(
            float(self.tokens_processed), device=batch["input_ids"].device
        )

        return loss_dict


def run_stage1_9(
    student_model: nn.Module,
    teacher_model_name: str,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    config: DictConfig,
    layerwise_config: DictConfig,
    output_dir: Path,
    resume_from: Optional[Path] = None,
    run_manager: Optional[Any] = None,
    experiment_name: Optional[str] = None,
) -> nn.Module:
    """
    Run Stage 1.9: Layer-wise distillation.

    Aligns BitNet hidden states with the original full-precision teacher model
    using configurable loss metrics (MSE, cosine, KL, etc.).

    Args:
        student_model: BitNet student model from Stage 1
        teacher_model_name: HuggingFace name/path for teacher
            (should be the original pre-quantized model)
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        config: Training configuration
        layerwise_config: Layer-wise distillation configuration
        output_dir: Output directory for checkpoints
        resume_from: Optional checkpoint to resume from
        run_manager: Optional RunManager for GCS checkpoint uploads
        experiment_name: Name for GCS checkpoint path (e.g., "bitdistill_smollm2_135m")

    Returns:
        Distilled student model with aligned hidden states
    """
    logger.info("=" * 60)
    logger.info("Stage 1.9: Starting layer-wise distillation")
    logger.info("=" * 60)

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load teacher model with hidden state support
    teacher = HiddenStateTeacherWrapper(
        model_name_or_path=teacher_model_name,
        device=device,
        load_in_fp16=layerwise_config.get("teacher_fp16", True),
        offload_to_cpu=layerwise_config.get("teacher_offload", False),
        load_in_4bit=layerwise_config.get("teacher_4bit", False),
        use_flash_attention=layerwise_config.get("teacher_flash_attention", False),  # Disabled until flash_attn added to Modal (issue #8)
    )

    # Convert to saliency-aware layers if curriculum is enabled
    curriculum_config = layerwise_config.get("saliency_curriculum", None)
    if curriculum_config is not None and curriculum_config.get("enabled", False):
        from wrinklefree.models.bitlinear import convert_bitlinear_to_saliency_aware

        student_model = convert_bitlinear_to_saliency_aware(student_model)
        logger.info("Converted BitLinear layers to SaliencyAwareBitLinear for curriculum training")

    # Move student to device with consistent bfloat16 dtype
    student_model = student_model.to(device=device, dtype=torch.bfloat16)

    # Enable gradient checkpointing for memory efficiency (40% VRAM reduction)
    gradient_ckpt_cfg = layerwise_config.get("gradient_checkpointing", None)
    if gradient_ckpt_cfg or layerwise_config.get("enable_gradient_checkpointing", False):
        if hasattr(student_model, "gradient_checkpointing_enable"):
            student_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for student model (40% VRAM reduction)")
        else:
            logger.warning("Model does not support gradient_checkpointing_enable()")

    # Apply torch.compile for performance (2.9x speedup on A40)
    # Only for single GPU - FSDP has its own compilation strategy
    torch_compile_cfg = getattr(config, "torch_compile", None)
    if torch_compile_cfg and getattr(torch_compile_cfg, "enabled", False) and world_size == 1:
        compile_mode = getattr(torch_compile_cfg, "mode", "reduce-overhead")
        fullgraph = getattr(torch_compile_cfg, "fullgraph", False)
        logger.info(f"Applying torch.compile with mode={compile_mode}, fullgraph={fullgraph}")
        student_model = torch.compile(student_model, mode=compile_mode, fullgraph=fullgraph)

    # Optional FSDP wrapping for multi-GPU
    if world_size > 1:
        from wrinklefree.models import BitNetDecoderLayer

        student_model = wrap_model_fsdp(
            student_model,
            transformer_layer_cls=BitNetDecoderLayer,
            sharding_strategy=config.distributed.fsdp.sharding_strategy,
            mixed_precision=config.distributed.fsdp.mixed_precision.enabled,
            activation_checkpointing=config.distributed.fsdp.activation_checkpointing.enabled,
        )

    # Create optimizer
    optimizer = create_optimizer(
        student_model,
        learning_rate=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        optimizer_type=config.training.optimizer.get("type", "adamw_8bit"),
    )

    # Calculate max steps - use explicit max_steps if set, otherwise from total_tokens
    tokens_per_step = (
        config.training.batch_size
        * config.training.max_seq_length
        * config.training.gradient_accumulation_steps
        * world_size
    )

    # Prefer explicit max_steps over calculated from total_tokens
    explicit_max_steps = getattr(config.training, "max_steps", None)
    if explicit_max_steps is not None:
        max_steps = explicit_max_steps
        total_tokens = max_steps * tokens_per_step
        logger.info(f"Training for {max_steps} steps (~{total_tokens:,} tokens)")
    else:
        total_tokens = getattr(config.training, "total_tokens", 100_000_000)
        max_steps = total_tokens // tokens_per_step
        logger.info(f"Training for {max_steps} steps ({total_tokens:,} tokens)")

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.training.scheduler.type,
        num_training_steps=max_steps,
        num_warmup_steps=config.training.scheduler.warmup_steps,
    )

    # Create trainer
    trainer = Stage19Trainer(
        model=student_model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config.training,
        layerwise_config=layerwise_config,
        device=device,
        rank=rank,
        world_size=world_size,
        run_manager=run_manager,
        experiment_name=experiment_name,
        stage="stage1_9",
    )
    trainer.output_dir = output_dir

    # Set total steps for saliency curriculum scheduler
    if trainer.saliency_curriculum is not None:
        trainer.saliency_curriculum.set_total_steps(max_steps)
        logger.info(f"Saliency curriculum configured for {max_steps} total steps")

    # Resume if specified
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)

    # Train
    metrics = trainer.train()

    logger.info("=" * 60)
    logger.info(f"Stage 1.9: Complete. Processed {trainer.tokens_processed:,} tokens")
    logger.info(f"Stage 1.9: Final metrics: {metrics}")
    logger.info("=" * 60)

    return student_model, trainer.train_losses
