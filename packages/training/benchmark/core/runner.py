"""Benchmark runner using real Stage2 training infrastructure."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from benchmark.core.metrics import BenchmarkMetrics
from benchmark.core.memory import MemoryTracker, clear_memory

logger = logging.getLogger(__name__)


class _CombinedOptimizer:
    """Wrapper to combine multiple optimizers (e.g., Muon + AdamW).

    This is needed because Muon only optimizes 2D hidden layer parameters,
    while other parameters (embeddings, biases, norms) should use AdamW.
    """

    def __init__(self, optimizers: list):
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            # Muon doesn't accept closure argument - don't even pass None
            if closure is not None:
                try:
                    opt.step(closure)
                except TypeError:
                    opt.step()
            else:
                opt.step()

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state_dict in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state_dict)


@dataclass
class RunnerConfig:
    """Configuration for benchmark runner."""

    warmup_steps: int = 10
    measurement_steps: int = 500
    sequence_length: int = 512
    device: str = "cuda"
    dtype: str = "float32"
    target_memory_gb: float = 35.0  # Target memory usage for all trials (lower due to large model)
    memory_buffer: float = 0.85  # Use 85% of target to leave headroom


class BenchmarkRunner:
    """Runs benchmark trials using real Stage2 training.

    Uses existing Trainer infrastructure from wrinklefree.training.
    """

    def __init__(
        self,
        base_config: Optional[DictConfig] = None,
        runner_config: Optional[RunnerConfig] = None,
        model_name: str = "Qwen/Qwen3-4B-Base",
        stage1_checkpoint: Optional[Path] = None,
    ):
        self.base_config = base_config or OmegaConf.create({})
        self.runner_config = runner_config or RunnerConfig()
        self.device = torch.device(self.runner_config.device)
        self.memory_tracker = MemoryTracker(self.device)
        self.model_name = model_name
        self.stage1_checkpoint = stage1_checkpoint

        # Load tokenizer for the actual model
        logger.info(f"Loading tokenizer for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _find_optimal_batch_size(
        self,
        model: nn.Module,
        optimizer_type: str,
        target_memory_gb: float,
    ) -> int:
        """Find the largest batch size that fits within target memory.

        This ensures all trials use approximately the same memory,
        making fair comparisons between optimizers.

        Args:
            model: The model to profile
            optimizer_type: Type of optimizer (affects memory)
            target_memory_gb: Target memory usage in GB

        Returns:
            Optimal batch size
        """
        clear_memory()
        torch.cuda.reset_peak_memory_stats()

        # Binary search for optimal batch size
        min_bs, max_bs = 1, 128
        optimal_bs = 1

        while min_bs <= max_bs:
            mid_bs = (min_bs + max_bs) // 2

            try:
                clear_memory()
                torch.cuda.reset_peak_memory_stats()

                # Create a test batch
                test_input = torch.randint(
                    0, 50257, (mid_bs, self.runner_config.sequence_length),
                    device=self.device
                )

                # Forward pass with autocast for mixed precision
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(test_input)

                # Check memory usage
                peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

                # Training typically uses 3-4x forward pass memory
                # (activations, gradients, optimizer states)
                multiplier = 3.5 if optimizer_type == "muon" else 4.0
                estimated_train_mem = peak_mem_gb * multiplier

                if estimated_train_mem <= target_memory_gb * self.runner_config.memory_buffer:
                    optimal_bs = mid_bs
                    min_bs = mid_bs + 1
                else:
                    max_bs = mid_bs - 1

                del test_input
                clear_memory()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    max_bs = mid_bs - 1
                    clear_memory()
                else:
                    raise

        logger.info(f"Auto-selected batch_size={optimal_bs} for {optimizer_type} (target: {target_memory_gb}GB)")
        return max(optimal_bs, 1)  # At least 1

    def run_trial(
        self,
        trial_params: dict[str, Any],
        trial_id: int = 0,
    ) -> BenchmarkMetrics:
        """Run a benchmark trial using real Stage2 training.

        Args:
            trial_params: Parameters from Ax including optimizer_type, lr, batch_size, etc.
            trial_id: Trial identifier

        Returns:
            BenchmarkMetrics with convergence and throughput measurements
        """
        logger.info(f"Trial {trial_id}: {trial_params}")

        clear_memory()
        self.memory_tracker.reset()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Build model
            model = self._build_model().to(self.device)

            # Auto-calculate batch size if not specified or set to "auto"
            optimizer_type = trial_params.get("optimizer_type", "muon")
            if trial_params.get("batch_size") is None or trial_params.get("batch_size") == "auto":
                batch_size = self._find_optimal_batch_size(
                    model, optimizer_type, self.runner_config.target_memory_gb
                )
                trial_params = {**trial_params, "batch_size": batch_size}
            else:
                batch_size = trial_params.get("batch_size", 16)

            logger.info(f"Using batch_size={batch_size} for optimizer={optimizer_type}")

            # Create optimizer
            optimizer = self._create_optimizer(model, trial_params)

            # Create dataloader using existing infrastructure
            dataloader = self._create_dataloader(trial_params)

            # Build trainer config
            config = self._build_trainer_config(trial_params)

            # Create trainer using existing infrastructure
            # Use loss_fn=None so model computes loss internally
            from wrinklefree.training.trainer import Trainer

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=None,  # Model computes cross-entropy loss when labels passed
                train_dataloader=dataloader,
                config=config,
                device=self.device,
                rank=0,
                world_size=1,
            )

            # Run training
            start_time = time.time()
            trainer.train()
            wall_time = time.time() - start_time

            # Extract metrics
            initial_loss = trainer.train_losses[0] if trainer.train_losses else 0.0
            final_loss = trainer.train_losses[-1] if trainer.train_losses else 0.0
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
            allocated_memory_gb = torch.cuda.memory_allocated() / 1e9

            # Compute throughput
            batch_size = trial_params.get("batch_size", 8)
            seq_len = self.runner_config.sequence_length
            num_steps = self.runner_config.measurement_steps
            total_tokens = num_steps * batch_size * seq_len
            throughput = total_tokens / wall_time

            metrics = BenchmarkMetrics.compute(
                throughput_tokens_per_sec=throughput,
                peak_memory_gb=peak_memory_gb,
                allocated_memory_gb=allocated_memory_gb,
                final_loss=final_loss,
                initial_loss=initial_loss,
                num_steps=num_steps,
                grad_norms=[],  # Trainer doesn't expose these
                optimizer_type=trial_params.get("optimizer_type", "apollo"),
                batch_size=batch_size,
                learning_rate=trial_params.get("learning_rate", 1e-4),
                gradient_accumulation_steps=trial_params.get("gradient_accumulation_steps", 1),
                wall_time_seconds=wall_time,
                trial_id=trial_id,
                lambda_logits=trial_params.get("lambda_logits"),
                gamma_attention=trial_params.get("gamma_attention"),
                temperature=trial_params.get("temperature"),
                influence_enabled=trial_params.get("influence_enabled", False),
                influence_lambda_reg=trial_params.get("influence_lambda_reg"),
                influence_threshold=trial_params.get("influence_threshold"),
            )

            logger.info(
                f"Trial {trial_id}: convergence={metrics.convergence_per_sec_per_gb:.4f}, "
                f"loss {initial_loss:.4f} -> {final_loss:.4f}"
            )
            return metrics

        finally:
            if "model" in locals():
                del model
            if "optimizer" in locals():
                del optimizer
            clear_memory()

    def _build_model(self) -> nn.Module:
        """Build BitNet model from Stage 1 checkpoint."""
        from transformers import AutoModelForCausalLM
        from wrinklefree.training.stage1 import convert_model_to_bitnet

        if self.stage1_checkpoint and self.stage1_checkpoint.exists():
            # Load from Stage 1 checkpoint
            logger.info(f"Loading Stage 1 checkpoint from {self.stage1_checkpoint}")

            # First load the base model (use bfloat16 to save memory)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Get model dimensions from config
            hidden_size = model.config.hidden_size
            intermediate_size = model.config.intermediate_size

            # Convert to BitNet architecture
            model = convert_model_to_bitnet(model, hidden_size, intermediate_size)

            # Load Stage 1 weights
            from safetensors.torch import load_file
            state_dict = load_file(self.stage1_checkpoint / "model.safetensors")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded Stage 1: {len(missing)} missing, {len(unexpected)} unexpected keys")

            # Enable gradient checkpointing to save memory
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

            # Ensure entire model is in bfloat16
            model = model.to(torch.bfloat16)

            return model
        else:
            # Run Stage 1 conversion on-the-fly
            logger.info(f"No Stage 1 checkpoint, converting {self.model_name} to BitNet on-the-fly")
            from wrinklefree.training.stage1 import run_stage1

            model, _ = run_stage1(
                pretrained_model_name=self.model_name,
                output_dir=Path("./outputs/stage1_checkpoint"),
            )
            return model

    def _build_trainer_config(self, trial_params: dict[str, Any]) -> DictConfig:
        """Build config for Trainer."""
        config = {
            "max_steps": self.runner_config.measurement_steps,
            "batch_size": trial_params.get("batch_size", 8),
            "gradient_accumulation_steps": trial_params.get("gradient_accumulation_steps", 1),
            "gradient_clipping": 1.0,
            "log_interval": 100,
            "eval_interval": 99999,  # Disable eval
            "save_interval": 99999,  # Disable checkpointing
        }

        # Pass wandb config from base_config if available
        if self.base_config and hasattr(self.base_config, "logging"):
            config["logging"] = OmegaConf.to_container(self.base_config.logging)

        return OmegaConf.create(config)

    def _create_dataloader(self, trial_params: dict[str, Any]):
        """Create dataloader using MixedDataset with influence."""
        batch_size = trial_params.get("batch_size", 8)
        influence_enabled = trial_params.get("influence_enabled", True)

        if influence_enabled:
            # Use MixedDataset with influence-based data selection
            from wrinklefree.data.mixed_dataset import create_mixed_dataloader

            return create_mixed_dataloader(
                sources=[
                    {"path": "HuggingFaceFW/fineweb-edu", "name": "fineweb-edu", "subset": "sample-10BT", "weight": 0.7},
                    {"path": "allenai/c4", "name": "c4-en", "subset": "en", "weight": 0.3},
                ],
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                max_length=self.runner_config.sequence_length,
                num_workers=2,
            )
        else:
            # Simple pretrain dataloader without influence
            from wrinklefree.data.pretrain_dataset import create_pretrain_dataloader

            return create_pretrain_dataloader(
                dataset_path="HuggingFaceFW/fineweb-edu",
                dataset_name="sample-10BT",
                tokenizer=self.tokenizer,
                batch_size=batch_size,
                max_length=self.runner_config.sequence_length,
                num_workers=2,
                seed=42,
                packed=True,
            )

    def _create_optimizer(
        self,
        model: nn.Module,
        trial_params: dict[str, Any],
    ) -> torch.optim.Optimizer:
        """Create optimizer based on trial parameters."""
        optimizer_type = trial_params.get("optimizer_type", "muon")
        learning_rate = trial_params.get("learning_rate", 1e-4)
        weight_decay = trial_params.get("weight_decay", 0.1)

        # Separate params for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if optimizer_type == "muon":
            # Use official torch.optim.Muon (PyTorch 2.9+)
            muon_params = []
            adamw_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(p in name.lower() for p in ["embed", "head", "lm_head", "bias", "norm", "ln_"]):
                    adamw_params.append(param)
                elif param.ndim >= 2:
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            # Official PyTorch Muon for 2D hidden layer params
            muon_opt = torch.optim.Muon(muon_params, lr=learning_rate, momentum=0.95, nesterov=True)
            adamw_opt = torch.optim.AdamW(adamw_params, lr=learning_rate * 0.1, betas=(0.9, 0.95), weight_decay=weight_decay)
            return _CombinedOptimizer([muon_opt, adamw_opt])

        if optimizer_type == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(param_groups, lr=learning_rate, betas=(0.9, 0.95))
            except ImportError as e:
                raise RuntimeError(f"bitsandbytes not available: {e}") from e

        if optimizer_type in ("apollo", "apollo_mini"):
            try:
                from apollo_torch import APOLLOAdamW
                return APOLLOAdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95), scale_front=(optimizer_type == "apollo_mini"))
            except ImportError as e:
                raise RuntimeError(f"apollo-torch not available: {e}") from e

        return torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))


def run_single_benchmark(
    trial_params: dict[str, Any],
    runner_config: Optional[RunnerConfig] = None,
    base_config: Optional[DictConfig] = None,
) -> BenchmarkMetrics:
    """Convenience function to run a single benchmark trial."""
    runner = BenchmarkRunner(base_config=base_config, runner_config=runner_config)
    return runner.run_trial(trial_params)
