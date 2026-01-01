"""Base training stage interface.

Provides abstract base class for all training stages with common functionality.
Each stage (pretrain, midtrain, posttrain) implements compute_loss differently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm


@dataclass
class StageConfig:
    """Configuration for a training stage.

    Reference: MobileLLM-R1 paper (arXiv:2509.24945) hyperparameters.
    """

    name: str
    num_steps: int = 0
    num_epochs: int = 0  # Alternative to num_steps for SFT
    batch_size_per_gpu: int = 16
    seq_len: int = 2048

    # Optimizer settings
    learning_rate: float = 4e-3
    weight_decay: float = 0.1

    # Scheduler settings
    scheduler_type: str = "linear_decay"  # linear_decay, linear_decay_to_zero, cosine
    warmup_steps: int = 2000
    warmup_ratio: float = 0.0  # Alternative to warmup_steps (for SFT)
    lr_decay_ratio: float = 0.1  # Final LR = initial LR * decay_ratio

    # Precision
    dtype: str = "bfloat16"
    use_fp8: bool = False

    # Gradient settings
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Gradient checkpointing for memory efficiency
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_mode: str = "quantized"  # "standard" or "quantized" (INT8)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    memory_allocated_gb: float = 0.0
    step: int = 0
    epoch: int = 0
    extra: dict = field(default_factory=dict)


class TrainingStage(ABC):
    """Abstract base class for training stages.

    Each training stage (pretrain, midtrain, posttrain) implements:
    - compute_loss(): Stage-specific loss computation
    - Optional: setup_dataloader(), setup_optimizer(), etc.

    The base class provides:
    - train_step(): Single training step with gradient accumulation
    - run(): Full training loop
    - save_checkpoint(), load_checkpoint()
    """

    def __init__(
        self,
        config: StageConfig,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        dataloader: DataLoader,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        influence_filter: Optional[Any] = None,
    ):
        """Initialize training stage.

        Args:
            config: Stage configuration
            model: Model to train
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            dataloader: Training data loader
            device: Device to train on
            rank: Process rank for distributed training
            world_size: Total number of processes
            influence_filter: Optional SelfBoostingFilter for data filtering
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.influence_filter = influence_filter

        self.global_step = 0
        self.epoch = 0
        self._accumulated_loss = 0.0
        self._accumulation_count = 0

        # Set up dtype
        self.dtype = getattr(torch, config.dtype, torch.bfloat16)

        # Gradient scaler for mixed precision
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=(config.dtype == "float16"))

    @abstractmethod
    def compute_loss(
        self,
        batch: dict[str, Tensor],
    ) -> Tuple[Tensor, dict[str, float]]:
        """Compute loss for a single batch.

        Args:
            batch: Dictionary containing input tensors

        Returns:
            Tuple of (loss tensor, metrics dictionary)
        """
        pass

    def train_step(self, batch: dict[str, Tensor]) -> Optional[TrainingMetrics]:
        """Execute a single training step with gradient accumulation.

        Args:
            batch: Input batch dictionary

        Returns:
            TrainingMetrics if step was executed (after accumulation), else None
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

        # Apply influence-based filtering if configured
        if self.influence_filter is not None:
            # Refresh filter cache periodically
            self.influence_filter.refresh(self.global_step, show_progress=False)
            # Filter batch to keep only high-influence samples
            batch, _ = self.influence_filter.filter_batch(batch)
            # Skip if batch is empty after filtering
            if batch["input_ids"].size(0) == 0:
                return None

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", dtype=self.dtype):
            loss, metrics = self.compute_loss(batch)
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        self.grad_scaler.scale(loss).backward()

        self._accumulated_loss += loss.item() * self.config.gradient_accumulation_steps
        self._accumulation_count += 1

        # Only step optimizer after accumulation
        if self._accumulation_count < self.config.gradient_accumulation_steps:
            return None

        # Gradient clipping
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_norm,
        )

        # Optimizer step
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()

        self.global_step += 1

        # Collect metrics
        training_metrics = TrainingMetrics(
            loss=self._accumulated_loss,
            learning_rate=self.scheduler.get_last_lr()[0],
            grad_norm=grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
            step=self.global_step,
            epoch=self.epoch,
            memory_allocated_gb=torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            extra=metrics,
        )

        # Reset accumulation
        self._accumulated_loss = 0.0
        self._accumulation_count = 0

        return training_metrics

    def run(
        self,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        log_interval: int = 100,
        checkpoint_interval: int = 1000,
        checkpoint_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
    ) -> Iterator[TrainingMetrics]:
        """Run the full training stage.

        Args:
            max_steps: Maximum steps (overrides config)
            max_epochs: Maximum epochs (overrides config)
            log_interval: How often to yield metrics
            checkpoint_interval: How often to call checkpoint callback
            checkpoint_callback: Function to call for checkpointing
            progress_callback: Function to call with progress updates

        Yields:
            TrainingMetrics for each logged step
        """
        max_steps = max_steps or self.config.num_steps
        max_epochs = max_epochs or self.config.num_epochs or float("inf")

        # Determine if we're step-based or epoch-based
        use_steps = max_steps > 0

        pbar = tqdm(
            total=max_steps if use_steps else None,
            desc=f"Training {self.config.name}",
            disable=self.rank != 0,
        )

        while self.epoch < max_epochs:
            self.epoch += 1

            for batch in self.dataloader:
                metrics = self.train_step(batch)

                if metrics is not None:
                    pbar.update(1)

                    # Log metrics
                    if self.global_step % log_interval == 0:
                        yield metrics

                    # Checkpoint
                    if checkpoint_callback and self.global_step % checkpoint_interval == 0:
                        checkpoint_callback(self, metrics)

                    # Progress callback
                    if progress_callback:
                        progress_callback(self, metrics)

                    # Check step limit
                    if use_steps and self.global_step >= max_steps:
                        pbar.close()
                        return

        pbar.close()

    def state_dict(self) -> dict:
        """Get state dictionary for checkpointing.

        Returns:
            Dictionary containing stage state
        """
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint.

        Args:
            state_dict: Checkpoint dictionary
        """
        self.global_step = state_dict["global_step"]
        self.epoch = state_dict["epoch"]
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        if "grad_scaler_state_dict" in state_dict:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler_state_dict"])
