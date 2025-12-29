"""Diffusion fine-tuning for BitNet to DLM conversion.

Fine-tunes an adapted BitNet model with block diffusion objectives,
requiring only ~1B tokens (500x less than training from scratch).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import PreTrainedTokenizer

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for diffusion fine-tuning."""

    # Token budget
    total_tokens: int = 1_000_000_000  # 1B tokens
    max_seq_length: int = 512

    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 8

    # Optimizer
    learning_rate: float = 5e-5  # Less conservative than 1e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # Scheduler
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # Diffusion
    mask_ratio: float = 0.15
    complementary_training: bool = True
    token_shift: bool = True

    # Block diffusion
    block_size: int = 32
    num_diffusion_steps: int = 8

    # Checkpointing
    save_interval: int = 5000
    keep_last_n: int = 3
    output_dir: str = "./outputs"

    # Logging
    log_interval: int = 100
    wandb_project: Optional[str] = "wrinklefree-dlm"
    wandb_run_name: Optional[str] = None

    # Compute
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = False

    def __post_init__(self):
        self.dtype_torch = getattr(torch, self.dtype)


class DiffusionFineTuner:
    """Fine-tune BitNet model with block diffusion objectives.

    Training approach (from Fast-dLLM v2):
    - Uses masked token prediction with complementary masks
    - Token shift mechanism for within-block dependencies
    - ~1B tokens total (500x less than training DLM from scratch)
    - Preserves BitLinear quantization during training

    Example:
        >>> finetuner = DiffusionFineTuner(model, config, tokenizer)
        >>> finetuner.train(dataloader, total_tokens=1_000_000_000)
        >>> finetuner.save_checkpoint("./outputs/dlm")
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        """Initialize the fine-tuner.

        Args:
            model: Adapted BitNet model (from BlockDiffusionAdapter)
            config: Training configuration
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        # Ensure model is on correct device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device, dtype=config.dtype_torch)

        # Get mask token ID
        if tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            # Add mask token if not present
            tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            self.mask_token_id = tokenizer.mask_token_id

        # Initialize optimizer and scheduler
        self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.tokens_seen = 0
        self.best_loss = float("inf")

        # Compile model if requested (PyTorch 2.0+)
        if config.compile_model:
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Get trainable parameters (skip frozen layers)
        params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

        # Calculate total steps
        tokens_per_step = (
            self.config.batch_size
            * self.config.max_seq_length
            * self.config.gradient_accumulation_steps
        )
        self.total_steps = self.config.total_tokens // tokens_per_step

        # Cosine scheduler with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * self.config.min_lr_ratio,
        )

        logger.info(
            f"Optimizer setup: {len(params)} param groups, "
            f"{self.total_steps} total steps, "
            f"LR={self.config.learning_rate}"
        )

    def create_diffusion_batch(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create training batch with masked tokens.

        Args:
            input_ids: Original input token IDs [B, L]

        Returns:
            masked_input: Input with some tokens replaced by [MASK]
            targets: Original tokens for prediction
            mask: Boolean mask indicating which tokens to predict
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create mask for tokens to predict
        if self.config.complementary_training:
            # Complementary mask based on step
            mask = self._create_complementary_mask(seq_len, device)
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        else:
            # Random mask
            rand = torch.rand(batch_size, seq_len, device=device)
            mask = rand < self.config.mask_ratio

        # Don't mask special tokens (BOS, EOS, PAD)
        special_mask = self._get_special_token_mask(input_ids)
        mask = mask & ~special_mask

        # Create masked input
        masked_input = input_ids.clone()
        masked_input[mask] = self.mask_token_id

        return masked_input, input_ids, mask

    def _create_complementary_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create complementary mask for current step."""
        positions = torch.arange(seq_len, device=device)
        pattern = self.global_step % self.config.num_diffusion_steps
        return (positions % self.config.num_diffusion_steps) == pattern

    def _get_special_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get mask for special tokens that shouldn't be masked."""
        special_ids = {
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }
        special_ids.discard(None)

        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in special_ids:
            mask |= input_ids == token_id

        return mask

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss only on masked positions.

        If token_shift is enabled, we predict each masked token from
        the preceding token's logits.

        Args:
            logits: Model output logits [B, L, V]
            targets: Target token IDs [B, L]
            mask: Boolean mask for loss positions [B, L]

        Returns:
            Scalar loss value
        """
        if self.config.token_shift:
            # Shift logits: predict position i from logits at i-1
            shifted_logits = torch.zeros_like(logits)
            shifted_logits[:, 1:] = logits[:, :-1]
            logits = shifted_logits

        # Flatten for cross-entropy
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Compute loss only on masked positions
        loss = torch.nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
        )

        # Apply mask
        masked_loss = loss * mask_flat.float()
        num_masked = mask_flat.sum().clamp(min=1)

        return masked_loss.sum() / num_masked

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step with gradient accumulation.

        Args:
            batch: Dictionary with 'input_ids' and 'attention_mask'

        Returns:
            Metrics dictionary
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Create masked batch
        masked_input, targets, predict_mask = self.create_diffusion_batch(input_ids)

        # Get noise embedding for current step
        if hasattr(self.model, "noise_embedding"):
            timestep = self.global_step % self.config.num_diffusion_steps
            noise_embed = self.model.noise_embedding(
                torch.tensor([timestep], device=self.device)
            )
        else:
            noise_embed = None

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=self.config.dtype_torch):
            outputs = self.model(
                input_ids=masked_input,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            # Compute loss
            loss = self.compute_loss(logits, targets, predict_mask)
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update tokens seen
        self.tokens_seen += input_ids.numel()

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "tokens": input_ids.numel(),
            "masked_ratio": predict_mask.float().mean().item(),
        }

    def train(
        self,
        dataloader: DataLoader,
        total_tokens: Optional[int] = None,
    ) -> nn.Module:
        """Run full fine-tuning loop.

        Args:
            dataloader: Training data loader
            total_tokens: Override total tokens from config

        Returns:
            Fine-tuned model
        """
        if total_tokens is not None:
            self.config.total_tokens = total_tokens

        # Initialize wandb
        if WANDB_AVAILABLE and self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )

        self.model.train()
        accumulation_counter = 0
        running_loss = 0.0

        logger.info(f"Starting training for {self.config.total_tokens:,} tokens")

        pbar = tqdm(total=self.config.total_tokens, desc="Training", unit="tok")
        pbar.update(self.tokens_seen)

        for epoch in range(1000):  # Large number, we exit on token budget
            for batch in dataloader:
                # Training step
                metrics = self.train_step(batch)
                running_loss += metrics["loss"]
                accumulation_counter += 1

                # Update progress
                pbar.update(metrics["tokens"])

                # Gradient step
                if accumulation_counter >= self.config.gradient_accumulation_steps:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Scheduler step (after warmup)
                    if self.global_step >= self.config.warmup_steps:
                        self.scheduler.step()

                    self.global_step += 1
                    accumulation_counter = 0

                    # Logging
                    if self.global_step % self.config.log_interval == 0:
                        avg_loss = running_loss / self.config.log_interval
                        lr = self.optimizer.param_groups[0]["lr"]

                        log_msg = (
                            f"Step {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Tokens: {self.tokens_seen:,}"
                        )
                        logger.info(log_msg)

                        if WANDB_AVAILABLE and self.config.wandb_project:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/lr": lr,
                                "train/tokens": self.tokens_seen,
                                "train/step": self.global_step,
                            })

                        running_loss = 0.0

                    # Checkpointing
                    if self.global_step % self.config.save_interval == 0:
                        self._save_checkpoint()

                # Check token budget
                if self.tokens_seen >= self.config.total_tokens:
                    logger.info(f"Reached token budget: {self.tokens_seen:,}")
                    pbar.close()
                    self._save_checkpoint(final=True)
                    return self.model

        pbar.close()
        return self.model

    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if final:
            ckpt_dir = output_dir / "final"
        else:
            ckpt_dir = output_dir / f"step_{self.global_step}"

        ckpt_dir.mkdir(exist_ok=True)

        # Save model
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": vars(self.config),
        }
        torch.save(state, ckpt_dir / "training_state.pt")

        logger.info(f"Saved checkpoint to {ckpt_dir}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints(output_dir)

    def _cleanup_checkpoints(self, output_dir: Path):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.name.startswith("step_")],
            key=lambda x: int(x.name.split("_")[1]),
        )

        for ckpt in checkpoints[: -self.config.keep_last_n]:
            import shutil

            shutil.rmtree(ckpt)
            logger.debug(f"Removed old checkpoint: {ckpt}")
