"""Distillation trainer for knowledge distillation."""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from distillation.losses import BitDistillLoss, TCSDistillLoss
from distillation.teachers.base import BaseTeacher
from distillation.training.config import DistillationConfig, LossConfig

logger = logging.getLogger(__name__)


def apply_dlm_masking(
    input_ids: torch.Tensor,
    mask_token_id: int,
    mask_prob: float = 0.15,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random masking for DLM training.

    Args:
        input_ids: Original input IDs [batch, seq_len]
        mask_token_id: Token ID to use for [MASK]
        mask_prob: Probability of masking each token (default 15%)
        ignore_index: Label value for non-masked positions

    Returns:
        masked_input_ids: Input with random tokens replaced by [MASK]
        mask_labels: Labels for masked positions (-100 for non-masked)
    """
    masked_input_ids = input_ids.clone()
    mask_labels = torch.full_like(input_ids, ignore_index)

    # Create random mask
    mask = torch.rand_like(input_ids.float()) < mask_prob

    # Don't mask padding tokens (token_id = 0 typically)
    # Also don't mask first/last tokens for safety
    mask[:, 0] = False
    mask[:, -1] = False

    # Apply masking
    masked_input_ids[mask] = mask_token_id
    mask_labels[mask] = input_ids[mask]

    return masked_input_ids, mask_labels




class DistillationTrainer:
    """
    Trainer for knowledge distillation.

    Features:
    - Flexible teacher backends (local, vLLM)
    - Toggle-able attention distillation
    - Integration with cheapertraining influence for dataset rebalancing
    - Support for resuming from student checkpoint

    Args:
        student: Student model to train
        teacher: Teacher model (frozen)
        train_dataloader: Training data loader
        config: Distillation configuration
        eval_dataloader: Optional evaluation data loader
        mixed_dataset: Optional MixedDataset for influence updates
        device: Device to train on
        rank: Process rank for distributed training
        world_size: Total number of processes
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: BaseTeacher,
        train_dataloader: DataLoader,
        config: DistillationConfig,
        eval_dataloader: Optional[DataLoader] = None,
        mixed_dataset: Optional[Any] = None,  # MixedDataset from data_handler
        device: torch.device = torch.device("cuda"),
        rank: int = 0,
        world_size: int = 1,
        mask_token_id: Optional[int] = None,  # For DLM masking
    ):
        self.student = student
        self.teacher = teacher
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.mixed_dataset = mixed_dataset
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.mask_token_id = mask_token_id

        # DLM masking probability (standard 15%)
        self.mask_prob = 0.15

        # Create loss function based on student type
        if config.student_type == "dlm":
            # TCS loss for DLM students (block-wise attention distillation)
            self.loss_fn = TCSDistillLoss(
                lambda_tcs=config.loss.lambda_logits,
                gamma_attention=config.loss.gamma_attention,
                temperature=config.loss.temperature,
                top_k=config.loss.top_k,
                block_size=config.loss.block_size,
                distill_layer=config.loss.distill_layer,
            )
            logger.info(
                f"Using TCSDistillLoss for DLM student (block_size={config.loss.block_size}, "
                f"gamma_attention={config.loss.gamma_attention})"
            )
        else:
            # Standard BitDistillLoss for AR models
            self.loss_fn = BitDistillLoss(
                lambda_logits=config.loss.lambda_logits,
                gamma_attention=config.loss.gamma_attention,
                temperature=config.loss.temperature,
                use_relation_distill=config.loss.use_relation_distill,
                distill_layer=config.loss.distill_layer,
            )
            logger.info("Using BitDistillLoss for BitNet student")

        # Whether to output attentions (only needed if gamma > 0)
        self.output_attentions = config.loss.gamma_attention > 0

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.train_losses = []
        self.eval_losses = []

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # WandB setup
        self.wandb = None
        self.wandb_enabled = False
        if self.rank == 0 and config.wandb_enabled:
            self._setup_wandb()

        # Influence components (initialized lazily)
        self._influence_calculator = None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config.

        Supports:
        - muonclip: Muon + QK-clipping for training stability (recommended)
        - adamw_8bit: 8-bit AdamW via bitsandbytes (memory efficient)
        - adamw: Standard AdamW
        """
        optimizer_type = self.config.optimizer_type.lower()

        if optimizer_type == "muonclip" or optimizer_type == "muon":
            from muon import MuonClip, MuonConfig

            # Get model config if available (for QK-clipping)
            model_config = getattr(self.student, "config", None)

            # Use separate learning rates: Muon for main weights, Adam for bias/norm
            # This follows the recommended pattern from muon-clip library
            config = MuonConfig(
                unified_lr=False,  # Use separate LRs for Muon and Adam
                lr_muon=self.config.lr_muon,  # For main weights (~1e-3)
                lr_adam=self.config.lr_adam,  # For bias/norm (~5e-5)
                muon_beta=0.95,
                muon_decay=self.config.weight_decay,
                adam_betas=(0.9, 0.95),
                adam_eps=1e-8,
                adam_decay=self.config.weight_decay,
                enable_clipping=False,  # Disable QK-clipping for now
                log_dir="",  # Empty string triggers writer creation (muon-clip bug workaround)
            )

            logger.info(
                f"Using MuonClip optimizer (lr_muon={self.config.lr_muon}, "
                f"lr_adam={self.config.lr_adam}, weight_decay={self.config.weight_decay})"
            )

            return MuonClip(self.student, model_config, config)

        if optimizer_type == "adamw_8bit":
            try:
                import bitsandbytes as bnb

                # Separate params with/without weight decay
                decay_params = []
                no_decay_params = []
                for name, param in self.student.named_parameters():
                    if not param.requires_grad:
                        continue
                    if "bias" in name or "norm" in name or "ln" in name:
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

                return bnb.optim.AdamW8bit(
                    [
                        {"params": decay_params, "weight_decay": self.config.weight_decay},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ],
                    lr=self.config.learning_rate,
                )
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to AdamW")

        # Standard AdamW with separate param groups
        decay_params = []
        no_decay_params = []
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Calculate warmup steps: use explicit value or default to 3% of max_steps
        warmup_steps = self.config.warmup_steps
        if warmup_steps <= 0:
            warmup_steps = int(self.config.max_steps * self.config.warmup_ratio)
            logger.info(f"Using warmup_ratio={self.config.warmup_ratio} -> {warmup_steps} warmup steps")

        if self.config.scheduler_type == "cosine":
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps - warmup_steps,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            # Linear warmup only
            return LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )

    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb as wandb_module
            import os
            from datetime import datetime

            api_key = os.environ.get("WANDB_API_KEY")
            if not api_key:
                # VERY LOUD WARNING - WandB is critical for monitoring training!
                warning_msg = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ██╗    ██╗ █████╗ ██████╗ ███╗   ██╗██╗███╗   ██╗ ██████╗ ██╗              ║
║  ██║    ██║██╔══██╗██╔══██╗████╗  ██║██║████╗  ██║██╔════╝ ██║              ║
║  ██║ █╗ ██║███████║██████╔╝██╔██╗ ██║██║██╔██╗ ██║██║  ███╗██║              ║
║  ██║███╗██║██╔══██║██╔══██╗██║╚██╗██║██║██║╚██╗██║██║   ██║╚═╝              ║
║  ╚███╔███╔╝██║  ██║██║  ██║██║ ╚████║██║██║ ╚████║╚██████╔╝██╗              ║
║   ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   WANDB_API_KEY NOT SET - WANDB LOGGING DISABLED!                           ║
║                                                                              ║
║   You will NOT be able to monitor training losses in real-time!             ║
║   This makes it impossible to detect training instabilities early.          ║
║                                                                              ║
║   To fix: Add --env WANDB_API_KEY=<your-key> when launching the job         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
                print(warning_msg)
                logger.warning("WANDB_API_KEY not set, disabling wandb")
                return

            self.wandb = wandb_module

            # Generate meaningful run name if not provided
            run_name = self.config.run_name
            if not run_name:
                # Extract model name from checkpoint path
                checkpoint_path = self.config.student_checkpoint_path
                if checkpoint_path:
                    path_parts = checkpoint_path.rstrip('/').split('/')
                    # Try to find a meaningful model identifier
                    model_id = "unknown"
                    for part in reversed(path_parts):
                        if 'bitnet' in part.lower() or 'dlm' in part.lower() or 'checkpoint' in part.lower():
                            model_id = part
                            break
                        if part and not part.startswith('gs:'):
                            model_id = part
                            break
                else:
                    model_id = "unknown"

                # Format: {student_type}-{model_id}-{timestamp}
                timestamp = datetime.now().strftime("%m%d_%H%M")
                run_name = f"{self.config.student_type}-{model_id}-{timestamp}"

            self.wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "student_type": self.config.student_type,
                    "student_checkpoint": self.config.student_checkpoint_path,
                    "loss": {
                        "lambda_logits": self.config.loss.lambda_logits,
                        "gamma_attention": self.config.loss.gamma_attention,
                        "temperature": self.config.loss.temperature,
                        "top_k": self.config.loss.top_k,
                        "block_size": self.config.loss.block_size,
                    },
                    "training": {
                        "max_steps": self.config.max_steps,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                        "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                        "optimizer": self.config.optimizer_type,
                    },
                },
            )
            self.wandb_enabled = True
            logger.info(f"WandB initialized: {run_name}")

        except ImportError:
            logger.warning("wandb not installed, disabling logging")

    def train(self) -> dict:
        """
        Run the distillation training loop.

        Returns:
            Dictionary of final training metrics
        """
        logger.info(f"Starting distillation training for {self.config.max_steps} steps")

        self.student.train()
        data_iter = iter(self.train_dataloader)

        accumulated_loss = 0.0
        step_start_time = time.time()

        # max_steps is NUMBER OF OPTIMIZER STEPS
        # Total forward steps = max_steps * gradient_accumulation_steps
        total_forward_steps = self.config.max_steps * self.config.gradient_accumulation_steps
        start_forward_step = self.global_step * self.config.gradient_accumulation_steps

        pbar = tqdm(
            range(start_forward_step, total_forward_steps),
            desc="Distilling",
            disable=self.rank != 0,
        )

        for step in pbar:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            loss_dict = self._forward_step(batch)
            loss = loss_dict["loss"] / self.config.gradient_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.config.gradient_clipping,
                    )

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                # Log metrics
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(
                        accumulated_loss,
                        loss_dict,
                        step_start_time,
                        pbar,
                    )
                    accumulated_loss = 0.0
                    step_start_time = time.time()

                # Evaluate
                if self.eval_dataloader is not None and self.global_step % self.config.eval_interval == 0:
                    self._evaluate()

                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint()

                # Update influence weights
                if (
                    self.config.influence_enabled
                    and self.mixed_dataset is not None
                    and self.global_step % self.config.influence_update_interval == 0
                ):
                    self._update_influence_weights()

                # Check if done
                if self.global_step >= self.config.max_steps:
                    break

        # Final save
        self._save_checkpoint(final=True)

        logger.info(f"Distillation complete. Final step: {self.global_step}")

        return {
            "final_step": self.global_step,
            "final_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "best_eval_loss": self.best_eval_loss,
        }

    def _forward_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward step with teacher distillation.

        For DLM students:
        - Student receives masked input and predicts original tokens
        - Teacher receives original input (AR: predicts next token)
        - Logits are aligned: teacher[:, :-1] corresponds to student[:, 1:]
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # DLM-specific handling
        if self.config.student_type == "dlm" and self.mask_token_id is not None:
            # Apply masking to student input
            masked_input_ids, mask_labels = apply_dlm_masking(
                input_ids,
                mask_token_id=self.mask_token_id,
                mask_prob=self.mask_prob,
            )

            # Teacher gets original (unmasked) input
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=self.output_attentions,
                )

            # Student gets masked input
            student_outputs = self.student(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                output_attentions=self.output_attentions,
            )

            # Handle different output formats
            if isinstance(student_outputs, dict):
                student_logits = student_outputs["logits"]
                student_attentions = student_outputs.get("attentions")
            else:
                student_logits = student_outputs.logits
                student_attentions = getattr(student_outputs, "attentions", None)

            teacher_logits = teacher_outputs["logits"]
            teacher_attentions = teacher_outputs.get("attentions")

            # CRITICAL: Align logits for AR teacher -> DLM student
            # AR teacher at position i predicts token i+1
            # DLM student at position i predicts token i (reconstruction)
            # So we align: teacher[:, i] with student[:, i+1]
            # i.e., teacher[:, :-1] corresponds to student[:, 1:]
            aligned_teacher_logits = teacher_logits[:, :-1, :].contiguous()
            aligned_student_logits = student_logits[:, 1:, :].contiguous()
            aligned_labels = mask_labels[:, 1:].contiguous()

            # Align attention masks if present
            aligned_attention_mask = None
            if attention_mask is not None:
                aligned_attention_mask = attention_mask[:, 1:].contiguous()

            # Compute distillation loss with aligned tensors
            loss_dict = self.loss_fn(
                student_logits=aligned_student_logits,
                teacher_logits=aligned_teacher_logits,
                student_attentions=student_attentions,
                teacher_attentions=teacher_attentions,
                labels=aligned_labels,
                attention_mask=aligned_attention_mask,
            )

        else:
            # Standard AR distillation (non-DLM)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=self.output_attentions,
                )

            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=self.output_attentions,
            )

            # Handle different output formats
            if isinstance(student_outputs, dict):
                student_logits = student_outputs["logits"]
                student_attentions = student_outputs.get("attentions")
            else:
                student_logits = student_outputs.logits
                student_attentions = getattr(student_outputs, "attentions", None)

            loss_dict = self.loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_outputs["logits"],
                student_attentions=student_attentions,
                teacher_attentions=teacher_outputs.get("attentions"),
                labels=batch["labels"],
                attention_mask=attention_mask,
            )

        return loss_dict

    def _log_metrics(
        self,
        accumulated_loss: float,
        loss_dict: dict,
        step_start_time: float,
        pbar: tqdm,
    ):
        """Log training metrics."""
        elapsed = time.time() - step_start_time
        steps_per_sec = self.config.log_interval / elapsed if elapsed > 0 else 0

        lr = self.optimizer.param_groups[0]["lr"]

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{accumulated_loss:.4f}",
            "lr": f"{lr:.2e}",
            "s/step": f"{1/steps_per_sec:.2f}" if steps_per_sec > 0 else "N/A",
        })

        self.train_losses.append(accumulated_loss)

        # Log to wandb
        if self.wandb_enabled:
            log_dict = {
                "train/loss": accumulated_loss,
                "train/ce_loss": loss_dict["ce_loss"].item(),
                "train/learning_rate": lr,
                "train/steps_per_sec": steps_per_sec,
                "train/epoch": self.epoch,
            }

            # Handle both BitDistill and TCS loss keys
            if "logits_distill_loss" in loss_dict:
                log_dict["train/logits_distill_loss"] = loss_dict["logits_distill_loss"].item()
            if "tcs_loss" in loss_dict:
                log_dict["train/tcs_loss"] = loss_dict["tcs_loss"].item()
            if "attention_distill_loss" in loss_dict:
                log_dict["train/attention_distill_loss"] = loss_dict["attention_distill_loss"].item()
            if "attention_loss" in loss_dict:
                log_dict["train/attention_loss"] = loss_dict["attention_loss"].item()

            self.wandb.log(log_dict, step=self.global_step)

    def _evaluate(self):
        """Run evaluation on eval dataset."""
        if self.eval_dataloader is None:
            return

        self.student.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss_dict = self._forward_step(batch)
                total_loss += loss_dict["loss"].item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.eval_losses.append(avg_loss)

        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            logger.info(f"New best eval loss: {avg_loss:.4f}")

        if self.wandb_enabled:
            self.wandb.log({
                "eval/loss": avg_loss,
                "eval/best_loss": self.best_eval_loss,
            }, step=self.global_step)

        self.student.train()

        logger.info(f"Eval loss: {avg_loss:.4f}")

    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        if self.rank != 0:
            return

        name = "final" if final else f"step_{self.global_step}"
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model state dict (handle FSDP if present)
        if hasattr(self.student, "module"):
            model_state = self.student.module.state_dict()
        else:
            model_state = self.student.state_dict()

        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "config": {
                "lambda_logits": self.config.loss.lambda_logits,
                "gamma_attention": self.config.loss.gamma_attention,
                "temperature": self.config.loss.temperature,
            },
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        # Upload to GCS if configured
        if self.config.gcs.enabled and self.global_step % self.config.gcs.upload_interval == 0:
            self._upload_to_gcs(checkpoint_dir, name)
            self._cleanup_old_gcs_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the latest N."""
        checkpoints_dir = self.output_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return

        # List all step checkpoints (exclude 'final')
        step_dirs = sorted(
            [d for d in checkpoints_dir.iterdir() if d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )

        # Keep only latest N
        while len(step_dirs) > self.config.keep_last_n:
            old_dir = step_dirs.pop(0)
            import shutil
            shutil.rmtree(old_dir)
            logger.debug(f"Removed old checkpoint: {old_dir}")

    def _upload_to_gcs(self, local_path: Path, checkpoint_name: str) -> bool:
        """Upload checkpoint to GCS if configured.

        Returns True if upload succeeded, False otherwise.
        """
        if not self.config.gcs.enabled:
            return False

        gcs_bucket = self.config.gcs.bucket
        experiment_name = Path(self.config.student_checkpoint_path).stem

        gcs_path = f"gs://{gcs_bucket}/distillation/{experiment_name}/{checkpoint_name}/"
        logger.info(f"Uploading checkpoint to {gcs_path}")

        try:
            result = subprocess.run(
                ["gsutil", "-m", "cp", "-r", f"{local_path}/*", gcs_path],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"GCS upload complete: {gcs_path}")
                return True
            else:
                logger.warning(f"gsutil failed: {result.stderr}")
                # Try gcloud storage as fallback
                result = subprocess.run(
                    ["gcloud", "storage", "cp", "-r", f"{local_path}/*", gcs_path],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    logger.info(f"GCS upload complete (gcloud): {gcs_path}")
                    return True
                logger.warning(f"gcloud storage also failed: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.warning("gsutil/gcloud not found, skipping GCS upload")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("GCS upload timed out")
            return False

    def _cleanup_old_gcs_checkpoints(self):
        """Delete old checkpoints from GCS, keeping only the N most recent."""
        if not self.config.gcs.enabled:
            return

        gcs_bucket = self.config.gcs.bucket
        keep_n = self.config.gcs.keep_n
        experiment_name = Path(self.config.student_checkpoint_path).stem

        gcs_prefix = f"gs://{gcs_bucket}/distillation/{experiment_name}/"
        try:
            result = subprocess.run(
                ["gsutil", "ls", "-d", f"{gcs_prefix}step_*"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                result = subprocess.run(
                    ["gcloud", "storage", "ls", f"{gcs_prefix}step_*"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            if result.returncode != 0 or not result.stdout:
                return

            checkpoints = result.stdout.strip().split("\n")
            step_checkpoints = []
            for cp in checkpoints:
                cp = cp.rstrip("/")
                if "step_" in cp:
                    try:
                        step = int(cp.split("step_")[-1])
                        step_checkpoints.append((step, cp))
                    except ValueError:
                        continue

            step_checkpoints.sort(key=lambda x: x[0], reverse=True)
            to_delete = step_checkpoints[keep_n:]

            if not to_delete:
                return

            logger.info(f"Cleaning up {len(to_delete)} old GCS checkpoints")
            for step, cp_path in to_delete:
                try:
                    delete_path = cp_path if cp_path.endswith("/") else cp_path + "/"
                    result = subprocess.run(
                        ["gsutil", "-m", "rm", "-r", delete_path],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode == 0:
                        logger.debug(f"Deleted old GCS checkpoint: step_{step}")
                    else:
                        subprocess.run(
                            ["gcloud", "storage", "rm", "-r", delete_path],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                except Exception as e:
                    logger.warning(f"Failed to delete GCS checkpoint step_{step}: {e}")
        except Exception as e:
            logger.warning(f"GCS cleanup failed: {e}")

    def _update_influence_weights(self):
        """Update dataset mixture weights using influence functions."""
        if self.mixed_dataset is None:
            return

        try:
            from data_handler.influence import InfluenceDistillation

            logger.info("Computing influence-based mixture weights...")

            # Compute new weights (simplified - actual implementation would be more complex)
            # This is a placeholder that shows the integration point
            # The real implementation would use InfluenceDistillation.compute_mixture_weights()

            logger.info("Influence weight update complete")

        except ImportError:
            logger.warning("cheapertraining not available, skipping influence update")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load training state from checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if hasattr(self.student, "module"):
            self.student.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.student.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])
        self.eval_losses = checkpoint.get("eval_losses", [])

        logger.info(f"Resumed from step {self.global_step}")
