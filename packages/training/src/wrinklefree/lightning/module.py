"""WrinkleFree Lightning Module.

Wraps the model and ObjectiveManager in a LightningModule for clean training.
Reuses existing objective system without modification.
"""

import logging
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

from wrinklefree.objectives import ObjectiveManager

logger = logging.getLogger(__name__)


class WrinkleFreeLightningModule(pl.LightningModule):
    """Lightning module for WrinkleFree training.

    Integrates with the existing ObjectiveManager system for multi-objective
    training (DLM, LRC, distillation, etc.).

    Args:
        model: The model to train (BitNet or standard transformer)
        objective_manager: ObjectiveManager with configured objectives
        teacher_model: Optional teacher model for distillation objectives
        optimizer_cfg: Optimizer configuration dict
        scheduler_cfg: Scheduler configuration dict
        gradient_clipping: Max gradient norm (0 to disable)
    """

    def __init__(
        self,
        model: nn.Module,
        objective_manager: ObjectiveManager,
        teacher_model: Optional[nn.Module] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        scheduler_cfg: Optional[DictConfig] = None,
        gradient_clipping: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.objective_manager = objective_manager
        self.teacher_model = teacher_model
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.gradient_clipping = gradient_clipping

        # Freeze teacher if present
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # Track tokens for logging
        self.tokens_processed = 0

        # Save hyperparameters (excluding large objects)
        self.save_hyperparameters(ignore=["model", "objective_manager", "teacher_model"])

    def forward(self, **batch) -> dict[str, Any]:
        """Forward pass through model."""
        output_hidden_states = self.objective_manager.requires_hidden_states
        output_attentions = self.objective_manager.requires_attentions

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        return {
            "logits": outputs.logits,
            "hidden_states": getattr(outputs, "hidden_states", None),
            "attentions": getattr(outputs, "attentions", None),
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Single training step.

        Uses ObjectiveManager to compute all objective losses and combine them.
        """
        # Preprocess batch (DLM masking, etc.)
        batch = self.objective_manager.preprocess_batch(batch)

        # Forward pass through student model
        model_outputs = self.forward(**batch)

        # Forward pass through teacher if needed
        teacher_outputs = None
        if self.teacher_model is not None and self.objective_manager.requires_teacher:
            with torch.no_grad():
                teacher_out = self.teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=self.objective_manager.requires_hidden_states,
                    output_attentions=self.objective_manager.requires_attentions,
                    return_dict=True,
                )
                teacher_outputs = {
                    "logits": teacher_out.logits,
                    "hidden_states": getattr(teacher_out, "hidden_states", None),
                    "attentions": getattr(teacher_out, "attentions", None),
                }

        # Compute combined loss via ObjectiveManager
        manager_output = self.objective_manager(model_outputs, batch, teacher_outputs)

        # Log all metrics (reuse existing get_wandb_metrics!)
        metrics = self.objective_manager.get_wandb_metrics(manager_output, prefix="train")
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        # Log main metrics to progress bar
        self.log("loss", manager_output.loss, prog_bar=True, sync_dist=True)
        if manager_output.perplexity is not None:
            self.log("ppl", manager_output.perplexity, prog_bar=True, sync_dist=True)

        # Update token count
        batch_tokens = batch["input_ids"].numel()
        self.tokens_processed += batch_tokens
        self.log("train/tokens", self.tokens_processed, on_step=True, sync_dist=True)

        # Advance curriculum scheduler if present
        self.objective_manager.step_curriculum()

        return manager_output.loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step - same as training but without curriculum updates."""
        batch = self.objective_manager.preprocess_batch(batch)
        model_outputs = self.forward(**batch)

        teacher_outputs = None
        if self.teacher_model is not None and self.objective_manager.requires_teacher:
            with torch.no_grad():
                teacher_out = self.teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=self.objective_manager.requires_hidden_states,
                    return_dict=True,
                )
                teacher_outputs = {
                    "logits": teacher_out.logits,
                    "hidden_states": getattr(teacher_out, "hidden_states", None),
                }

        manager_output = self.objective_manager(model_outputs, batch, teacher_outputs)

        # Log validation metrics
        metrics = self.objective_manager.get_wandb_metrics(manager_output, prefix="val")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", manager_output.loss, prog_bar=True, sync_dist=True)

        return manager_output.loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler.

        Supports Muon (default) and AdamW optimizers.
        """
        optimizer_type = self.optimizer_cfg.get("type", "muonclip")
        learning_rate = self.optimizer_cfg.get("learning_rate", 1e-4)
        weight_decay = self.optimizer_cfg.get("weight_decay", 0.1)

        if optimizer_type.lower() == "muonclip":
            optimizer = self._create_muon_optimizer(learning_rate, weight_decay)
        elif optimizer_type.lower() == "adamw_8bit":
            optimizer = self._create_adamw_8bit_optimizer(learning_rate, weight_decay)
        else:
            optimizer = self._create_adamw_optimizer(learning_rate, weight_decay)

        # Create scheduler if configured
        scheduler_config = None
        if self.scheduler_cfg:
            scheduler = self._create_scheduler(optimizer)
            if scheduler is not None:
                scheduler_config = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }

        if scheduler_config:
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return optimizer

    def _create_muon_optimizer(self, learning_rate: float, weight_decay: float):
        """Create Muon optimizer (FSDP-compatible via muon_fsdp2)."""
        from muon_fsdp2 import Muon

        # Separate parameters: Muon for 2D weights, Adam for 1D (bias, norm, embeddings)
        muon_params = []
        adam_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and "embed" not in name.lower():
                muon_params.append(param)
            else:
                adam_params.append(param)

        lr_muon = self.optimizer_cfg.get("lr_muon", learning_rate)
        lr_adam = self.optimizer_cfg.get("lr_adam", learning_rate)

        optimizer = Muon([
            {"params": muon_params, "lr": lr_muon, "use_muon": True},
            {"params": adam_params, "lr": lr_adam, "use_muon": False},
        ])

        logger.info(
            f"Created Muon optimizer: {len(muon_params)} Muon params, "
            f"{len(adam_params)} Adam params, lr_muon={lr_muon:.2e}, lr_adam={lr_adam:.2e}"
        )
        return optimizer

    def _create_adamw_8bit_optimizer(self, learning_rate: float, weight_decay: float):
        """Create 8-bit AdamW optimizer via bitsandbytes."""
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        logger.info(f"Created AdamW 8-bit optimizer: lr={learning_rate:.2e}")
        return optimizer

    def _create_adamw_optimizer(self, learning_rate: float, weight_decay: float):
        """Create standard AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        logger.info(f"Created AdamW optimizer: lr={learning_rate:.2e}")
        return optimizer

    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        scheduler_type = self.scheduler_cfg.get("type", "cosine_warmup")
        warmup_steps = self.scheduler_cfg.get("warmup_steps", 1000)
        max_steps = self.scheduler_cfg.get("max_steps", 10000)
        min_lr_ratio = self.scheduler_cfg.get("min_lr_ratio", 0.1)

        if scheduler_type == "cosine_warmup":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=optimizer.defaults["lr"] * min_lr_ratio,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            logger.info(
                f"Created cosine warmup scheduler: warmup={warmup_steps}, "
                f"max_steps={max_steps}, min_lr_ratio={min_lr_ratio}"
            )
            return scheduler

        elif scheduler_type == "constant":
            return None

        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using constant LR")
            return None

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        """Custom gradient clipping - let Lightning handle it."""
        # Use Lightning's built-in clipping
        if self.gradient_clipping > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.gradient_clipping,
                gradient_clip_algorithm="norm",
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Add extra state to checkpoint."""
        checkpoint["tokens_processed"] = self.tokens_processed
        if self.objective_manager.curriculum is not None:
            checkpoint["curriculum_state"] = self.objective_manager.curriculum.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore extra state from checkpoint."""
        self.tokens_processed = checkpoint.get("tokens_processed", 0)
        if self.objective_manager.curriculum is not None and "curriculum_state" in checkpoint:
            self.objective_manager.curriculum.load_state_dict(checkpoint["curriculum_state"])
