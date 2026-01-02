"""Custom Lightning callbacks for WrinkleFree training.

Provides:
- GCSCheckpointCallback: Upload checkpoints to GCS
- ZClipCallback: Adaptive gradient clipping
- TokenCountCallback: Track tokens and stop at target
- InfluenceTrackerCallback: Influence-based dataset remixing
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class GCSCheckpointCallback(Callback):
    """Upload checkpoints to Google Cloud Storage.

    Uploads checkpoints to GCS after each save, using the same path structure
    as the existing training pipeline.

    Args:
        bucket: GCS bucket name (e.g., "wrinklefree-checkpoints")
        experiment_name: Experiment name for path organization
        stage: Training stage name (e.g., "lightning", "stage2")
        upload_final: Whether to upload final checkpoint
        upload_interval: Upload every N checkpoints (0 = only final)
    """

    def __init__(
        self,
        bucket: str,
        experiment_name: str = "default",
        stage: str = "lightning",
        upload_final: bool = True,
        upload_interval: int = 0,
    ):
        super().__init__()
        self.bucket = bucket
        self.experiment_name = experiment_name
        self.stage = stage
        self.upload_final = upload_final
        self.upload_interval = upload_interval
        self._checkpoint_count = 0

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Called when checkpoint is being saved."""
        self._checkpoint_count += 1

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Upload final checkpoint to GCS."""
        if not self.upload_final:
            return

        if trainer.is_global_zero:
            # Get the last checkpoint path from trainer
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if not ckpt_path:
                ckpt_path = trainer.checkpoint_callback.last_model_path

            if ckpt_path and Path(ckpt_path).exists():
                self._upload_to_gcs(Path(ckpt_path), "final")

    def _upload_to_gcs(self, local_path: Path, checkpoint_type: str) -> bool:
        """Upload checkpoint to GCS using gsutil."""
        gcs_path = (
            f"gs://{self.bucket}/checkpoints/{self.experiment_name}/"
            f"{self.stage}_checkpoint/checkpoints/{checkpoint_type}/"
        )

        try:
            # Upload directory or file
            if local_path.is_dir():
                cmd = ["gcloud", "storage", "cp", "-r", str(local_path), gcs_path]
            else:
                cmd = ["gcloud", "storage", "cp", str(local_path), gcs_path]

            logger.info(f"Uploading checkpoint to {gcs_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"GCS upload failed: {result.stderr}")
                return False

            logger.info(f"Successfully uploaded to {gcs_path}")
            return True

        except subprocess.TimeoutExpired:
            logger.error("GCS upload timed out")
            return False
        except Exception as e:
            logger.error(f"GCS upload error: {e}")
            return False


class ZClipCallback(Callback):
    """Adaptive gradient clipping using ZClip algorithm.

    Uses z-score based anomaly detection to clip only gradient spikes,
    not normal large gradients.

    Args:
        z_threshold: Z-score threshold for clipping (default: 3.0)
        ema_decay: EMA decay for gradient norm statistics (default: 0.99)
        enabled: Whether to enable ZClip (default: True)
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        ema_decay: float = 0.99,
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self.z_threshold = z_threshold
        self.ema_decay = ema_decay
        self._zclip = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Initialize ZClip."""
        if self.enabled:
            from data_handler.training import ZClip

            self._zclip = ZClip(
                z_threshold=self.z_threshold,
                ema_decay=self.ema_decay,
            )
            logger.info(
                f"ZClip enabled: z_threshold={self.z_threshold}, "
                f"ema_decay={self.ema_decay}"
            )

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Apply ZClip before optimizer step."""
        if not self.enabled or self._zclip is None:
            return

        # Apply ZClip
        stats = self._zclip.clip(pl_module.model)

        # Log stats
        if trainer.is_global_zero:
            pl_module.log("train/grad_norm_raw", stats.raw_norm, prog_bar=False)
            pl_module.log("train/grad_norm", stats.clipped_norm, prog_bar=False)
            pl_module.log("train/grad_clipped", float(stats.was_clipped), prog_bar=False)
            if stats.ema_mean is not None:
                pl_module.log("train/zclip_ema_mean", stats.ema_mean, prog_bar=False)
                pl_module.log("train/zclip_ema_std", stats.ema_std, prog_bar=False)


class TokenCountCallback(Callback):
    """Track tokens processed and optionally stop at target.

    Args:
        max_tokens: Maximum tokens to train on (0 = no limit)
        seq_length: Sequence length for token counting
        log_interval: Log token count every N steps
    """

    def __init__(
        self,
        max_tokens: int = 0,
        seq_length: int = 2048,
        log_interval: int = 100,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.seq_length = seq_length
        self.log_interval = log_interval
        self.tokens_processed = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Update token count after each batch."""
        # Count tokens in batch
        # Note: This is called per micro-batch, so we count all tokens.
        # With gradient accumulation, this is called N times per optimizer step.
        batch_size = batch["input_ids"].shape[0]
        tokens_in_batch = batch_size * self.seq_length * trainer.world_size

        self.tokens_processed += tokens_in_batch

        # Log periodically
        if trainer.global_step % self.log_interval == 0:
            pl_module.log("train/tokens_total", self.tokens_processed, prog_bar=False)

            if self.max_tokens > 0:
                progress = self.tokens_processed / self.max_tokens
                pl_module.log("train/token_progress", progress, prog_bar=True)

        # Check if we've hit the token limit
        if self.max_tokens > 0 and self.tokens_processed >= self.max_tokens:
            logger.info(
                f"Reached token limit: {self.tokens_processed:,} >= {self.max_tokens:,}"
            )
            trainer.should_stop = True

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Save token count to checkpoint."""
        checkpoint["tokens_processed"] = self.tokens_processed

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        """Restore token count from checkpoint."""
        self.tokens_processed = checkpoint.get("tokens_processed", 0)
        logger.info(f"Resumed with {self.tokens_processed:,} tokens processed")


class QKClipCallback(Callback):
    """QK clipping for attention stability.

    Clips the spectral norm of QK projections to prevent attention instability.

    Args:
        threshold: Spectral norm threshold (default: 1.0)
        alpha: EMA decay for threshold adaptation (default: 0.99)
        enabled: Whether to enable QK clipping (default: True)
    """

    def __init__(
        self,
        threshold: float = 1.0,
        alpha: float = 0.99,
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self.threshold = threshold
        self.alpha = alpha

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Apply QK clipping after backward pass."""
        if not self.enabled:
            return

        from data_handler.training.qk_clip import apply_qk_clip

        stats = apply_qk_clip(
            pl_module.model,
            threshold=self.threshold,
            alpha=self.alpha,
            enabled=True,
        )

        if trainer.is_global_zero and stats is not None:
            pl_module.log("qk_clip/max_spectral_norm", stats.max_score, prog_bar=False)
            pl_module.log("qk_clip/was_clipped", float(stats.was_clipped), prog_bar=False)
            pl_module.log("qk_clip/scale_factor", stats.scale_factor, prog_bar=False)


class LambdaWarmupCallback(Callback):
    """Gradual quantization warmup via lambda parameter.

    Ramps up the quantization strength over warmup steps.

    Args:
        warmup_steps: Number of steps to warm up over
        schedule: "linear" or "cosine"
        enabled: Whether to enable warmup (default: True)
    """

    def __init__(
        self,
        warmup_steps: int = 1000,
        schedule: str = "linear",
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self._lambda_warmup = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Initialize lambda warmup."""
        if self.enabled:
            from bitnet_arch.quantization import LambdaWarmup, set_global_lambda_warmup

            self._lambda_warmup = LambdaWarmup(
                warmup_steps=self.warmup_steps,
                schedule=self.schedule,
            )
            set_global_lambda_warmup(self._lambda_warmup)
            logger.info(
                f"Lambda warmup enabled: {self.warmup_steps} steps, schedule={self.schedule}"
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Step lambda warmup after each batch."""
        if self._lambda_warmup is not None:
            self._lambda_warmup.step()

            if trainer.is_global_zero:
                pl_module.log(
                    "train/lambda",
                    self._lambda_warmup.lambda_val,
                    prog_bar=False,
                )


class InfluenceTrackerCallback(Callback):
    """Lightning callback for influence-based dataset remixing.

    Wraps data_handler.influence.InfluenceTracker to integrate with Lightning.
    Dynamically adjusts dataset mixture weights during training based on
    influence scores computed via DataInf or InfluenceDistillation.

    Self-disables if:
    - influence.enabled=false in config
    - No mixed_dataset available (single-source data)
    - No probe_dataloaders available

    Args:
        config: Full training configuration (DictConfig or dict).
            Must contain 'influence' section with:
            - enabled: bool - Master switch
            - update_interval: int - Steps between weight updates
            - learning_rate: float - Weight update learning rate
            - warmup_steps: int - Steps before first update
            - method: str - "datainf" or "distillation"

    Example config:
        ```yaml
        influence:
          enabled: true
          update_interval: 1000
          learning_rate: 0.2
          warmup_steps: 500
          method: datainf
        ```

    Usage:
        callbacks.append(InfluenceTrackerCallback(config=cfg))
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._tracker = None
        self._enabled = False

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        """Create InfluenceTracker once we have access to model and datamodule."""
        if stage != "fit":
            return

        # Only import when needed (data_handler may not be installed)
        try:
            from data_handler.influence import InfluenceTracker
        except ImportError:
            logger.warning(
                "InfluenceTrackerCallback: data_handler.influence not available, "
                "influence tracking disabled"
            )
            return

        # Get datamodule from trainer
        datamodule = trainer.datamodule
        if datamodule is None:
            logger.warning(
                "InfluenceTrackerCallback: no datamodule available, "
                "influence tracking disabled"
            )
            return

        mixed_dataset = datamodule.get_mixed_dataset()
        probe_dataloaders = datamodule.get_probe_dataloaders()

        # Debug logging
        logger.info(f"InfluenceTrackerCallback: mixed_dataset={mixed_dataset is not None}")
        logger.info(f"InfluenceTrackerCallback: probe_dataloaders={probe_dataloaders is not None}")

        # InfluenceTracker expects config with 'influence' at root level,
        # but Hydra config has it under 'training.influence'.
        # Restructure config to match expected format.
        from omegaconf import OmegaConf

        if hasattr(self.config, "training"):
            # Full Hydra config - extract training section and move influence to root
            tracker_config = OmegaConf.to_container(self.config.training, resolve=True)
            # Move influence config to root level where InfluenceTracker expects it
            if "influence" in tracker_config:
                tracker_config = {"influence": tracker_config["influence"]}
            else:
                tracker_config = {"influence": {"enabled": False}}
        else:
            # Already a dict or simple config
            tracker_config = self.config

        logger.info(f"InfluenceTrackerCallback: tracker_config influence section={tracker_config.get('influence', {})}")

        # Create the tracker (it will self-disable if not configured)
        self._tracker = InfluenceTracker(
            config=tracker_config,
            model=pl_module.model,
            mixed_dataset=mixed_dataset,
            probe_dataloaders=probe_dataloaders,
        )

        self._enabled = self._tracker.is_enabled
        if self._enabled:
            logger.info("InfluenceTrackerCallback: initialized and enabled")
        else:
            logger.info(
                f"InfluenceTrackerCallback: disabled "
                f"(mixed_dataset={mixed_dataset is not None}, "
                f"influence.enabled={tracker_config.get('influence', {}).get('enabled', False)})"
            )

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Cache probe gradients at training start."""
        if self._tracker and self._enabled:
            self._tracker.on_train_begin()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Check for weight update after each batch."""
        if self._tracker and self._enabled:
            self._tracker.on_step_end(trainer.global_step)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Optional probe cache refresh at epoch end."""
        if self._tracker and self._enabled:
            self._tracker.on_epoch_end(trainer.current_epoch)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log final weight history."""
        if self._tracker and self._enabled:
            self._tracker.on_train_end()

    @property
    def is_enabled(self) -> bool:
        """Check if influence tracking is active."""
        return self._enabled

    def get_current_weights(self) -> dict[str, float]:
        """Get current dataset mixture weights."""
        if self._tracker:
            return self._tracker.get_current_weights()
        return {}

    def get_weight_history(self) -> list[dict]:
        """Get history of weight updates."""
        if self._tracker:
            return self._tracker.get_weight_history()
        return []
