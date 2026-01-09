"""Custom Lightning callbacks for WrinkleFree training.

Provides:
- GCSCheckpointCallback: Upload checkpoints to GCS
- ZClipCallback: Adaptive gradient clipping
- TokenCountCallback: Track tokens and stop at target
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


# =============================================================================
# GCS Exceptions - FAIL LOUDLY
# =============================================================================


class GCSAuthError(Exception):
    """Raised when GCS authentication is missing or invalid.

    This is raised at training startup if gcs.enabled=true but auth is not configured.
    """

    pass


class GCSUploadError(Exception):
    """Raised when GCS upload fails after retries.

    This crashes training - checkpoints are critical for fault tolerance.
    If GCS is optional for your use case, set gcs.enabled=false.
    """

    pass


def validate_gcs_auth(bucket: str) -> None:
    """Validate GCS authentication upfront.

    FAILS LOUDLY if auth is missing or bucket is inaccessible.
    Call this at training startup when gcs.enabled=true.

    Args:
        bucket: GCS bucket name to validate access to

    Raises:
        GCSAuthError: If gcloud not installed, not authenticated, or bucket inaccessible
    """
    # 1. Check gcloud is installed
    try:
        result = subprocess.run(
            ["gcloud", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise GCSAuthError(
                "gcloud CLI not working properly.\n"
                "Install: https://cloud.google.com/sdk/docs/install"
            )
    except FileNotFoundError:
        raise GCSAuthError(
            "gcloud CLI not found.\n"
            "Install: https://cloud.google.com/sdk/docs/install\n"
            "Or set gcs.enabled=false if GCS is not needed."
        )
    except subprocess.TimeoutExpired:
        raise GCSAuthError("gcloud --version timed out")

    # 2. Check gcloud is authenticated
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise GCSAuthError(
                "gcloud not authenticated.\n"
                f"Error: {result.stderr.strip()}\n"
                "Run: gcloud auth login\n"
                "Or for service accounts: gcloud auth activate-service-account --key-file=KEY.json"
            )
    except subprocess.TimeoutExpired:
        raise GCSAuthError("gcloud auth print-access-token timed out")

    # 3. Check bucket is accessible
    try:
        result = subprocess.run(
            ["gcloud", "storage", "ls", f"gs://{bucket}/"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "403" in stderr or "AccessDenied" in stderr:
                raise GCSAuthError(
                    f"Access denied to bucket '{bucket}'.\n"
                    f"Error: {stderr}\n"
                    "Check bucket permissions or use a different bucket."
                )
            elif "404" in stderr or "NotFound" in stderr:
                raise GCSAuthError(
                    f"Bucket '{bucket}' not found.\n"
                    f"Error: {stderr}\n"
                    "Create the bucket or check the bucket name."
                )
            else:
                raise GCSAuthError(
                    f"Cannot access bucket '{bucket}'.\n"
                    f"Error: {stderr}"
                )
    except subprocess.TimeoutExpired:
        raise GCSAuthError(f"gcloud storage ls gs://{bucket}/ timed out")

    logger.info(f"GCS auth validated: bucket '{bucket}' is accessible")


class MuonClipInitCallback(Callback):
    """Re-initialize MuonClip hooks after BatchSizeFinder completes.

    PROBLEM: MuonClip registers forward hooks on q_proj layers to capture attention
    inputs for QK-clipping. However, Lightning's BatchSizeFinder runs test training
    steps during on_fit_start(), cycling through model.eval() and model.train() calls.

    BUG IN MUON-CLIP (upstream): The HookRecorder.remove_hooks() method removes hook
    handles but doesn't reset the is_registered flag to False. This causes:
    1. model.eval() → remove_hooks() called → hooks removed, but is_registered=True
    2. model.train() → register_input_hook() called → exits early due to is_registered=True
    3. Result: hooks never re-registered, attn_inputs stays empty → KeyError in optimizer

    SOLUTION: This callback runs at on_train_start (after BatchSizeFinder completes)
    and forces hook re-registration by resetting is_registered=False first.

    Reference: https://github.com/GAD-cell/muon-clip
    """

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Re-initialize MuonClip hooks after BatchSizeFinder completes."""
        # Get the optimizer - may be wrapped by Lightning
        optimizer = trainer.optimizers[0]
        if hasattr(optimizer, "_optimizer"):
            # Lightning wraps optimizers
            optimizer = optimizer._optimizer

        # Check if this is a MuonClip optimizer with hook_recorder
        if not hasattr(optimizer, "hook_recorder"):
            return

        hook_recorder = optimizer.hook_recorder

        # Force re-registration by resetting the flag
        # This is needed because remove_hooks() doesn't reset is_registered (upstream bug)
        hook_recorder.is_registered = False

        # Re-register hooks on the model
        hook_recorder.register_input_hook(pl_module.model)

        # Log the fix
        num_hooks = len(hook_recorder.handles) if hasattr(hook_recorder, "handles") else 0
        logger.info(
            f"MuonClipInitCallback: Re-registered {num_hooks} hooks after BatchSizeFinder "
            f"(upstream bug workaround for is_registered flag)"
        )


class GCSCheckpointCallback(Callback):
    """Upload checkpoints to Google Cloud Storage.

    Uploads checkpoints to GCS after each save for fault tolerance.
    Long training runs MUST upload checkpoints - losing hours of GPU
    time to a crash is unacceptable.

    Also saves dlm_config.json alongside checkpoints for inference compatibility.
    The DLM config contains mask_token_id, mask_prob, etc. needed for DLM inference.

    Args:
        bucket: GCS bucket name (e.g., "wrinklefree-checkpoints")
        experiment_name: Experiment name for path organization
        stage: Training stage name (e.g., "lightning", "stage2")
        dlm_config: Optional DLM configuration dict with keys:
            - mask_token_id: Token ID used for masking (typically 0)
            - mask_prob: Masking probability during training
            - ignore_index: Label ignore index (typically -100)
            - training_method: E.g., "unified-dlm"
    """

    def __init__(
        self,
        bucket: str,
        experiment_name: str = "default",
        stage: str = "lightning",
        dlm_config: Optional[dict] = None,
    ):
        super().__init__()
        self.bucket = bucket
        self.experiment_name = experiment_name
        self.stage = stage
        self.dlm_config = dlm_config
        self._last_uploaded_path: Optional[str] = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check for new checkpoints after each batch and upload."""
        if not trainer.is_global_zero:
            return

        # Get latest checkpoint path
        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return

        ckpt_path = ckpt_callback.last_model_path
        if not ckpt_path or ckpt_path == self._last_uploaded_path:
            return

        # New checkpoint saved - upload it
        if Path(ckpt_path).exists():
            step = trainer.global_step
            ckpt_dir = Path(ckpt_path).parent

            # Save dlm_config.json alongside checkpoint
            if self.dlm_config:
                self._save_dlm_config(ckpt_dir)

            self._upload_to_gcs(Path(ckpt_path), f"step_{step:06d}")
            self._last_uploaded_path = ckpt_path

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Upload final checkpoint to GCS."""
        if not trainer.is_global_zero:
            return

        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is None:
            return

        ckpt_path = ckpt_callback.last_model_path
        if ckpt_path and Path(ckpt_path).exists():
            ckpt_dir = Path(ckpt_path).parent

            # Save dlm_config.json alongside final checkpoint
            if self.dlm_config:
                self._save_dlm_config(ckpt_dir)

            self._upload_to_gcs(Path(ckpt_path), "final")

    def _save_dlm_config(self, checkpoint_dir: Path) -> None:
        """Save dlm_config.json for inference compatibility.

        This file is needed by DLM inference servers to know the mask_token_id
        and other masking parameters used during training.
        """
        config_path = checkpoint_dir / "dlm_config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(self.dlm_config, f, indent=2)
            logger.info(f"Saved DLM config to {config_path}")
        except Exception as e:
            logger.warning(f"Failed to save dlm_config.json: {e}")

    def _upload_to_gcs(self, local_path: Path, checkpoint_type: str) -> None:
        """Upload checkpoint to GCS with retry.

        FAILS LOUDLY after 3 failed attempts - checkpoints are critical.

        Args:
            local_path: Local checkpoint path to upload
            checkpoint_type: Type label (e.g., "step_001000", "final")

        Raises:
            GCSUploadError: If upload fails after 3 retries
        """
        max_retries = 3
        retry_delay = 30  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                self._upload_to_gcs_once(local_path, checkpoint_type)
                return  # Success
            except GCSUploadError as e:
                if attempt == max_retries:
                    raise GCSUploadError(
                        f"GCS upload failed after {max_retries} attempts for {local_path}:\n"
                        f"{e}\n\n"
                        f"If GCS is optional, set gcs.enabled=false"
                    )
                logger.warning(
                    f"GCS upload attempt {attempt}/{max_retries} failed: {e}\n"
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)

    def _upload_to_gcs_once(self, local_path: Path, checkpoint_type: str) -> None:
        """Single upload attempt.

        Raises:
            GCSUploadError: On any failure (non-zero exit, timeout, etc.)
        """
        gcs_path = (
            f"gs://{self.bucket}/checkpoints/{self.experiment_name}/"
            f"{self.stage}_checkpoint/checkpoints/{checkpoint_type}/"
        )

        # Upload checkpoint file
        if local_path.is_dir():
            cmd = ["gcloud", "storage", "cp", "-r", str(local_path), gcs_path]
        else:
            cmd = ["gcloud", "storage", "cp", str(local_path), gcs_path]

        logger.info(f"Uploading checkpoint to {gcs_path}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            raise GCSUploadError(f"Upload timed out after 300s for {local_path}")

        if result.returncode != 0:
            raise GCSUploadError(f"gcloud storage cp failed:\n{result.stderr.strip()}")

        logger.info(f"Successfully uploaded checkpoint to {gcs_path}")

        # Also upload dlm_config.json if it exists alongside the checkpoint
        dlm_config_path = local_path.parent / "dlm_config.json"
        if dlm_config_path.exists():
            dlm_cmd = ["gcloud", "storage", "cp", str(dlm_config_path), gcs_path]
            logger.info(f"Uploading dlm_config.json to {gcs_path}")
            try:
                dlm_result = subprocess.run(dlm_cmd, capture_output=True, text=True, timeout=60)
                if dlm_result.returncode != 0:
                    # DLM config is less critical - warn but don't fail
                    logger.warning(f"Failed to upload dlm_config.json: {dlm_result.stderr}")
                else:
                    logger.info("Successfully uploaded dlm_config.json")
            except subprocess.TimeoutExpired:
                logger.warning("dlm_config.json upload timed out")


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
            from wf_data.training import ZClip

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


class DatasetRatioCallback(Callback):
    """Logs dataset mixture ratios to WandB.

    Always enabled. Logs both configured (initial) and observed (actual)
    dataset sampling ratios to track data distribution during training.

    Logged metrics:
    - data/configured_weight_{name}: Initial mixture weights (logged once at start)
    - data/observed_weight_{name}: Actual sampling ratios over training
    - data/total_samples: Cumulative samples processed from mixed dataset

    Args:
        log_interval: How often to log observed ratios (in steps)
    """

    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log initial configured weights at training start."""
        if trainer.datamodule is None:
            return

        if not hasattr(trainer.datamodule, "get_mixed_dataset"):
            return

        mixed = trainer.datamodule.get_mixed_dataset()
        if mixed is None:
            return

        if hasattr(mixed, "get_current_weights"):
            weights = mixed.get_current_weights()
            for name, weight in weights.items():
                pl_module.log(
                    f"data/configured_weight_{name}",
                    weight,
                    on_step=False,
                    on_epoch=False,
                    prog_bar=False,
                )
            logger.info(f"Dataset configured weights: {weights}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Log observed sampling ratios periodically."""
        step = trainer.global_step
        if step % self.log_interval != 0:
            return

        if trainer.datamodule is None:
            return

        if not hasattr(trainer.datamodule, "get_mixed_dataset"):
            return

        mixed = trainer.datamodule.get_mixed_dataset()
        if mixed is None:
            return

        if hasattr(mixed, "get_sampling_stats"):
            stats = mixed.get_sampling_stats()
            # Log observed weights (actual sampling ratios)
            for name, weight in stats.get("observed_weights", {}).items():
                pl_module.log(
                    f"data/observed_weight_{name}",
                    weight,
                    prog_bar=False,
                )
            # Log total samples
            pl_module.log(
                "data/total_samples",
                float(stats.get("total_samples", 0)),
                prog_bar=False,
            )


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

        from wf_data.training.qk_clip import apply_qk_clip

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
            from wf_arch.quantization import LambdaWarmup, set_global_lambda_warmup

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
        """Step lambda warmup only on optimizer steps (not every batch).

        CRITICAL: With gradient accumulation, on_train_batch_end fires for
        every micro-batch. We only want to step lambda on actual optimizer
        steps to prevent warmup from completing 16x too fast.
        """
        if self._lambda_warmup is not None:
            # Only step on optimizer steps (when gradient accumulation completes)
            # batch_idx+1 because batch_idx is 0-indexed
            accum_steps = trainer.accumulate_grad_batches
            if (batch_idx + 1) % accum_steps == 0:
                self._lambda_warmup.step()
                # Log lambda per optimizer step (not every micro-batch)
                pl_module.log(
                    "train/lambda",
                    self._lambda_warmup.lambda_val,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )


class RunManagerCallback(Callback):
    """GCS auto-resume with fingerprinting.

    Manages the lifecycle of training runs including:
    - Generating a fingerprint from config for run identification
    - Checking for existing completed runs (skip if already done)
    - Resuming from GCS checkpoint if available
    - Updating run status (RUNNING → COMPLETED/FAILED)

    This replaces the manual fingerprinting logic from the legacy train.py.

    Args:
        config: Full Hydra config (DictConfig or dict)
        skip_recovery: If True, skip GCS auto-resume check
        skip_completed: If True, exit early if run already completed

    Example:
        callbacks.append(RunManagerCallback(config=cfg))
    """

    def __init__(
        self,
        config,
        skip_recovery: bool = False,
        skip_completed: bool = True,
    ):
        super().__init__()
        self.config = config
        self.skip_recovery = skip_recovery
        self.skip_completed = skip_completed
        self._run_manager = None
        self._fingerprint = None
        self._should_skip = False
        self._resume_checkpoint_path = None
        self._resume_wandb_id = None

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        """Initialize fingerprinting and check for existing runs."""
        if stage != "fit":
            return

        rank = trainer.global_rank

        # Only rank 0 handles GCS operations
        if rank != 0:
            return

        try:
            from wf_train.utils import (
                AuditLogger,
                RunManager,
                RunStatus,
                generate_fingerprint,
            )
            from omegaconf import OmegaConf
        except ImportError:
            logger.warning(
                "RunManagerCallback: Required utilities not available, "
                "auto-resume disabled"
            )
            return

        # Generate fingerprint from config
        self._fingerprint, fp_metadata = generate_fingerprint(self.config)
        logger.info(f"Run fingerprint: {self._fingerprint[:16]}...")

        # Warn if git is dirty
        if fp_metadata.get("git_dirty", False):
            logger.warning(
                "Training with uncommitted git changes - results may not be reproducible"
            )

        # Get GCS config
        if isinstance(self.config, dict):
            gcs_config = self.config.get("gcs", {})
        else:
            gcs_config = OmegaConf.to_container(
                self.config.get("gcs", {}), resolve=True
            )

        # Skip if GCS not enabled or recovery disabled
        if not gcs_config.get("enabled", False) or self.skip_recovery:
            if self.skip_recovery:
                logger.info("Skip recovery mode: GCS auto-resume disabled")
            return

        # Initialize audit logger
        audit_logger = AuditLogger(enabled=True)

        try:
            # Create run manager
            output_dir = Path(self.config.get("output_dir", "/tmp/checkpoints"))
            self._run_manager = RunManager(
                fingerprint=self._fingerprint,
                gcs_bucket=gcs_config.get("bucket", "wrinklefree-checkpoints"),
                audit_logger=audit_logger,
                fingerprint_metadata=fp_metadata,
                gcs_prefix=gcs_config.get("experiment_prefix", "experiments"),
                local_cache_dir=output_dir / ".fingerprint_cache",
                rank=rank,
            )

            # Check for existing run
            should_resume, ckpt_path, wandb_id = self._run_manager.check_and_resume()

            # Handle already-completed runs
            if self._run_manager.is_completed():
                if self.skip_completed:
                    logger.info(
                        f"Run {self._fingerprint[:8]} already COMPLETED. "
                        "Set skip_completed=False to re-run."
                    )
                    self._should_skip = True
                    trainer.should_stop = True
                    return

            # Store resume info
            if should_resume and ckpt_path:
                self._resume_checkpoint_path = ckpt_path
                self._resume_wandb_id = wandb_id
                logger.info(f"Resuming from checkpoint: {ckpt_path}")
                if wandb_id:
                    logger.info(f"WandB run ID: {wandb_id}")

        except Exception as e:
            logger.error(f"RunManager initialization failed: {e}")
            # Don't fail training - just disable auto-resume
            self._run_manager = None

    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Update run status to RUNNING."""
        if self._should_skip:
            return

        if self._run_manager and trainer.is_global_zero:
            from wf_train.utils import RunStatus

            self._run_manager.update_status(RunStatus.RUNNING)

            # Update with WandB info if available
            try:
                import wandb

                if wandb.run and wandb.run.url:
                    self._run_manager.update_status(
                        RunStatus.RUNNING,
                        wandb_run_id=wandb.run.id,
                        wandb_url=wandb.run.url,
                    )
                    logger.info(f"Updated metadata with W&B URL: {wandb.run.url}")
            except Exception:
                pass

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Update run status to COMPLETED."""
        if self._should_skip:
            return

        if self._run_manager and trainer.is_global_zero:
            from wf_train.utils import RunStatus

            self._run_manager.update_status(
                RunStatus.COMPLETED,
                global_step=trainer.global_step,
            )
            logger.info(f"Run {self._fingerprint[:8]} marked as COMPLETED")

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        """Update run status to FAILED on exception."""
        if self._run_manager and trainer.is_global_zero:
            from wf_train.utils import RunStatus

            self._run_manager.update_status(
                RunStatus.FAILED,
                global_step=trainer.global_step,
                error_message=str(exception),
            )
            logger.info(f"Run {self._fingerprint[:8]} marked as FAILED")

    @property
    def fingerprint(self) -> str | None:
        """Get the run fingerprint."""
        return self._fingerprint

    @property
    def resume_checkpoint_path(self) -> Path | None:
        """Get the path to resume checkpoint if available."""
        return self._resume_checkpoint_path

    @property
    def resume_wandb_id(self) -> str | None:
        """Get the WandB run ID for resuming."""
        return self._resume_wandb_id

    @property
    def should_skip(self) -> bool:
        """Check if training should be skipped (already completed)."""
        return self._should_skip
