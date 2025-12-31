"""Main training loop for BitNet models."""

# Patch muon_fsdp2 before importing it (fixes missing muon_update function)
import wrinklefree.training.muon_patch  # noqa: F401

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from wrinklefree.utils.run_manager import RunManager

logger = logging.getLogger(__name__)


def download_checkpoint_from_gcs(
    bucket_name: str,
    stage: str,
    local_dir: Path,
    prefix: str = "checkpoints",
) -> Optional[Path]:
    """
    Download a checkpoint from Google Cloud Storage.

    Checkpoints are stored at: gs://{bucket_name}/{prefix}/{stage}/

    Args:
        bucket_name: GCS bucket name (e.g., "wrinklefree-checkpoints")
        stage: Training stage folder name (e.g., "smoke-test")
        local_dir: Local directory to download to
        prefix: Prefix path in bucket (default: "checkpoints")

    Returns:
        Path to downloaded checkpoint directory, or None if not found
    """
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Build the GCS prefix path
        gcs_prefix = f"{prefix}/{stage}/"
        print(f"[DEBUG GCS] Looking for checkpoint at: gs://{bucket_name}/{gcs_prefix}")

        # List blobs with the prefix
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        print(f"[DEBUG GCS] Found {len(blobs)} blobs")
        if not blobs:
            print(f"[DEBUG GCS] No checkpoint found at gs://{bucket_name}/{gcs_prefix}")
            logger.info(f"No checkpoint found at gs://{bucket_name}/{gcs_prefix}")
            return None

        # Download all files
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG GCS] Downloading {len(blobs)} files to {local_dir}")
        logger.info(f"Downloading checkpoint from GCS: gs://{bucket_name}/{gcs_prefix}")

        for blob in blobs:
            # Get relative path within the checkpoint folder
            rel_path = blob.name[len(gcs_prefix):]
            if rel_path and not blob.name.endswith("/"):
                local_file = local_dir / rel_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(local_file))
                logger.debug(f"Downloaded: {blob.name} -> {local_file}")

        # Return the local checkpoint directory
        print(f"[DEBUG GCS] ✓ Downloaded checkpoint to {local_dir}")
        logger.info(f"Downloaded checkpoint to {local_dir}")
        return local_dir

    except ImportError:
        print("[DEBUG GCS] ✗ google-cloud-storage not installed")
        logger.warning("google-cloud-storage not installed, cannot download from GCS")
        return None
    except Exception as e:
        print(f"[DEBUG GCS] ✗ Failed to download checkpoint: {e}")
        logger.warning(f"Failed to download checkpoint from GCS: {e}")
        return None


class Trainer:
    """
    Main trainer class for BitNet models.

    Handles training loop, logging, checkpointing, and evaluation.

    Args:
        model: The model to train
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        config: Training configuration
        device: Device to train on
        rank: Process rank for distributed training
        world_size: Total number of processes
        run_manager: Optional RunManager for GCS checkpoint uploads
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[DictConfig] = None,
        device: torch.device = torch.device("cuda"),
        rank: int = 0,
        world_size: int = 1,
        run_manager: Optional["RunManager"] = None,
        experiment_name: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.run_manager = run_manager
        self.experiment_name = experiment_name
        self.stage = stage

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Get training-specific config (support both full config and training-only config)
        training_config = getattr(config, "training", config) if config else {}

        # Config values with defaults (use training_config for training-specific settings)
        self.max_steps = getattr(training_config, "max_steps", 10000)
        self.gradient_accumulation_steps = getattr(training_config, "gradient_accumulation_steps", 1)
        self.gradient_clipping = getattr(training_config, "gradient_clipping", 1.0)
        # log_interval can be at config.log_interval or config.logging.log_interval
        logging_config = getattr(training_config, "logging", {})
        self.log_interval = getattr(logging_config, "log_interval", getattr(training_config, "log_interval", 10))
        self.eval_interval = getattr(training_config, "eval_interval", 500)
        # save_interval can be at config.save_interval or config.checkpoint.save_interval
        checkpoint_config = getattr(training_config, "checkpoint", {})
        self.save_interval = getattr(checkpoint_config, "save_interval", getattr(training_config, "save_interval", 1000))
        self.output_dir = Path(getattr(config, "output_dir", "./outputs"))

        # Metrics tracking
        self.train_losses = []
        self.eval_losses = []
        
        # WandB setup
        self.wandb_enabled = False
        if self.rank == 0:
            wandb_config = getattr(training_config, "logging", {})
            if hasattr(wandb_config, "wandb"):
                wandb_config = wandb_config.wandb
            elif isinstance(wandb_config, dict):
                wandb_config = wandb_config.get("wandb", {})
            else:
                wandb_config = {}

            if wandb_config.get("enabled", False) if isinstance(wandb_config, dict) else getattr(wandb_config, "enabled", False):
                try:
                    import wandb
                except ImportError:
                    raise RuntimeError(
                        "WandB is enabled in config but wandb package is not installed. "
                        "Install it with: uv add wandb"
                    )

                # Validate API key is present
                import os
                api_key = os.environ.get("WANDB_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        "WandB is enabled but WANDB_API_KEY environment variable is not set. "
                        "Set it with: export WANDB_API_KEY=your_api_key"
                    )

                # Validate API key format (alphanumeric + underscores only)
                if not api_key.replace("_", "").isalnum():
                    raise RuntimeError(
                        f"WANDB_API_KEY has invalid format. "
                        f"API keys should only contain letters, numbers, and underscores. "
                        f"Got: {api_key[:10]}... (length: {len(api_key)})"
                    )

                self.wandb = wandb

                project = wandb_config.get("project", "wrinklefree") if isinstance(wandb_config, dict) else getattr(wandb_config, "project", "wrinklefree")
                entity = wandb_config.get("entity", None) if isinstance(wandb_config, dict) else getattr(wandb_config, "entity", None)
                tags = wandb_config.get("tags", []) if isinstance(wandb_config, dict) else getattr(wandb_config, "tags", [])

                # Get run name from config or generate one
                run_name = wandb_config.get("name", None) if isinstance(wandb_config, dict) else getattr(wandb_config, "name", None)
                if not run_name:
                    from wrinklefree.training.run_naming import generate_run_name
                    run_name = generate_run_name(config)

                # Initialize wandb - it handles async uploads internally
                wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    tags=list(tags) if tags else [],
                    config=dict(config) if config else {},
                )
                self.wandb_enabled = True
                logger.info(f"WandB initialized: project={project}, entity={entity}")

                # Print wandb URL prominently for easy access
                if wandb.run and wandb.run.url:
                    print("\n" + "=" * 60)
                    print(f"WANDB RUN: {wandb.run.url}")
                    print("=" * 60 + "\n")

    def train(self) -> dict[str, float]:
        """
        Main training loop.

        Returns:
            Dictionary of final metrics
        """
        self.model.train()
        accumulated_loss = 0.0
        num_accumulated = 0
        start_time = time.time()

        if self.rank == 0:
            logger.info(f"Starting training for {self.max_steps} steps")

        # Verify dataloader lengths match across ranks (FSDP debug helper)
        if self.world_size > 1 and self.train_dataloader is not None:
            import torch.distributed as dist
            local_len = torch.tensor(len(self.train_dataloader), device=self.device)
            gathered = [torch.zeros_like(local_len) for _ in range(self.world_size)]
            dist.all_gather(gathered, local_len)
            if self.rank == 0:
                logger.info(f"Train batches per rank: {[x.item() for x in gathered]}")
            if not all(x == gathered[0] for x in gathered):
                raise RuntimeError(
                    f"Dataloader batch count mismatch across ranks: {[x.item() for x in gathered]}. "
                    "This will cause FSDP collective operation hangs."
                )

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

                self.global_step += 1
                avg_loss = accumulated_loss / num_accumulated
                self.train_losses.append(avg_loss)
                
                # Get lr for logging
                lr = self.optimizer.param_groups[0]["lr"]

                # Logging
                if self.global_step % self.log_interval == 0 and self.rank == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0

                    log_msg = (
                        f"Step {self.global_step}/{self.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f}"
                    )

                    # Add component losses if available
                    for key, value in loss_dict.items():
                        if key != "loss":
                            if isinstance(value, torch.Tensor):
                                log_msg += f" | {key}: {value.item():.4f}"
                            elif isinstance(value, (int, float)):
                                log_msg += f" | {key}: {value:.4f}"
                            # Skip lists (like layer_losses) in console log

                    logger.info(log_msg)
                    
                    # WandB logging
                    if self.wandb_enabled:
                        progress_pct = (self.global_step / self.max_steps) * 100 if self.max_steps > 0 else 0
                        wandb_log = {
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/steps_per_sec": steps_per_sec,
                            "train/step": self.global_step,
                            "train/epoch": self.epoch,
                            "train/progress_pct": progress_pct,
                        }
                        # Add component metrics from loss_dict
                        for key, value in loss_dict.items():
                            if key != "loss":
                                if isinstance(value, torch.Tensor):
                                    wandb_log[f"train/{key}"] = value.item()
                                else:
                                    wandb_log[f"train/{key}"] = value

                        # Add GPU memory stats
                        if torch.cuda.is_available():
                            wandb_log["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                            wandb_log["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                            wandb_log["system/gpu_max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

                        self.wandb.log(wandb_log, step=self.global_step)

                # Evaluation
                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.eval_interval == 0
                ):
                    eval_loss = self.evaluate()
                    if self.rank == 0:
                        logger.info(f"Step {self.global_step} | Eval Loss: {eval_loss:.4f}")

                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        # All ranks must call save_checkpoint for FSDP (collective op)
                        self.save_checkpoint("best")

                # Checkpointing
                # NOTE: All ranks must call save_checkpoint for FSDP state dict gathering
                # (collective operation), but only rank 0 writes to disk
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                pbar.update(1)
                pbar.set_postfix({"loss": avg_loss, "lr": lr})

                accumulated_loss = 0.0
                num_accumulated = 0

        pbar.close()

        # Final evaluation
        final_metrics = {"train_loss": self.train_losses[-1] if self.train_losses else 0.0}
        if self.eval_dataloader is not None:
            final_metrics["eval_loss"] = self.evaluate()

        # Save final checkpoint (all ranks must participate for FSDP)
        self.save_checkpoint("final")
        if self.rank == 0:
            logger.info(f"Training complete! Final metrics: {final_metrics}")

        return final_metrics

    def _forward_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        Perform a single forward step.

        Args:
            batch: Input batch

        Returns:
            Dictionary containing loss values
        """
        # Use autocast for mixed precision training (bfloat16)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.loss_fn is not None:
                # Use custom loss function (for distillation)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    position_ids=batch.get("position_ids"),
                    output_attentions=True,
                )

                # Get teacher outputs if available
                teacher_outputs = batch.get("teacher_outputs")

                if teacher_outputs is not None:
                    loss_dict = self.loss_fn(
                        student_logits=outputs["logits"],
                        teacher_logits=teacher_outputs["logits"],
                        student_attentions=outputs.get("attentions"),
                        teacher_attentions=teacher_outputs.get("attentions"),
                        labels=batch["labels"],
                        attention_mask=batch.get("attention_mask"),
                    )
                else:
                    # Simple LM loss
                    loss_dict = {"loss": outputs["loss"]}
            else:
                # Use model's built-in loss
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    position_ids=batch.get("position_ids"),
                    labels=batch["labels"],
                )
                loss_dict = {"loss": outputs["loss"]}

        return loss_dict

    def _move_to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate the model.

        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=self.rank != 0,
        ):
            batch = self._move_to_device(batch)
            loss_dict = self._forward_step(batch)
            total_loss += loss_dict["loss"].item()
            num_batches += 1

        self.model.train()

        avg_loss = total_loss / max(num_batches, 1)

        # Sync loss across ranks for consistent save decisions (FSDP requirement)
        if self.world_size > 1:
            import torch.distributed as dist
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()

        self.eval_losses.append(avg_loss)
        return avg_loss

    def save_checkpoint(self, name: str) -> None:
        """
        Save a checkpoint.

        NOTE: All ranks must call this method for FSDP models because
        state_dict gathering is a collective operation.

        Args:
            name: Checkpoint name
        """
        import torch.distributed as dist

        checkpoint_dir = self.output_dir / "checkpoints" / name
        if self.rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get state dict (handles FSDP if needed)
        # This is a COLLECTIVE operation - all ranks must participate
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(self.model, FSDP):
            from wrinklefree.training.fsdp_wrapper import get_fsdp_state_dict
            model_state = get_fsdp_state_dict(self.model, self.rank)
        else:
            model_state = self.model.state_dict()

        # Only rank 0 writes to disk
        if self.rank == 0:
            checkpoint = {
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_eval_loss": self.best_eval_loss,
                "train_losses": self.train_losses,
                "eval_losses": self.eval_losses,
            }

            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

            torch.save(checkpoint, checkpoint_dir / "checkpoint.pt")
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

            # Upload to GCS
            # Use synchronous upload for "final" checkpoint to ensure it completes
            # before process exits (daemon threads get killed on exit)
            if self.run_manager is not None:
                is_final = name == "final"
                self.run_manager.upload_checkpoint(
                    local_path=checkpoint_dir / "checkpoint.pt",
                    checkpoint_type=name,
                    experiment_name=self.experiment_name,
                    stage=self.stage,
                    background=not is_final,  # Sync for final, async for others
                )

        # Barrier to ensure checkpoint is written before any rank proceeds
        if dist.is_initialized() and self.world_size > 1:
            dist.barrier()

    def load_checkpoint(self, path: Path) -> None:
        """
        Load a checkpoint.

        Args:
            path: Path to checkpoint directory or file
        """
        if path.is_dir():
            checkpoint_path = path / "checkpoint.pt"
        else:
            checkpoint_path = path

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(self.model, FSDP):
            from wrinklefree.training.fsdp_wrapper import load_fsdp_state_dict
            load_fsdp_state_dict(self.model, checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state (skip if RESUME_OPTIMIZER=false or optimizer type changed)
        skip_optimizer = os.environ.get("RESUME_OPTIMIZER", "true").lower() == "false"
        if skip_optimizer:
            if self.rank == 0:
                logger.info("Skipping optimizer state load (RESUME_OPTIMIZER=false)")
        elif "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as e:
                if "doesn't match the size" in str(e):
                    if self.rank == 0:
                        logger.warning(f"Optimizer state mismatch (likely different optimizer type), skipping: {e}")
                else:
                    raise

        # Reset learning rate to config value (checkpoint may have different LR)
        config_lr = None
        if self.config:
            # Try multiple access patterns for OmegaConf/dict compatibility
            try:
                if hasattr(self.config, "training") and hasattr(self.config.training, "optimizer"):
                    config_lr = self.config.training.optimizer.lr
                elif hasattr(self.config, "optimizer"):
                    config_lr = self.config.optimizer.lr
            except (AttributeError, KeyError):
                pass

        if config_lr is not None:
            old_lr = self.optimizer.param_groups[0]["lr"]
            if old_lr != config_lr:
                # Update optimizer LRs (including initial_lr for scheduler compatibility)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = config_lr
                    if "initial_lr" in param_group:
                        param_group["initial_lr"] = config_lr
                if self.rank == 0:
                    logger.info(f"Reset LR from checkpoint value {old_lr} to config value {config_lr}")

        # Load scheduler state (skip if switching optimizers)
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint and not skip_optimizer:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                if self.rank == 0:
                    logger.warning(f"Failed to load scheduler state, starting fresh: {e}")
            # Also reset scheduler's base_lrs if we changed the LR
            # Must handle SequentialLR by updating sub-schedulers
            if config_lr is not None:
                def _update_sched_lr(sched):
                    if hasattr(sched, "base_lrs"):
                        sched.base_lrs = [config_lr] * len(sched.base_lrs)

                _update_sched_lr(self.scheduler)

                # Recursively update sub-schedulers (for SequentialLR/ChainedScheduler)
                if hasattr(self.scheduler, "_schedulers"):
                    for sub_sched in self.scheduler._schedulers:
                        _update_sched_lr(sub_sched)

        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        self.train_losses = checkpoint.get("train_losses", [])
        self.eval_losses = checkpoint.get("eval_losses", [])

        if self.rank == 0:
            logger.info(f"Loaded checkpoint from {checkpoint_path} at step {self.global_step}")


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    optimizer_type: str = "muonclip",
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer for the model.

    Args:
        model: Model to optimize
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        optimizer_type: Type of optimizer:
            - "muonclip" (default): Muon + QK-clipping for training stability
            - "adamw_8bit": 8-bit AdamW via bitsandbytes (memory efficient)
            - "adamw": Standard AdamW

    Returns:
        Optimizer instance
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "muonclip":
        try:
            # Use muon_fsdp2 - FSDP-compatible Muon optimizer
            # This uses gather-scatter instead of broadcast for sharded parameters
            from muon_fsdp2 import Muon

            # Separate parameters: Muon for 2D weights, Adam for 1D (bias, norm)
            muon_params = []
            adam_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # Muon works on 2D+ matrices, use Adam for embeddings and 1D params
                if param.ndim >= 2 and "embed" not in name.lower():
                    muon_params.append(param)
                else:
                    adam_params.append(param)

            lr_muon = kwargs.get("lr_muon", learning_rate)
            lr_adam = kwargs.get("lr_adam", learning_rate)

            # muon_fsdp2 API: only param_groups, parameters set per-group
            optimizer = Muon([
                {"params": muon_params, "lr": lr_muon, "use_muon": True},
                {"params": adam_params, "lr": lr_adam, "use_muon": False}
            ])
            logger.info(
                f"Using Muon (FSDP2) optimizer: {len(muon_params)} Muon params (lr={lr_muon}), "
                f"{len(adam_params)} Adam params (lr={lr_adam})"
            )
            return optimizer
        except ImportError:
            logger.warning("muon_fsdp2 not available, falling back to AdamW 8-bit")
            optimizer_type = "adamw_8bit"

    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Don't apply weight decay to bias and norm parameters
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=learning_rate,
                betas=betas,
            )
            logger.info("Using 8-bit AdamW optimizer")
            return optimizer
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to standard AdamW")

    return torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=betas,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "wsd",
    num_training_steps: int = 10000,
    num_warmup_steps: int = 100,
    num_stable_steps: int | None = None,
    num_decay_steps: int | None = None,
    min_lr_ratio: float = 0.0,
    decay_type: str = "linear",
) -> Any:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("wsd", "cosine", "linear", "constant")
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        num_stable_steps: Steps at peak LR (WSD only). If None, calculated from decay_steps.
        num_decay_steps: Steps to decay to min_lr (WSD only). If None, uses 20% of total.
        min_lr_ratio: Minimum LR as ratio of peak LR (default 0.0 for WSD)
        decay_type: Decay schedule type ("linear" or "cosine", WSD only)

    Returns:
        Scheduler instance
    """
    if scheduler_type == "wsd":
        # Warmup-Stable-Decay (WSD) scheduler
        # Used by DeepSeek-V3, maintains high LR during stable phase
        # Reference: https://arxiv.org/abs/2410.05192
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, LambdaLR, CosineAnnealingLR
        import logging

        # Calculate decay/stable steps if not provided
        if num_decay_steps is None:
            num_decay_steps = int(num_training_steps * 0.2)  # 20% for decay
        if num_stable_steps is None:
            num_stable_steps = max(1, num_training_steps - num_warmup_steps - num_decay_steps)

        # Validate
        total_phases = num_warmup_steps + num_stable_steps + num_decay_steps
        if total_phases != num_training_steps:
            logging.getLogger(__name__).info(
                f"WSD schedule: {num_warmup_steps} warmup + {num_stable_steps} stable + "
                f"{num_decay_steps} decay = {total_phases} steps (total: {num_training_steps})"
            )

        # Phase 1: Warmup (linear ramp to peak LR)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(1, num_warmup_steps),
        )

        # Phase 2: Stable (constant peak LR)
        stable_scheduler = LambdaLR(optimizer, lambda _: 1.0)

        # Phase 3: Decay (linear or cosine decay to min_lr)
        peak_lr = optimizer.param_groups[0]["lr"]
        if decay_type == "cosine":
            decay_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_decay_steps),
                eta_min=peak_lr * min_lr_ratio,
            )
        else:  # linear
            decay_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=max(min_lr_ratio, 1e-8),  # Avoid exactly 0
                total_iters=max(1, num_decay_steps),
            )

        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, stable_scheduler, decay_scheduler],
            milestones=[num_warmup_steps, num_warmup_steps + num_stable_steps],
        )

    elif scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

        # Warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )

        # Cosine decay - ensure T_max >= 1 to avoid division by zero
        cosine_t_max = max(1, num_training_steps - num_warmup_steps)
        if num_warmup_steps >= num_training_steps:
            import logging
            logging.getLogger(__name__).warning(
                f"warmup_steps ({num_warmup_steps}) >= num_training_steps ({num_training_steps}), "
                f"cosine decay will only run for 1 step"
            )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
        )

        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )

    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR

        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=num_training_steps,
        )

    elif scheduler_type == "constant":
        from torch.optim.lr_scheduler import LambdaLR

        return LambdaLR(optimizer, lambda _: 1.0)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
