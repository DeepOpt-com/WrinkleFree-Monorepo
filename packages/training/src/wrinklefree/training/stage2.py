"""Stage 2: Continue Pre-Training - Adapt model to 1.58-bit quantization.

This unified stage supports two modes:
1. Pure LM training (default): pre_stage_2.enabled=false
2. Distillation mode: pre_stage_2.enabled=true (merges old stage1_9 functionality)

When pre_stage_2 is enabled, the trainer:
- Loads a teacher model for hidden state extraction
- Computes layer-wise distillation loss
- Applies a distillation schedule that ramps down over training
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cheapertraining.training import PlateauEarlyStopping, ZClip
from wrinklefree.distillation import (
    ContinuePretrainLoss,
    HiddenStateTeacherWrapper,
    LayerwiseDistillationLoss,
    LayerwiseLossType,
)
from wrinklefree.quantization.lambda_warmup import (
    LambdaWarmup,
    set_global_lambda_warmup,
)
from wrinklefree.training.fsdp_wrapper import setup_distributed, wrap_model_fsdp
from wrinklefree.training.tensor_parallel import setup_2d_parallel
from wrinklefree.training.trainer import Trainer, create_optimizer, create_scheduler

logger = logging.getLogger(__name__)


class Stage2Trainer(Trainer):
    """
    Trainer for Stage 2 continue pre-training.

    Supports two modes:
    1. Pure LM training (default): Uses ContinuePretrainLoss for next-token prediction
    2. Pre-stage-2 distillation: Combines layer-wise distillation with LM loss

    Args:
        model: BitNet student model
        optimizer: Optimizer instance
        train_dataloader: Training data loader
        config: Training configuration (DictConfig)
        teacher: Optional HiddenStateTeacherWrapper for distillation mode
        pre_stage_2_config: Optional config for distillation settings
        **kwargs: Additional arguments for base Trainer
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        config: DictConfig,
        teacher: Optional[HiddenStateTeacherWrapper] = None,
        pre_stage_2_config: Optional[DictConfig] = None,
        next_phase_loader_future: Optional[Any] = None,  # concurrent.futures.Future
        switch_step: Optional[int] = None,
        **kwargs,
    ):
        # Determine mode: distillation vs pure LM
        self.pre_stage_2_enabled = teacher is not None and pre_stage_2_config is not None
        self.teacher = teacher

        # Setup loss function based on mode
        if self.pre_stage_2_enabled:
            # Distillation mode: use LayerwiseDistillationLoss
            layerwise_cfg = pre_stage_2_config.layerwise
            loss_type_str = layerwise_cfg.get("loss_type", "mse_normalized")
            loss_type = LayerwiseLossType(loss_type_str)
            layer_weights = layerwise_cfg.get("layer_weights", None)

            loss_fn = LayerwiseDistillationLoss(
                loss_type=loss_type,
                layer_weights=layer_weights,
                hidden_size=layerwise_cfg.get("hidden_size"),
                vocab_size=layerwise_cfg.get("vocab_size"),
                temperature=layerwise_cfg.get("temperature", 1.0),
                normalize=layerwise_cfg.get("normalize", True),
            )
            logger.info(f"Pre-stage-2 mode: loss_type={loss_type_str}, layer_weights={layer_weights}")
        else:
            # Pure LM mode: use ContinuePretrainLoss
            loss_fn = ContinuePretrainLoss()

        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            config=config,
            **kwargs,
        )

        # Curriculum learning: swap dataloader mid-training
        # The next phase dataloader is loaded in a background thread
        self.next_phase_loader_future = next_phase_loader_future
        self.next_phase_dataloader = None  # Will be set when future completes
        self.switch_step = switch_step
        if switch_step is not None:
            logger.info(f"Curriculum: Will switch to Phase 2 dataloader at step {switch_step}")

        # Get training config (support both full config and training-only config)
        self.training_cfg = getattr(config, "training", config)

        # Token counting
        self.total_tokens = getattr(self.training_cfg, "total_tokens", 10_000_000_000)
        self.tokens_processed = 0
        self.seq_length = getattr(self.training_cfg, "max_seq_length", 2048)

        # Lambda warmup for gradual quantization
        lambda_warmup_cfg = getattr(self.training_cfg, "lambda_warmup", None)
        if lambda_warmup_cfg is not None and getattr(lambda_warmup_cfg, "enabled", False):
            warmup_steps = getattr(lambda_warmup_cfg, "warmup_steps", 1000)
            schedule = getattr(lambda_warmup_cfg, "schedule", "linear")
            self.lambda_warmup = LambdaWarmup(
                warmup_steps=warmup_steps,
                schedule=schedule,
            )
            set_global_lambda_warmup(self.lambda_warmup)
            logger.info(
                f"Lambda warmup enabled: {warmup_steps} steps, schedule={schedule}"
            )
        else:
            self.lambda_warmup = None
            # Ensure full quantization when warmup not enabled
            set_global_lambda_warmup(None)

        # Pre-stage-2 distillation schedule settings
        if self.pre_stage_2_enabled:
            schedule_config = pre_stage_2_config.get("distill_schedule", None)
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
                # Use fixed LM loss weight from layerwise config
                self.lm_loss_weight = pre_stage_2_config.layerwise.get("lm_loss_weight", 0.5)
        else:
            self.distill_schedule_enabled = False

        # Track influence-aware optimizer
        self.has_influence = self._check_influence_optimizer()
        if self.has_influence:
            logger.info("Influence-aware optimizer detected - will log mixture weights")

        # Early stopping setup
        early_stop_cfg = getattr(self.training_cfg, "early_stopping", None)
        self.early_stopper = PlateauEarlyStopping(
            patience=getattr(early_stop_cfg, "patience", 5) if early_stop_cfg else 5,
            min_delta=getattr(early_stop_cfg, "min_delta", 0.01) if early_stop_cfg else 0.01,
            mode="min",
            min_evals=getattr(early_stop_cfg, "min_evals", 10) if early_stop_cfg else 10,
            enabled=getattr(early_stop_cfg, "enabled", False) if early_stop_cfg else False,
            rank=self.rank,
        )
        if self.early_stopper.enabled:
            logger.info(
                f"Early stopping enabled: patience={self.early_stopper.patience}, "
                f"min_delta={self.early_stopper.min_delta}"
            )

        # Loss EMA for early stopping (smoothed loss tracking)
        self.loss_ema: float = 0.0
        self.loss_ema_alpha: float = 0.99  # Smoothing factor

        # ZClip adaptive gradient clipping
        zclip_cfg = getattr(self.training_cfg, "zclip", None)
        if zclip_cfg is not None and getattr(zclip_cfg, "enabled", True):
            z_threshold = getattr(zclip_cfg, "z_threshold", 3.0)
            ema_decay = getattr(zclip_cfg, "ema_decay", 0.99)
            self.zclip = ZClip(z_threshold=z_threshold, ema_decay=ema_decay)
            logger.info(f"ZClip adaptive gradient clipping enabled: z_threshold={z_threshold}")
        else:
            self.zclip = None
            logger.info("ZClip disabled, using fixed gradient clipping")

    def _check_influence_optimizer(self) -> bool:
        """Check if optimizer is InfluenceAwareOptimizer."""
        optimizer_class_name = self.optimizer.__class__.__name__
        return optimizer_class_name == "InfluenceAwareOptimizer"

    def _get_mixture_weights(self) -> Optional[Dict[str, float]]:
        """Get current mixture weights if using influence-aware optimizer."""
        if not self.has_influence:
            return None

        try:
            # InfluenceAwareOptimizer has mixed_dataset attribute
            if hasattr(self.optimizer, "mixed_dataset"):
                return self.optimizer.mixed_dataset.get_current_weights()
        except Exception as e:
            logger.warning(f"Failed to get mixture weights: {e}")

        return None

    def _get_current_distill_weight(self) -> float:
        """
        Get current distillation weight based on schedule.

        Returns a value between 0.0 and 1.0 representing the distillation weight.
        This weight is applied as: loss = distill_weight * distill_loss + (1 - distill_weight) * lm_loss
        """
        if not self.pre_stage_2_enabled:
            return 0.0  # No distillation in pure LM mode

        if not self.distill_schedule_enabled:
            # Use fixed weight from lm_loss_weight config
            return 1.0 - getattr(self, "lm_loss_weight", 0.5)

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
        """Training loop with token counting.

        Note: No separate eval pass - all data is unseen in streaming pre-training.
        """
        import time
        from tqdm import tqdm

        self.model.train()

        # Calculate max_steps from token count, but respect explicit max_steps override
        # Priority: explicit max_steps > calculated from total_tokens
        tokens_per_step = (
            self.training_cfg.batch_size
            * self.seq_length
            * self.gradient_accumulation_steps
            * self.world_size
        )
        calculated_max_steps = self.total_tokens // tokens_per_step

        # Check if max_steps was explicitly set in config
        explicit_max_steps = getattr(self.training_cfg, "max_steps", None)
        if explicit_max_steps is not None:
            # Explicit max_steps takes priority
            self.max_steps = explicit_max_steps
            logger.info(f"Using explicit max_steps={explicit_max_steps} (calculated would be {calculated_max_steps})")
        else:
            # Fall back to calculated from total_tokens
            self.max_steps = calculated_max_steps

        logger.info(f"Stage 2: Training for {self.max_steps} steps ({self.total_tokens:,} tokens)")

        # Use tensor for loss accumulation to avoid GPU sync every micro-batch
        accumulated_loss = torch.tensor(0.0, device=self.device)
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
            # Curriculum: Check if background-loaded Phase 2 dataloader is ready
            if (
                self.next_phase_loader_future is not None
                and self.next_phase_dataloader is None
                and self.next_phase_loader_future.done()
            ):
                # Get the loaded dataloader from the future
                result = self.next_phase_loader_future.result()
                self.next_phase_dataloader = result[0]  # (dataloader, mixed_dataset, probe_loaders)
                if self.rank == 0:
                    logger.info(f"[Background] Phase 2 dataloader ready at step {self.global_step}")
                self.next_phase_loader_future = None  # Clear future

            # Curriculum: Switch dataloader at switch_step (if Phase 2 is ready)
            if (
                self.next_phase_dataloader is not None
                and self.switch_step is not None
                and self.global_step >= self.switch_step
            ):
                if self.rank == 0:
                    logger.info(f"Curriculum: Switching to Phase 2 dataloader at step {self.global_step}")
                self.train_dataloader = self.next_phase_dataloader
                data_iter = iter(self.train_dataloader)
                self.next_phase_dataloader = None  # Prevent re-switching
                import gc
                gc.collect()

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

            # Accumulate loss on GPU (avoid .item() sync every micro-batch)
            accumulated_loss += loss_dict["loss"].detach()
            num_accumulated += 1

            # Optimizer step
            grad_norm = None
            raw_grad_norm = None
            was_clipped = False
            if num_accumulated >= self.gradient_accumulation_steps:
                # Gradient clipping (FSDP-aware with ZClip support)
                if self.zclip is not None:
                    # Use ZClip adaptive clipping
                    if hasattr(self.model, "clip_grad_norm_"):
                        # For FSDP: compute stats then let FSDP do the actual clipping
                        stats = self.zclip.clip(self.model)
                        raw_grad_norm = stats.raw_norm
                        grad_norm = stats.clipped_norm
                        was_clipped = stats.was_clipped
                    else:
                        stats = self.zclip.clip(self.model)
                        raw_grad_norm = stats.raw_norm
                        grad_norm = stats.clipped_norm
                        was_clipped = stats.was_clipped
                elif self.gradient_clipping > 0:
                    # Fallback to fixed gradient clipping
                    if hasattr(self.model, "clip_grad_norm_"):
                        grad_norm = self.model.clip_grad_norm_(self.gradient_clipping)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clipping,
                        )
                    raw_grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                    was_clipped = raw_grad_norm > self.gradient_clipping

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                # Step lambda warmup (gradual quantization)
                if self.lambda_warmup is not None:
                    self.lambda_warmup.step()

                self.global_step += 1
                # Only sync GPU here (once per optimizer step, not every micro-batch)
                avg_loss = (accumulated_loss / num_accumulated).item()
                accumulated_loss.zero_()  # Reset for next accumulation
                self.train_losses.append(avg_loss)

                # Update loss EMA for early stopping
                if self.loss_ema == 0.0:
                    self.loss_ema = avg_loss
                else:
                    self.loss_ema = self.loss_ema_alpha * self.loss_ema + (1 - self.loss_ema_alpha) * avg_loss

                # Get lr for logging
                lr = self.optimizer.param_groups[0]["lr"]

                # Compute perplexity for display (always computed for pbar)
                ppl = loss_dict.get("perplexity", torch.tensor(0.0)).item()

                # Logging
                if self.global_step % self.log_interval == 0 and self.rank == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0

                    log_msg = (
                        f"Step {self.global_step}/{self.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"PPL: {ppl:.2f} | "
                        f"LR: {lr:.2e} | "
                        f"Steps/s: {steps_per_sec:.2f}"
                    )
                    logger.info(log_msg)

                    # WandB logging
                    if self.wandb_enabled:
                        wandb_log = {
                            "train/loss": avg_loss,
                            "train/perplexity": ppl,
                            "train/lr": lr,
                            "train/steps_per_sec": steps_per_sec,
                            "train/step": self.global_step,
                            "train/tokens_processed": self.tokens_processed,
                        }
                        # Log gradient norm (with ZClip support)
                        if grad_norm is not None:
                            gn = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                            wandb_log["train/grad_norm"] = gn
                            wandb_log["train/grad_clipped"] = 1.0 if was_clipped else 0.0
                            if raw_grad_norm is not None:
                                wandb_log["train/grad_norm_raw"] = raw_grad_norm
                            # Log ZClip stats if available
                            if self.zclip is not None and hasattr(self.zclip, 'ema_mean') and self.zclip.ema_mean is not None:
                                wandb_log["train/zclip_ema_mean"] = self.zclip.ema_mean
                                wandb_log["train/zclip_ema_std"] = (self.zclip.ema_var + 1e-8) ** 0.5
                        # Log lambda if warmup is active
                        if self.lambda_warmup is not None:
                            wandb_log["train/lambda"] = self.lambda_warmup.lambda_val

                        # Log distillation metrics if in pre_stage_2 mode
                        if self.pre_stage_2_enabled:
                            if "distill_loss" in last_loss_dict:
                                wandb_log["train/distill_loss"] = last_loss_dict["distill_loss"].item()
                            if "lm_loss" in last_loss_dict:
                                wandb_log["train/lm_loss"] = last_loss_dict["lm_loss"].item()
                            if "mean_layer_loss" in last_loss_dict:
                                wandb_log["train/mean_layer_loss"] = last_loss_dict["mean_layer_loss"].item()
                            # Log distillation schedule
                            current_distill_weight = self._get_current_distill_weight()
                            wandb_log["schedule/distill_weight"] = current_distill_weight
                            wandb_log["schedule/lm_weight"] = 1.0 - current_distill_weight

                        # Log mixture weights if using influence-aware optimizer
                        if self.has_influence:
                            mixture_weights = self._get_mixture_weights()
                            if mixture_weights:
                                for name, weight in mixture_weights.items():
                                    wandb_log[f"influence/weight_{name}"] = weight

                                # Also log total number of influence updates performed
                                if hasattr(self.optimizer, "_last_update_step"):
                                    num_updates = (self.global_step - self.optimizer._last_update_step) // self.optimizer.update_interval + 1
                                    if self.global_step >= self.optimizer.update_interval:
                                        wandb_log["influence/num_updates"] = max(0, num_updates)

                        self.wandb.log(wandb_log, step=self.global_step)

                # Early stopping check (on log interval, using smoothed loss)
                if self.early_stopper.check(self.loss_ema, self.global_step):
                    if self.rank == 0:
                        logger.warning("Stopping training early due to loss plateau.")
                        self.early_stopper.save_json(self.output_dir)
                    # Save checkpoint before exiting (all ranks must participate for FSDP)
                    self.save_checkpoint(f"early_stop_{self.global_step}")
                    break

                # Checkpointing (no eval)
                # NOTE: All ranks must call save_checkpoint for FSDP state dict gathering
                # (collective operation), but only rank 0 writes to disk
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                pbar.update(1)
                pbar.set_postfix({"loss": avg_loss, "ppl": ppl, "lr": lr})

                # Note: accumulated_loss already reset via .zero_() above
                num_accumulated = 0

        pbar.close()

        # Final metrics
        final_metrics = {
            "train_loss": self.train_losses[-1] if self.train_losses else 0.0,
            "perplexity": last_loss_dict.get("perplexity", torch.tensor(0.0)).item(),
            "tokens_processed": self.tokens_processed,
        }

        # Save final checkpoint (all ranks must participate for FSDP)
        self.save_checkpoint("final")
        if self.rank == 0:
            logger.info(f"Training complete! Final metrics: {final_metrics}")

        return final_metrics

    def _forward_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """Forward step with token tracking.

        Handles both modes:
        - Pure LM: uses ContinuePretrainLoss
        - Pre-stage-2: combines layer-wise distillation with LM loss
        """
        # Track tokens
        batch_tokens = batch["input_ids"].numel()
        self.tokens_processed += batch_tokens

        if self.pre_stage_2_enabled:
            return self._forward_step_distillation(batch)
        else:
            return self._forward_step_lm(batch)

    def _forward_step_lm(self, batch: dict) -> dict[str, torch.Tensor]:
        """Pure LM forward step (default mode)."""
        # Forward pass through model (with position_ids for sequence packing)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            position_ids=batch.get("position_ids"),
        )

        # Compute loss using ContinuePretrainLoss
        loss_dict = self.loss_fn(
            logits=outputs["logits"],
            labels=batch["labels"],
        )

        # Add Stage 2 specific metrics
        loss_dict["tokens_processed"] = torch.tensor(float(self.tokens_processed))

        # Compute perplexity (clamped to avoid overflow)
        ppl = torch.exp(torch.clamp(loss_dict["loss"].detach(), max=10.0))
        loss_dict["perplexity"] = ppl

        return loss_dict

    def _forward_step_distillation(self, batch: dict) -> dict[str, torch.Tensor]:
        """Pre-stage-2 forward step with hidden state distillation + LM loss."""
        # Get teacher hidden states and logits
        teacher_outputs = self.teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

        # Get student hidden states and logits
        student_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
        )

        # Extract hidden states (skip embedding layer at index 0)
        student_hidden = list(student_outputs["hidden_states"])[1:]
        teacher_hidden = list(teacher_outputs["hidden_states"])[1:]

        # Compute layerwise distillation loss
        loss_dict = self.loss_fn(
            student_hidden_states=student_hidden,
            teacher_hidden_states=teacher_hidden,
            attention_mask=batch.get("attention_mask"),
        )

        # Get scheduled distillation weight
        distill_weight = self._get_current_distill_weight()
        lm_weight = 1.0 - distill_weight

        # Combine with LM loss if using scheduled weights
        if lm_weight > 0 and "labels" in batch:
            # Get logits
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

            # Cross-entropy loss against ground truth
            ce_loss = F.cross_entropy(
                shift_student.view(-1, shift_student.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # KL divergence from teacher logits (soft distillation)
            temperature = 2.0
            kl_loss = F.kl_div(
                F.log_softmax(shift_student / temperature, dim=-1).view(-1, shift_student.size(-1)),
                F.softmax(shift_teacher.detach() / temperature, dim=-1).view(-1, shift_teacher.size(-1)),
                reduction="batchmean",
            ) * (temperature ** 2)

            # Combined LM loss: 0.5 CE + 0.5 KL
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

            # Compute perplexity from CE loss
            ppl = torch.exp(torch.clamp(ce_loss.detach(), max=10.0))
            loss_dict["perplexity"] = ppl
        else:
            # Pure distillation (no LM loss)
            loss_dict["perplexity"] = torch.tensor(0.0)

        # Add tracking metrics
        loss_dict["tokens_processed"] = torch.tensor(float(self.tokens_processed), device=batch["input_ids"].device)

        return loss_dict

    def save_checkpoint(self, name: str) -> None:
        """Save checkpoint with early stopper state."""
        # Call parent to save base checkpoint
        super().save_checkpoint(name)

        # Add early stopper state to checkpoint (rank 0 only)
        if self.rank == 0 and self.early_stopper.enabled:
            checkpoint_dir = self.output_dir / "checkpoints" / name
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            if checkpoint_path.exists():
                import torch
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                checkpoint["early_stopper"] = self.early_stopper.state_dict()
                checkpoint["loss_ema"] = self.loss_ema
                torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint with early stopper state."""
        # Call parent to load base checkpoint
        super().load_checkpoint(path)

        # Restore early stopper state
        if path.is_dir():
            checkpoint_path = path / "checkpoint.pt"
        else:
            checkpoint_path = path

        import torch
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "early_stopper" in checkpoint:
            self.early_stopper.load_state_dict(checkpoint["early_stopper"])
            logger.info(f"Restored early stopper state: best={self.early_stopper.best:.4f}, wait={self.early_stopper.wait}")
        if "loss_ema" in checkpoint:
            self.loss_ema = checkpoint["loss_ema"]


def run_stage2(
    model: nn.Module,
    train_dataloader: DataLoader,
    config: DictConfig,
    output_dir: Path,
    resume_from: Optional[Path] = None,
    probe_dataloader: Optional[DataLoader] = None,
    run_manager: Optional[Any] = None,
    experiment_name: Optional[str] = None,
    teacher_model_name: Optional[str] = None,
    next_phase_loader_future: Optional[Any] = None,  # concurrent.futures.Future
    switch_step: Optional[int] = None,
) -> nn.Module:
    """
    Run Stage 2: Continue pre-training to adapt weight distributions.

    Supports two modes:
    1. Pure LM training (default): pre_stage_2.enabled=false
    2. Distillation mode: pre_stage_2.enabled=true (merges old stage1_9 functionality)

    Args:
        model: BitNet model from Stage 1
        train_dataloader: Pre-training data loader
        config: Training configuration
        output_dir: Output directory for checkpoints
        probe_dataloader: Optional probe dataloader for influence-based data selection
        resume_from: Optional checkpoint to resume from
        run_manager: Optional RunManager for GCS checkpoint uploads
        experiment_name: Name for GCS checkpoint path (e.g., "bitdistill_smollm2_135m")
        teacher_model_name: Optional teacher model name for pre_stage_2 mode.
            If not provided, uses config.model.teacher.pretrained when pre_stage_2.enabled=true.

    Returns:
        Trained model
    """
    # Check if pre_stage_2 mode is enabled
    pre_stage_2_config = getattr(config.training, "pre_stage_2", None)
    pre_stage_2_enabled = pre_stage_2_config is not None and getattr(pre_stage_2_config, "enabled", False)

    if pre_stage_2_enabled:
        logger.info("=" * 60)
        logger.info("Stage 2: Starting with pre_stage_2 mode (distillation enabled)")
        logger.info("=" * 60)
    else:
        logger.info("Stage 2: Starting continue pre-training (pure LM mode)")

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load teacher model if pre_stage_2 is enabled
    teacher = None
    if pre_stage_2_enabled:
        # Get teacher model name from argument or config
        if teacher_model_name is None:
            # Try to get from model.teacher.pretrained config
            if hasattr(config, "model") and hasattr(config.model, "teacher"):
                teacher_model_name = getattr(config.model.teacher, "pretrained", None)
            # Fallback to model.pretrained_name (same as original model)
            if teacher_model_name is None and hasattr(config, "model"):
                teacher_model_name = getattr(config.model, "pretrained_name", None)

        if teacher_model_name is None:
            raise ValueError(
                "pre_stage_2.enabled=true but no teacher model specified. "
                "Set model.teacher.pretrained in config or pass teacher_model_name argument."
            )

        # Get teacher loading settings
        teacher_cfg = pre_stage_2_config.get("teacher", {})
        teacher = HiddenStateTeacherWrapper(
            model_name_or_path=teacher_model_name,
            device=device,
            load_in_fp16=teacher_cfg.get("fp16", True),
            offload_to_cpu=teacher_cfg.get("offload_to_cpu", False),
            load_in_4bit=teacher_cfg.get("load_in_4bit", False),
            use_flash_attention=teacher_cfg.get("use_flash_attention", False),
        )
        logger.info(f"Teacher model loaded: {teacher_model_name}")

    # Move model to device with uniform dtype (required for FSDP)
    # FSDP requires all tensors to have the same dtype before wrapping
    model = model.to(device=device, dtype=torch.bfloat16)

    # Apply FP8 GEMM acceleration if configured and supported (DeepSeek-V3 style)
    # This must happen BEFORE FSDP wrapping
    fp8_cfg = getattr(config, "fp8", None)
    if fp8_cfg is not None and getattr(fp8_cfg, "enabled", False):
        from wrinklefree.models.fp8_bitlinear import convert_bitlinear_to_fp8
        from wrinklefree.quantization.fp8_gemm import (
            FP8Capability,
            FP8Config,
            detect_fp8_capability,
            log_fp8_config,
        )

        fp8_config = FP8Config(
            enabled=True,
            recipe=getattr(fp8_cfg, "recipe", "rowwise"),
            accumulator_dtype=getattr(fp8_cfg, "accumulator_dtype", "float32"),
            min_gemm_size=getattr(fp8_cfg, "min_gemm_size", 512),
            exclude_patterns=tuple(getattr(fp8_cfg, "exclude_patterns", [])),
        )

        # Log FP8 configuration
        log_fp8_config(fp8_config)

        capability = detect_fp8_capability()
        if capability != FP8Capability.NONE:
            logger.info("Applying FP8 GEMM acceleration to BitLinear layers")
            model = convert_bitlinear_to_fp8(model, fp8_config)
        else:
            logger.warning("FP8 requested but hardware does not support it, using BF16")

    # Apply torch.compile for performance (30-50% speedup)
    # Only for single GPU - FSDP has its own compilation strategy
    torch_compile_cfg = getattr(config, "torch_compile", None)
    if torch_compile_cfg and getattr(torch_compile_cfg, "enabled", False) and world_size == 1:
        compile_mode = getattr(torch_compile_cfg, "mode", "max-autotune")
        fullgraph = getattr(torch_compile_cfg, "fullgraph", True)
        dynamic = getattr(torch_compile_cfg, "dynamic", False)

        logger.info(
            f"Applying torch.compile with mode={compile_mode}, "
            f"fullgraph={fullgraph}, dynamic={dynamic}"
        )

        # For max performance: max-autotune + fullgraph=True + static shapes
        # reduce-overhead uses CUDA Graphs for static models
        compile_options = {}
        if compile_mode == "reduce-overhead":
            # Enable CUDA Graphs for static models (20-30% additional speedup)
            compile_options["options"] = {"triton.cudagraphs": True}
            logger.info("CUDA Graphs enabled via reduce-overhead mode")

        model = torch.compile(
            model,
            mode=compile_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            **compile_options,
        )

    # Wrap with distributed training (FSDP or TP+FSDP2)
    if world_size > 1:
        from wrinklefree.models import BitNetDecoderLayer

        # Check if using tensor parallelism
        dist_strategy = getattr(config.distributed, "strategy", "fsdp")
        tp_config = getattr(config.distributed, "tensor_parallel", None)
        use_tp = (
            dist_strategy == "tp_fsdp"
            or (tp_config is not None and getattr(tp_config, "enabled", False))
        )

        if use_tp:
            # 2D parallelism: Tensor Parallel + FSDP2
            tp_size = getattr(tp_config, "tp_size", 0) if tp_config else 0
            logger.info(f"Using 2D parallelism: TP+FSDP2 (tp_size={tp_size})")
            model, device_mesh = setup_2d_parallel(
                model,
                tp_size=tp_size,
                mixed_precision=config.distributed.fsdp.mixed_precision.enabled,
                activation_checkpointing=config.distributed.fsdp.activation_checkpointing.enabled,
            )
        else:
            # Standard FSDP1
            model = wrap_model_fsdp(
                model,
                transformer_layer_cls=BitNetDecoderLayer,
                sharding_strategy=config.distributed.fsdp.sharding_strategy,
                mixed_precision=config.distributed.fsdp.mixed_precision.enabled,
                activation_checkpointing=config.distributed.fsdp.activation_checkpointing.enabled,
            )

    # Create optimizer and scheduler
    opt_cfg = config.training.optimizer
    # For MuonClip: get separate LRs, fallback to lr if not specified
    lr_muon = getattr(opt_cfg, "lr_muon", getattr(opt_cfg, "lr", 0.02))
    lr_adam = getattr(opt_cfg, "lr_adam", getattr(opt_cfg, "lr", 2e-4))
    optimizer = create_optimizer(
        model,
        learning_rate=lr_muon,  # Primary LR (Muon uses this)
        weight_decay=config.training.optimizer.weight_decay,
        optimizer_type=config.training.optimizer.type,
        model_config=model.config if hasattr(model, "config") else None,
        log_dir=None,  # Disable muon's buggy tensorboard writer (uses WandB instead)
        # MuonClip-specific params
        lr_muon=lr_muon,
        lr_adam=lr_adam,
        momentum=getattr(opt_cfg, "momentum", 0.95),
        enable_clipping=getattr(opt_cfg, "enable_clipping", True),
        clipping_threshold=getattr(opt_cfg, "clipping_threshold", 50.0),
        clipping_alpha=getattr(opt_cfg, "clipping_alpha", 0.5),
    )

    # Influence-based data selection (optional)
    # Check both config.influence and config.training.influence for compatibility
    influence_config = None
    if hasattr(config, "training") and hasattr(config.training, "influence"):
        influence_config = config.training.influence
    elif hasattr(config, "influence"):
        influence_config = config.influence

    influence_enabled = influence_config is not None and influence_config.get("enabled", False)

    if influence_enabled:
        try:
            # Lazy import - only when influence is enabled
            from cheapertraining import (
                MixtureWeightCalculator,
                InfluenceAwareOptimizer,
                InfluenceConfig,
            )
        except ImportError:
            logger.error(
                "CheaperTraining library not found. "
                "Install it with: pip install -e ../WrinkleFree-CheaperTraining"
            )
            raise
        
        logger.info("Setting up influence-based data selection")
        
        # Create influence config
        # InfluenceConfig uses lambda_reg (regularization), not lambda_val
        inf_config = InfluenceConfig(
            lambda_reg=influence_config.config.get("lambda_val", 1e-4),
        )

        # Create mixture calculator (it creates its own DataInfCalculator internally)
        # MixtureWeightCalculator expects (model, probe_dataloader, ...)
        if probe_dataloader is None:
            logger.warning("No probe_dataloader provided - influence updates will be skipped")
        else:
            mixture_calc = MixtureWeightCalculator(
                model=model,
                probe_dataloader=probe_dataloader,
                influence_config=inf_config,
            )
        
        # Get mixed dataset from dataloader if available
        # Check for required interface (get_current_weights, update_weights_from_influence)
        # instead of specific class to support both MixedDataset and PretrainDataset
        dataset = train_dataloader.dataset
        if hasattr(dataset, "dataset"):
            # Nested dataset (e.g., from sampler wrapper)
            dataset = dataset.dataset

        # Check if dataset has the required CheaperTraining interface
        has_influence_interface = (
            hasattr(dataset, "get_current_weights") and
            hasattr(dataset, "update_weights_from_influence")
        )

        if has_influence_interface:
            mixed_dataset = dataset
            logger.info(f"Found mixed dataset with influence interface: {type(dataset).__name__}")
        else:
            logger.warning(
                f"Dataset {type(dataset).__name__} doesn't have influence interface "
                "(get_current_weights, update_weights_from_influence). Influence updates disabled."
            )
            mixed_dataset = None

        # Only wrap optimizer if both probe_dataloader and mixed_dataset are available
        if probe_dataloader is not None and mixed_dataset is not None:
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
        else:
            if probe_dataloader is None:
                logger.warning("Skipping InfluenceAwareOptimizer: no probe_dataloader")
            if mixed_dataset is None:
                logger.warning("Skipping InfluenceAwareOptimizer: no mixed_dataset")

    # Estimate max steps for scheduler
    tokens_per_step = (
        config.training.batch_size
        * config.training.max_seq_length
        * config.training.gradient_accumulation_steps
        * world_size
    )
    max_steps = config.training.total_tokens // tokens_per_step

    # Calculate WSD decay steps if using WSD scheduler
    num_decay_steps = None
    decay_type = "linear"
    min_lr_ratio = getattr(config.training.scheduler, "min_lr_ratio", 0.0)
    if config.training.scheduler.type == "wsd":
        decay_ratio = getattr(config.training.scheduler, "decay_ratio", 0.2)
        num_decay_steps = int(max_steps * decay_ratio)
        decay_type = getattr(config.training.scheduler, "decay_type", "linear")

    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.training.scheduler.type,
        num_training_steps=max_steps,
        num_warmup_steps=config.training.scheduler.warmup_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=min_lr_ratio,
        decay_type=decay_type,
    )

    # Create trainer (pass full config for run naming and wandb logging)
    # Pass teacher and pre_stage_2_config if in distillation mode
    trainer = Stage2Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        config=config,
        teacher=teacher if pre_stage_2_enabled else None,
        pre_stage_2_config=pre_stage_2_config if pre_stage_2_enabled else None,
        device=device,
        rank=rank,
        world_size=world_size,
        run_manager=run_manager,
        experiment_name=experiment_name,
        stage="stage2",
        next_phase_loader_future=next_phase_loader_future,
        switch_step=switch_step,
    )
    trainer.output_dir = output_dir

    # Resume if specified
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)

        # Fast-forward lambda_warmup to match restored global_step
        if trainer.lambda_warmup is not None:
            for _ in range(trainer.global_step):
                trainer.lambda_warmup.step()
            logger.info(f"Lambda warmup fast-forwarded to step {trainer.global_step}, lambda={trainer.lambda_warmup.lambda_val:.4f}")

    # Train
    metrics = trainer.train()

    logger.info(f"Stage 2: Complete. Processed {trainer.tokens_processed:,} tokens")

    return model, trainer.train_losses
