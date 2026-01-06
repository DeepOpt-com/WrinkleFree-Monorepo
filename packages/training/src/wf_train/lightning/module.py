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

from wf_train.objectives import ObjectiveManager

logger = logging.getLogger(__name__)


class CombinedMuonAdamWOptimizer(torch.optim.Optimizer):
    """Combined optimizer wrapper for Muon + AdamW.

    Presents a single optimizer interface to Lightning while internally
    managing both Muon (for 2D weights) and AdamW (for embeddings/biases).

    This allows using automatic optimization with Lightning while getting
    the benefits of Muon's Newton-Schulz orthogonalization for matrix weights.
    """

    def __init__(self, muon_opt, adam_opt, lr: float):
        # Initialize with empty param_groups - we manage them via sub-optimizers
        # Use a dummy parameter to satisfy Optimizer requirements
        self.muon_opt = muon_opt
        self.adam_opt = adam_opt
        # Store defaults for scheduler compatibility
        self.defaults = {"lr": lr}
        # Combine param_groups for state_dict compatibility
        self.param_groups = muon_opt.param_groups + adam_opt.param_groups
        # State dict is managed by sub-optimizers
        self.state = {}

    def step(self, closure=None):
        """Step both optimizers."""
        loss = None
        if closure is not None:
            loss = closure()
        self.muon_opt.step()
        self.adam_opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for both optimizers."""
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adam_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return combined state dict."""
        return {
            "muon": self.muon_opt.state_dict(),
            "adam": self.adam_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load combined state dict."""
        if "muon" in state_dict and "adam" in state_dict:
            self.muon_opt.load_state_dict(state_dict["muon"])
            self.adam_opt.load_state_dict(state_dict["adam"])
        else:
            # Fallback for legacy state dicts
            logger.warning("Loading legacy optimizer state dict format")

    def add_param_group(self, param_group):
        """Not supported for combined optimizer."""
        raise NotImplementedError("Cannot add param groups to combined optimizer")


class WrinkleFreeLightningModule(pl.LightningModule):
    """Lightning module for WrinkleFree training.

    Integrates with the existing ObjectiveManager system for multi-objective
    training (DLM, LRC, distillation, etc.).

    Args:
        model: The model to train (BitNet or standard transformer)
        objective_manager: ObjectiveManager with configured objectives
        teacher_model: Optional teacher model for distillation objectives
        teacher_cfg: Optional config for lazy teacher loading (if teacher_model is None)
        optimizer_cfg: Optimizer configuration dict
        scheduler_cfg: Scheduler configuration dict
        gradient_clipping: Max gradient norm (0 to disable)
        resume_cfg: Resume configuration dict (controls what state to load)
    """

    def __init__(
        self,
        model: nn.Module,
        objective_manager: ObjectiveManager,
        teacher_model: Optional[nn.Module] = None,
        teacher_cfg: Optional[dict] = None,
        optimizer_cfg: Optional[DictConfig] = None,
        scheduler_cfg: Optional[DictConfig] = None,
        gradient_clipping: float = 1.0,
        resume_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.model = model
        self.objective_manager = objective_manager
        self.teacher_model = teacher_model
        self.teacher_cfg = teacher_cfg  # For lazy loading
        self._teacher_load_attempted = False  # Track if we've tried to load
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}
        self.gradient_clipping = gradient_clipping
        self.resume_cfg = resume_cfg or {}

        # Freeze teacher if present
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # Track tokens for logging
        self.tokens_processed = 0

        # Note: We skip save_hyperparameters() to avoid omegaconf types in
        # checkpoints, which cause PyTorch 2.6+ weights_only=True loading issues.
        # Config is already managed by Hydra.

    def forward(self, **batch) -> dict[str, Any]:
        """Forward pass through model."""
        output_hidden_states = self.objective_manager.requires_hidden_states
        output_attentions = self.objective_manager.requires_attentions

        # Debug: log if hidden states requested
        if output_hidden_states and not hasattr(self, "_logged_hidden_states"):
            import logging
            logging.getLogger(__name__).info(
                f"Forward: output_hidden_states={output_hidden_states}"
            )
            self._logged_hidden_states = True

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Extract hidden_states - handle both CausalLMOutputWithPast and dict
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and hasattr(outputs, "get"):
            hidden_states = outputs.get("hidden_states")

        return {
            "logits": outputs.logits,
            "hidden_states": hidden_states,
            "attentions": getattr(outputs, "attentions", None),
        }

    def _lazy_load_teacher(self) -> bool:
        """Lazily load teacher model when first needed.

        Returns True if teacher was loaded successfully, False otherwise.
        Only attempts to load once - subsequent calls return cached result.

        When teacher is loaded, batch size is automatically reduced by the
        configured batch_size_factor (default 0.5) to prevent OOM errors.
        """
        if self._teacher_load_attempted:
            return self.teacher_model is not None

        self._teacher_load_attempted = True

        if self.teacher_cfg is None:
            logger.warning("Teacher needed but no teacher_cfg provided for lazy loading")
            return False

        logger.info(
            f"Lazy loading teacher model at step {self.global_step} "
            f"(distill weight became non-zero)"
        )

        # Reduce batch size BEFORE loading teacher to prevent OOM
        batch_size_factor = self.teacher_cfg.get("batch_size_factor", 0.5)
        self._reduce_batch_size(batch_size_factor)

        try:
            from wf_train.teachers import HiddenStateTeacher

            teacher_model_name = self.teacher_cfg.get("model_name")
            if not teacher_model_name:
                logger.warning("No teacher model name in teacher_cfg")
                return False

            # Get distill config for attention settings
            distill_cfg = self.teacher_cfg.get("distill_cfg", {})
            attention_enabled = distill_cfg.get("attention", {}).get("enabled", False)
            use_eager_attention = self.teacher_cfg.get("use_eager_attention", attention_enabled)

            self.teacher_model = HiddenStateTeacher(
                model_name_or_path=teacher_model_name,
                device=self.device,
                load_in_fp16=self.teacher_cfg.get("fp16", True),
                offload_to_cpu=self.teacher_cfg.get("offload_to_cpu", False),
                load_in_4bit=self.teacher_cfg.get("load_in_4bit", False),
                use_flash_attention=self.teacher_cfg.get("use_flash_attention", not attention_enabled),
                use_eager_attention=use_eager_attention,
            )

            # Freeze teacher
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

            logger.info(f"Teacher model loaded successfully: {teacher_model_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to lazy load teacher model: {e}")
            return False

    def _reduce_batch_size(self, factor: float) -> None:
        """Reduce batch size by the given factor when teacher is loaded.

        This prevents OOM errors when the teacher model increases memory usage.
        The dataloader is recreated with the new batch size.

        Args:
            factor: Multiply current batch size by this factor (e.g., 0.5 = halve)
        """
        datamodule = getattr(self, "_datamodule", None)
        if datamodule is None:
            logger.warning(
                "Cannot reduce batch size: no datamodule reference. "
                "If you encounter OOM, restart with smaller batch_size."
            )
            return

        old_batch_size = datamodule.batch_size
        new_batch_size = max(1, int(old_batch_size * factor))

        if new_batch_size == old_batch_size:
            return

        logger.info(
            f"Reducing batch size for teacher loading: {old_batch_size} -> {new_batch_size} "
            f"(factor={factor})"
        )

        # Update datamodule batch size
        datamodule.batch_size = new_batch_size

        # Tell trainer to reload dataloaders with new batch size
        if self.trainer is not None:
            try:
                self.trainer.reset_train_dataloader()
                logger.info(f"Dataloader recreated with batch_size={new_batch_size}")
            except Exception as e:
                logger.warning(f"Could not reset dataloader: {e}. New batch size will apply on next epoch.")

    def _get_teacher_outputs(self, batch: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Get teacher model outputs if teacher is configured and needed.

        Extracts logits, hidden_states, and attentions from teacher model.
        Returns None if no teacher or objectives don't require it.

        Supports lazy loading: if teacher_model is None but teacher_cfg is provided
        and distill weight is non-zero, will load the teacher on-demand.

        IMPORTANT: Uses _original_input_ids (unmasked) for teacher forward when DLM
        preprocessing has been applied. AR teachers were never trained on masked
        inputs and would produce garbage predictions on [MASK] tokens.
        """
        if not self.objective_manager.requires_teacher:
            return None

        # Check if distill weight is currently non-zero
        current_weights = self.objective_manager.get_current_weights()
        distill_weight = current_weights.get("distill", 0.0)
        if distill_weight <= 0:
            # Distill not active in current curriculum phase
            return None

        # Lazy load teacher if needed
        if self.teacher_model is None:
            if self.teacher_cfg is not None and not self._teacher_load_attempted:
                if self._lazy_load_teacher():
                    # CRITICAL: Skip distillation for THIS step. The current batch
                    # was loaded with the old (larger) batch size. Running teacher
                    # forward on it would likely OOM. Distillation resumes next step
                    # with the reduced batch size.
                    logger.info(
                        "Skipping distillation for current step to apply batch size reduction. "
                        "Distillation will resume on next step."
                    )
                    return None
                else:
                    return None
            else:
                return None

        # Use original unmasked input_ids for AR teacher (not masked input from DLM)
        # Falls back to batch["input_ids"] if DLM preprocessing wasn't applied
        teacher_input_ids = batch.get("_original_input_ids", batch["input_ids"])

        with torch.no_grad():
            # HiddenStateTeacher.forward() returns dict with logits, hidden_states, attentions
            teacher_out = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=batch.get("attention_mask"),
                output_attentions=self.objective_manager.requires_attentions,
            )
            # teacher_out is already a dict from HiddenStateTeacher
            return teacher_out

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Single training step.

        Uses ObjectiveManager to compute all objective losses and combine them.
        If LDC-MTL meta-optimization is enabled, uses learned weights instead of
        curriculum weights.
        """
        # Preprocess batch (DLM masking, etc.)
        batch = self.objective_manager.preprocess_batch(batch)

        # Forward pass through student model
        model_outputs = self.forward(**batch)

        # Forward pass through teacher if needed
        teacher_outputs = self._get_teacher_outputs(batch)

        # Compute combined loss via ObjectiveManager
        manager_output = self.objective_manager(model_outputs, batch, teacher_outputs)

        # Check if LDC-MTL meta-optimization is enabled
        final_loss = manager_output.loss
        meta_callback = getattr(self, "_meta_optimizer_callback", None)
        if meta_callback is not None and meta_callback.ldc_mtl is not None:
            # Use LDC-MTL to recompute weighted loss with learned weights
            individual_losses = {
                name: obj_out.loss
                for name, obj_out in manager_output.objective_outputs.items()
            }
            if len(individual_losses) > 1:
                final_loss, ldc_weights = meta_callback.ldc_mtl.compute_weighted_loss(
                    individual_losses
                )
                # Log LDC-MTL weights
                for name, weight in ldc_weights.items():
                    self.log(
                        f"meta/ldc_mtl/objective_weight_{name}",
                        weight,
                        on_step=True,
                        prog_bar=False,
                        sync_dist=True,
                    )

        # Log all metrics (reuse existing get_wandb_metrics!)
        metrics = self.objective_manager.get_wandb_metrics(manager_output, prefix="train")
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        # Log main metrics to progress bar only (not to WandB - already in metrics dict)
        self.log("loss", final_loss, prog_bar=True, sync_dist=True, logger=False)
        if manager_output.perplexity is not None:
            self.log("ppl", manager_output.perplexity, prog_bar=True, sync_dist=True, logger=False)

        # Update token count
        batch_tokens = batch["input_ids"].numel()
        self.tokens_processed += batch_tokens
        self.log("train/tokens", self.tokens_processed, on_step=True, sync_dist=True)

        # Advance curriculum scheduler once per GLOBAL step (optimizer step)
        # NOT per batch - curriculum phases are based on max_steps (global steps)
        # Use _last_curriculum_step to track when we last advanced the curriculum
        current_global_step = self.trainer.global_step
        if not hasattr(self, "_last_curriculum_step"):
            self._last_curriculum_step = -1
        if current_global_step > self._last_curriculum_step:
            self.objective_manager.step_curriculum()
            self._last_curriculum_step = current_global_step

        return final_loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step - computes clean perplexity WITHOUT DLM preprocessing.

        Unlike training, we skip DLM masking to get true language model perplexity
        on held-out validation data (e.g., C4). This provides a fair comparison
        metric across training runs regardless of which objectives are enabled.
        """
        # Forward pass WITHOUT preprocessing (no DLM masking)
        model_outputs = self.forward(**batch)

        # Compute CE loss directly for clean perplexity
        logits = model_outputs["logits"]
        labels = batch.get("labels", batch["input_ids"])

        # Shift for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Cross-entropy loss (ignore padding with -100)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Compute perplexity
        perplexity = torch.exp(loss)

        # Log validation metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/perplexity", perplexity, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)  # For checkpoint callback

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler.

        Supports:
        - "muon" (default): Official PyTorch Muon + AdamW (returns two optimizers)
        - "muonclip": Legacy muon-clip with QK-clipping (single optimizer)
        - "adamw": Standard AdamW, uses 8-bit by default for memory efficiency

        For multi-optimizer setups (Muon), returns [muon_opt, adam_opt] with
        corresponding schedulers for each.
        """
        # Debug: log optimizer_cfg contents
        # Convert DictConfig to dict for reliable access
        from omegaconf import OmegaConf
        if hasattr(self.optimizer_cfg, '_content'):
            opt_dict = OmegaConf.to_container(self.optimizer_cfg, resolve=True)
            logger.info(f"[DEBUG] optimizer_cfg (converted): {opt_dict}")
        else:
            opt_dict = self.optimizer_cfg
            logger.info(f"[DEBUG] optimizer_cfg (already dict): {opt_dict}")

        lr_muon_val = opt_dict.get('lr_muon', 0.02) if isinstance(opt_dict, dict) else self.optimizer_cfg.get('lr_muon', 0.02)
        logger.info(f"[DEBUG] lr_muon to use: {lr_muon_val}")

        optimizer_type = self.optimizer_cfg.get("type", "muon")  # New default
        learning_rate = self.optimizer_cfg.get("learning_rate", 1e-4)
        weight_decay = self.optimizer_cfg.get("weight_decay", 0.1)
        use_8bit = self.optimizer_cfg.get("use_8bit", True)

        if optimizer_type.lower() == "muon":
            # Official PyTorch Muon (2.9+) with combined wrapper
            lr_muon = self.optimizer_cfg.get("lr_muon", 0.02)
            lr_adam = self.optimizer_cfg.get("lr_adam", 3e-4)
            logger.info(f"[LR DEBUG] Creating Muon optimizer with lr_muon={lr_muon}, lr_adam={lr_adam}")
            optimizer = self._create_pytorch_muon_optimizer(
                lr_muon=lr_muon,
                lr_adam=lr_adam,
                weight_decay=self.optimizer_cfg.get("weight_decay", 0.01),
                momentum=self.optimizer_cfg.get("momentum", 0.95),
            )
            # Verify the LR in param_groups
            for i, pg in enumerate(optimizer.param_groups):
                logger.info(f"[LR DEBUG] param_groups[{i}]: lr={pg.get('lr', 'NO_LR')}, num_params={len(pg['params'])}")
        elif optimizer_type.lower() == "muonclip":
            # Legacy muon-clip (keep for backward compatibility)
            optimizer = self._create_muon_optimizer(learning_rate, weight_decay)
        elif optimizer_type.lower() == "adamw_8bit":
            # Explicit 8-bit request (backwards compat)
            optimizer = self._create_adamw_8bit_optimizer(learning_rate, weight_decay)
        elif optimizer_type.lower() in ("adamw", "adam"):
            # AdamW: use 8-bit by default, configurable via use_8bit
            if use_8bit:
                optimizer = self._create_adamw_8bit_optimizer(learning_rate, weight_decay)
            else:
                optimizer = self._create_adamw_optimizer(learning_rate, weight_decay)
        else:
            # Fallback for unknown types
            if use_8bit:
                optimizer = self._create_adamw_8bit_optimizer(learning_rate, weight_decay)
            else:
                optimizer = self._create_adamw_optimizer(learning_rate, weight_decay)

        # Create scheduler if configured (single optimizer path)
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
        """Create Muon optimizer with QK-clipping.

        Uses muon-clip for single GPU (proper QK-clipping support).
        Falls back to muon_fsdp2 for multi-GPU FSDP training.
        """
        import os
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        lr_muon = self.optimizer_cfg.get("lr_muon", learning_rate)
        lr_adam = self.optimizer_cfg.get("lr_adam", learning_rate)
        enable_clipping = self.optimizer_cfg.get("enable_clipping", True)
        clipping_threshold = self.optimizer_cfg.get("clipping_threshold", 50.0)
        clipping_alpha = self.optimizer_cfg.get("clipping_alpha", 0.5)
        momentum = self.optimizer_cfg.get("momentum", 0.95)
        # Layer name mapping for QK-clipping (configurable per model architecture)
        clipping_layers_mapping = self.optimizer_cfg.get(
            "clipping_layers_mapping", {"q_proj": "q_proj", "k_proj": "k_proj"}
        )

        if world_size == 1:
            # Single GPU: use muon-clip with proper QK-clipping
            return self._create_muonclip_optimizer(
                lr_muon=lr_muon,
                lr_adam=lr_adam,
                momentum=momentum,
                weight_decay=weight_decay,  # CRITICAL: Pass weight_decay!
                enable_clipping=enable_clipping,
                clipping_threshold=clipping_threshold,
                clipping_alpha=clipping_alpha,
                clipping_layers_mapping=clipping_layers_mapping,
            )
        else:
            # Multi-GPU: use muon_fsdp2 for FSDP compatibility
            return self._create_muon_fsdp_optimizer(lr_muon, lr_adam)

    def _create_muonclip_optimizer(
        self,
        lr_muon: float,
        lr_adam: float,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        enable_clipping: bool = True,
        clipping_threshold: float = 50.0,
        clipping_alpha: float = 0.5,
        clipping_layers_mapping: dict = None,
    ):
        """Create MuonClip optimizer for single GPU with QK-clipping.

        Reference: https://github.com/GAD-cell/muon-clip
        CRITICAL: model.train() must be called AFTER creating optimizer!
        MuonClip.__init__() overrides model.train() to register hooks.
        """
        from muon import MuonClip, MuonConfig

        # Default to LLaMA-style naming if not specified
        if clipping_layers_mapping is None:
            clipping_layers_mapping = {"q_proj": "q_proj", "k_proj": "k_proj"}

        # Build MuonConfig with configurable clipping_layers_mapping
        # CRITICAL: weight_decay is essential for Muon to prevent weights from growing too large
        # See: https://arxiv.org/abs/2502.16982 "Muon is Scalable for LLM Training"
        muon_config = MuonConfig(
            unified_lr=False,
            lr_muon=lr_muon,
            lr_adam=lr_adam,
            muon_beta=momentum,
            muon_decay=weight_decay,  # CRITICAL: Apply weight decay to Muon params
            adam_betas=(0.9, 0.999),
            adam_decay=weight_decay,  # CRITICAL: Apply weight decay to Adam params
            adam_eps=1e-8,
            ns_steps=5,  # Newton-Schulz iterations
            enable_clipping=enable_clipping,
            clipping_layers_mapping=clipping_layers_mapping,
            clipping_threshold=clipping_threshold,
            clipping_alpha=clipping_alpha,
            log_max_logits=False,  # Disabled: requires TensorBoard writer (we use WandB)
        )
        logger.info(f"MuonClip weight_decay: muon_decay={weight_decay}, adam_decay={weight_decay}")

        # MuonClip needs HuggingFace model config for architecture info
        model_config = getattr(self.model, "config", None)
        if model_config is None:
            raise ValueError("MuonClip requires model.config (HuggingFace config)")

        # Create optimizer - this overrides model.train()/eval() via override_model()
        optimizer = MuonClip(self.model, model_config, muon_config)

        # WORKAROUND for upstream bug in muon-clip's HookRecorder:
        # The remove_hooks() method removes hook handles but doesn't reset is_registered=False.
        # This causes hooks to never be re-registered after model.eval() → model.train() cycles
        # (like those in Lightning's BatchSizeFinder), leading to KeyError in optimizer.step().
        # We patch remove_hooks to reset the flag properly.
        # Reference: https://github.com/GAD-cell/muon-clip
        if hasattr(optimizer, "hook_recorder"):
            original_remove = optimizer.hook_recorder.remove_hooks

            def patched_remove_hooks():
                original_remove()
                # Reset flag so hooks can be re-registered on next model.train() call
                optimizer.hook_recorder.is_registered = False

            optimizer.hook_recorder.remove_hooks = patched_remove_hooks
            logger.info("Patched MuonClip hook_recorder.remove_hooks to fix is_registered flag bug")

        # WORKAROUND for upstream bug in muon-clip's flush_metrics():
        # The flush_metrics() method unconditionally tries to use self.writer.add_scalar(),
        # even when log_max_logits=False. The writer attribute is never initialized.
        # We add a no-op writer to prevent AttributeError.
        # Reference: https://github.com/GAD-cell/muon-clip
        class _NoOpWriter:
            """Dummy TensorBoard writer that ignores all calls."""

            def add_scalar(self, *args, **kwargs):
                pass

        optimizer.writer = _NoOpWriter()
        logger.info("Added no-op writer to MuonClip to fix missing writer bug")

        # CRITICAL: Call model.train() AFTER creating optimizer to register hooks!
        # MuonClip.__init__() overrides model.train() with hook registration logic.
        # See: https://github.com/GAD-cell/muon-clip
        self.model.train()
        logger.info(f"MuonClip hooks registered: {len(optimizer.hook_recorder.handles)} hooks active")

        logger.info(
            f"Created MuonClip optimizer (single GPU): "
            f"lr_muon={lr_muon:.2e}, lr_adam={lr_adam:.2e}, weight_decay={weight_decay}, "
            f"clipping={enable_clipping} (threshold={clipping_threshold}, alpha={clipping_alpha})"
        )
        return optimizer

    def _create_muon_fsdp_optimizer(self, lr_muon: float, lr_adam: float):
        """Create Muon optimizer for multi-GPU FSDP via muon_fsdp2."""
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

        optimizer = Muon([
            {"params": muon_params, "lr": lr_muon, "use_muon": True},
            {"params": adam_params, "lr": lr_adam, "use_muon": False},
        ])

        logger.info(
            f"Created Muon FSDP optimizer: {len(muon_params)} Muon params, "
            f"{len(adam_params)} Adam params, lr_muon={lr_muon:.2e}, lr_adam={lr_adam:.2e}"
        )
        return optimizer

    def _create_adamw_8bit_optimizer(self, learning_rate: float, weight_decay: float):
        """Create 8-bit AdamW optimizer via bitsandbytes.

        Falls back to standard AdamW if bitsandbytes is not available.
        """
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
            logger.info(f"Created AdamW 8-bit optimizer: lr={learning_rate:.2e}")
            return optimizer
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to standard AdamW")
            return self._create_adamw_optimizer(learning_rate, weight_decay)

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

    def _create_pytorch_muon_optimizer(
        self,
        lr_muon: float = 0.02,
        lr_adam: float = 3e-4,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
    ):
        """Create combined Muon + AdamW optimizer using official PyTorch Muon (2.9+).

        Returns a CombinedOptimizer wrapper that presents a single optimizer interface
        to Lightning while internally managing both Muon and AdamW.

        Parameter split (following Keller Jordan's recommendations):
        - Muon: 2D+ weights (attention, FFN projections), excluding embeddings/lm_head
        - AdamW: Embeddings, biases, lm_head, LayerNorm gains

        Default learning rates:
        - Muon: 0.02 (has built-in muP scaling, doesn't need retuning)
        - AdamW: 3e-4 (standard for embeddings/biases)
        """
        muon_params = []
        adam_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Muon: 2D+ weights, excluding embeddings and lm_head
            if param.ndim >= 2 and "embed" not in name.lower() and "lm_head" not in name.lower():
                muon_params.append(param)
            else:
                adam_params.append(param)

        muon_opt = torch.optim.Muon(
            muon_params,
            lr=lr_muon,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        adam_opt = torch.optim.AdamW(
            adam_params,
            lr=lr_adam,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

        # Wrap both optimizers in a combined interface for Lightning
        combined = CombinedMuonAdamWOptimizer(muon_opt, adam_opt, lr_muon)

        logger.info(
            f"Created PyTorch Muon + AdamW: {len(muon_params)} Muon params (lr={lr_muon}), "
            f"{len(adam_params)} AdamW params (lr={lr_adam}), weight_decay={weight_decay}"
        )
        return combined

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
            # Get peak LR - handle both standard optimizers and MuonClip
            if hasattr(optimizer, "defaults") and "lr" in optimizer.defaults:
                peak_lr = optimizer.defaults["lr"]
            elif hasattr(optimizer, "param_groups") and optimizer.param_groups:
                peak_lr = optimizer.param_groups[0].get("lr", 0.02)
            else:
                peak_lr = 0.02  # Fallback to Muon default
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=peak_lr * min_lr_ratio,
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

        elif scheduler_type == "wsd":
            # Warmup-Stable-Decay (WSD) scheduler
            # Used by DeepSeek-V3, maintains high LR during stable phase
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, LambdaLR, CosineAnnealingLR

            decay_ratio = self.scheduler_cfg.get("decay_ratio", 0.2)
            decay_type = self.scheduler_cfg.get("decay_type", "linear")
            num_decay_steps = int(max_steps * decay_ratio)
            num_stable_steps = max(1, max_steps - warmup_steps - num_decay_steps)

            logger.info(
                f"WSD schedule: {warmup_steps} warmup + {num_stable_steps} stable + "
                f"{num_decay_steps} decay = {max_steps} steps"
            )

            # Phase 1: Warmup (linear ramp to peak LR)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=max(1, warmup_steps),
            )

            # Phase 2: Stable (constant peak LR)
            stable_scheduler = LambdaLR(optimizer, lambda _: 1.0)

            # Phase 3: Decay (linear or cosine decay to min_lr)
            # Get peak LR - handle both standard optimizers and MuonClip
            if hasattr(optimizer, "defaults") and "lr" in optimizer.defaults:
                peak_lr = optimizer.defaults["lr"]
            elif hasattr(optimizer, "param_groups") and optimizer.param_groups:
                peak_lr = optimizer.param_groups[0].get("lr", 0.02)
            else:
                peak_lr = 0.02  # Fallback to Muon default
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
                    end_factor=max(min_lr_ratio, 1e-8),
                    total_iters=max(1, num_decay_steps),
                )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, stable_scheduler, decay_scheduler],
                milestones=[warmup_steps, warmup_steps + num_stable_steps],
            )
            logger.info(
                f"Created WSD scheduler: warmup={warmup_steps}, stable={num_stable_steps}, "
                f"decay={num_decay_steps} ({decay_type}), min_lr_ratio={min_lr_ratio}"
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
        """Custom gradient clipping - handles FSDP compatibility."""
        if self.gradient_clipping <= 0:
            return

        # Check if we're using FSDP strategy (doesn't support 'norm' clipping)
        strategy = self.trainer.strategy.__class__.__name__
        if "FSDP" in strategy:
            # FSDP doesn't support norm-based gradient clipping via Lightning
            # Use value clipping instead or skip (FSDP uses gradient scaling internally)
            logger.debug("Skipping gradient clipping for FSDP (use gradient_clip_val at trainer level)")
            return

        # Use Lightning's built-in clipping for non-FSDP strategies
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
        """Restore extra state from checkpoint.

        Respects resume_cfg flags:
        - load_scheduler_state: If False, skip loading curriculum state too
          (curriculum is like a scheduler - it controls training phases)
        - load_training_state: If False, reset tokens_processed
        """
        # Check resume config flags (default to True for backwards compatibility)
        load_scheduler_state = self.resume_cfg.get("load_scheduler_state", True)
        load_training_state = self.resume_cfg.get("load_training_state", True)

        # Restore tokens processed (unless doing fresh training state)
        if load_training_state:
            self.tokens_processed = checkpoint.get("tokens_processed", 0)
        else:
            self.tokens_processed = 0
            logger.info("Reset tokens_processed to 0 (load_training_state=false)")

        # Restore curriculum state (unless doing fresh scheduler state)
        # Curriculum is treated like a scheduler since it controls training phases
        if self.objective_manager.curriculum is not None and "curriculum_state" in checkpoint:
            if load_scheduler_state:
                self.objective_manager.curriculum.load_state_dict(checkpoint["curriculum_state"])
                logger.info(
                    f"Loaded curriculum state: step={checkpoint['curriculum_state'].get('current_step', 0)}"
                )
            else:
                # Reset curriculum to step 0 for fresh start
                logger.info(
                    "Skipping curriculum state load (load_scheduler_state=false) - "
                    "curriculum will start from step 0"
                )
