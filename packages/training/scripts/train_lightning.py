#!/usr/bin/env python3
"""PyTorch Lightning training entry point for WrinkleFree.

This is the new, simplified training script using PyTorch Lightning.
It provides:
- Auto batch size scaling via Tuner.scale_batch_size() (standard Lightning approach)
- Built-in DDP/FSDP support
- Clean separation of concerns
- All objectives work unchanged (DLM, LRC, distillation)

Usage:
    uv run python scripts/train_lightning.py model=smollm2_135m training=base
    uv run python scripts/train_lightning.py model=qwen3_4b training=bitdistill_full

With auto batch size:
    uv run python scripts/train_lightning.py model=smollm2_135m training=base \
        training.auto_batch_size=true
"""

# Force unbuffered stdout/stderr for real-time log visibility in cloud environments
# This must be set before any other imports to ensure all output is unbuffered
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata
from omegaconf._utils import ValueKind
from transformers import AutoModelForCausalLM, AutoTokenizer

# PyTorch 2.6+ requires explicit safe globals for omegaconf types in checkpoints
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, ValueKind])

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wf_arch.conversion import auto_convert_if_needed
from wf_train.lightning import (
    GCSCheckpointCallback,
    LambdaWarmupCallback,
    MuonClipInitCallback,
    QKClipCallback,
    RunManagerCallback,
    TokenCountCallback,
    WrinkleFreeDataModule,
    WrinkleFreeLightningModule,
    ZClipCallback,
)
from wf_train.meta import MetaOptimizerCallback, MetaOptimizationConfig
from wf_train.objectives import create_objective_manager
from wf_train.teachers import HiddenStateTeacher
from wf_train.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)

print("[DEBUG] Imports complete, script loaded!", flush=True)


def resolve_checkpoint_path(
    cfg: DictConfig,
    stage: str = "stage2",
) -> Path | str | None:
    """Resolve MODEL checkpoint path from local or HuggingFace.

    NOTE: This is for MODEL loading only. For Lightning training resume,
    use find_lightning_resume_checkpoint() instead.

    Priority:
    1. Explicit model.path in config (local safetensors)
    2. HuggingFace Hub (model.pretrained_name or model.name)

    Args:
        cfg: Hydra config
        stage: Training stage (unused, kept for compatibility)

    Returns:
        Path to model checkpoint or HF model name
    """
    # Check for model.path (local safetensors)
    model_path = cfg.model.get("path")
    if model_path:
        model_path = Path(model_path)
        if model_path.exists():
            logger.info(f"Using local model path: {model_path}")
            return model_path

    # Fall back to HuggingFace Hub
    hf_name = cfg.model.get("pretrained_name") or cfg.model.get("name")
    if hf_name:
        logger.info(f"Using HuggingFace model: {hf_name}")
        return hf_name

    return None


def find_lightning_resume_checkpoint(cfg: DictConfig) -> str | None:
    """Find Lightning checkpoint for resuming training.

    Searches for .ckpt files in:
    1. Explicit resume.ckpt_path in config
    2. GCS bucket (lightning_checkpoint/) - downloads locally
    3. Local output directory

    Args:
        cfg: Hydra config

    Returns:
        Path to LOCAL .ckpt file, or None if not found
        (GCS checkpoints are downloaded to local cache)
    """
    import subprocess

    def download_gcs_checkpoint(gcs_path: str, local_dir: Path) -> str | None:
        """Download GCS checkpoint to local directory."""
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / "resume.ckpt"

        logger.info(f"Downloading GCS checkpoint: {gcs_path} -> {local_path}")
        try:
            # Ensure gsutil is in PATH (may be installed in ~/google-cloud-sdk/bin)
            env = os.environ.copy()
            gcloud_bin = Path.home() / "google-cloud-sdk" / "bin"
            if gcloud_bin.exists():
                env["PATH"] = f"{gcloud_bin}:{env.get('PATH', '')}"

            result = subprocess.run(
                ["gsutil", "-m", "cp", gcs_path, str(local_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for large checkpoints
                env=env,
            )
            if result.returncode == 0 and local_path.exists():
                logger.info(f"Successfully downloaded checkpoint ({local_path.stat().st_size / 1e6:.1f} MB)")
                return str(local_path)
            else:
                logger.warning(f"gsutil download failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.warning("GCS checkpoint download timed out")
            return None
        except Exception as e:
            logger.warning(f"GCS checkpoint download failed: {e}")
            return None

    # Check for explicit resume path
    # First check training.resume.checkpoint_path (from unified.yaml)
    # Then fall back to top-level resume.ckpt_path (legacy)
    resume_cfg = cfg.training.get("resume", {}) if hasattr(cfg, "training") else cfg.get("resume", {})
    explicit_path = resume_cfg.get("checkpoint_path") or resume_cfg.get("ckpt_path")
    if explicit_path:
        explicit_path_str = str(explicit_path)
        if explicit_path_str.startswith("gs://"):
            # Download GCS checkpoint to local cache
            cache_dir = Path(cfg.output_dir) / ".resume_cache"
            local_path = download_gcs_checkpoint(explicit_path_str, cache_dir)
            if local_path:
                return local_path
            logger.warning(f"Failed to download explicit GCS checkpoint: {explicit_path}")
        elif Path(explicit_path).exists():
            logger.info(f"Resume from explicit local checkpoint: {explicit_path}")
            return str(explicit_path)
        else:
            logger.warning(f"Explicit resume checkpoint not found: {explicit_path}")

    # Try GCS bucket for lightning checkpoints
    gcs_config = cfg.get("gcs", {})
    if gcs_config.get("enabled", False):
        bucket = gcs_config.get("bucket", "wrinklefree-checkpoints")
        experiment_name = cfg.get("experiment_name", "default")
        gcs_prefix = f"gs://{bucket}/checkpoints/{experiment_name}/lightning_checkpoint/checkpoints"

        try:
            # List checkpoints in GCS, find the latest step
            result = subprocess.run(
                ["gsutil", "ls", gcs_prefix],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Parse step directories (e.g., step_000500/)
                dirs = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                step_dirs = [d for d in dirs if "/step_" in d]
                if step_dirs:
                    # Sort by step number (extract from step_XXXXXX)
                    step_dirs.sort(key=lambda x: int(x.split("/step_")[-1].rstrip("/")))
                    latest_step_dir = step_dirs[-1].rstrip("/")
                    # Look for last.ckpt in the step directory
                    gcs_ckpt_path = f"{latest_step_dir}/last.ckpt"
                    # Verify file exists
                    check_result = subprocess.run(
                        ["gsutil", "ls", gcs_ckpt_path],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if check_result.returncode == 0:
                        # Download to local cache
                        cache_dir = Path(cfg.output_dir) / ".resume_cache"
                        local_path = download_gcs_checkpoint(gcs_ckpt_path, cache_dir)
                        if local_path:
                            return local_path
        except subprocess.TimeoutExpired:
            logger.warning("GCS checkpoint search timed out")
        except Exception as e:
            logger.debug(f"GCS checkpoint search failed: {e}")

    # Try local checkpoint directory
    local_ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    if local_ckpt_dir.exists():
        # Find latest step checkpoint
        ckpt_files = list(local_ckpt_dir.glob("step_*/last.ckpt"))
        if ckpt_files:
            latest = max(ckpt_files, key=lambda p: int(p.parent.name.split("_")[-1]))
            logger.info(f"Found local resume checkpoint: {latest}")
            return str(latest)
        # Also check for last.ckpt directly
        last_ckpt = local_ckpt_dir / "last.ckpt"
        if last_ckpt.exists():
            logger.info(f"Found local resume checkpoint: {last_ckpt}")
            return str(last_ckpt)

    logger.info("No resume checkpoint found - starting fresh")
    return None


def load_model_and_tokenizer(cfg: DictConfig, device: str = "cuda"):
    """Load model and tokenizer from config.

    Uses resolve_checkpoint_path for intelligent checkpoint resolution:
    - Local path → GCS → HuggingFace Hub

    Handles special stages:
    - lrc_calibration: Converts BitLinear → BitLinearLRC and freezes non-LRC params
    """
    # Resolve checkpoint path (local > GCS > HuggingFace)
    stage = cfg.training.get("stage", "stage2")
    model_path = resolve_checkpoint_path(cfg, stage=stage)

    # Fallback to model.name if nothing found
    if model_path is None:
        model_path = cfg.model.name

    print(f"[DEBUG] Loading model from: {model_path}", flush=True)
    logger.info(f"Loading model from: {model_path}")

    # Convert Path to string for transformers compatibility
    model_path_str = str(model_path) if isinstance(model_path, Path) else model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path_str,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Auto-convert to BitNet if needed
    if cfg.training.get("auto_convert", {}).get("enabled", True):
        exclude_layers = cfg.training.get("auto_convert", {}).get("exclude_layers", None)
        # insert_subln: Add SubLN before projections (per BitNet paper)
        insert_subln = cfg.training.get("auto_convert", {}).get("insert_subln", False)
        # use_hadamard: Use HBitLinear for o_proj/down_proj (BitNet v2)
        use_hadamard = cfg.training.get("auto_convert", {}).get("use_hadamard", False)
        model = auto_convert_if_needed(
            model,
            hidden_size=cfg.model.hidden_size,
            intermediate_size=cfg.model.intermediate_size,
            exclude_layers=exclude_layers,
            insert_subln=insert_subln,
            use_hadamard=use_hadamard,
        )
        print(f"[DEBUG] Auto-converted to BitNet (insert_subln={insert_subln}, use_hadamard={use_hadamard})", flush=True)
        logger.info(f"Auto-converted model to BitNet (insert_subln={insert_subln}, use_hadamard={use_hadamard})")

    # ===========================================================================
    # STEP 1: Handle Salient Columns (AWQ-style) if enabled
    # Must be applied BEFORE LoRA so LoRA can wrap BitLinearSalient layers
    # ===========================================================================
    print("[DEBUG] Starting salient/LoRA setup...", flush=True)
    salient_cfg = cfg.training.get("salient", {})
    salient_enabled = salient_cfg.get("enabled", False)
    if salient_enabled:
        print("[DEBUG] Salient enabled, starting calibration...", flush=True)
        try:
            print("[DEBUG] Importing salient modules...", flush=True)
            from wf_arch import (
                calibrate_salient_columns,
                convert_bitlinear_to_salient,
                get_salient_stats,
            )
            from wf_data.data.factory import create_dataloader
            print("[DEBUG] Imports complete", flush=True)

            salient_ratio = salient_cfg.get("ratio", 0.01)
            calibration_samples = salient_cfg.get("calibration_samples", 128)
            calibration_data = salient_cfg.get("calibration_data", "fineweb")

            print(f"[DEBUG] Salient config: ratio={salient_ratio}, samples={calibration_samples}", flush=True)

            # Use synthetic data for fast calibration (random tokens)
            # This avoids slow streaming dataset startup while still providing
            # representative activation statistics for saliency scoring
            vocab_size = tokenizer.vocab_size
            seq_len = cfg.training.max_seq_length
            batch_size = 4
            print(f"[DEBUG] Synthetic data: vocab={vocab_size}, seq_len={seq_len}, batch={batch_size}", flush=True)

            def synthetic_dataloader():
                """Generate random token batches for calibration."""
                while True:
                    # Random tokens from vocabulary
                    input_ids = torch.randint(
                        0, vocab_size, (batch_size, seq_len), dtype=torch.long
                    )
                    yield {"input_ids": input_ids}

            calib_dataloader = synthetic_dataloader()
            print("[DEBUG] Created synthetic dataloader", flush=True)

            # Run calibration on GPU in float32 to avoid dtype mismatches
            # (BitLinear outputs float32, but lm_head may be bfloat16)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            original_dtype = next(model.parameters()).dtype
            print(f"[DEBUG] Moving model to {device} (float32 for calibration, was {original_dtype})...", flush=True)
            model = model.to(device=device, dtype=torch.float32)
            print(f"[DEBUG] Model on {device}, starting calibration...", flush=True)
            saliency_scores, salient_indices = calibrate_salient_columns(
                model,
                calib_dataloader,
                salient_ratio=salient_ratio,
                num_samples=calibration_samples,
                device=device,
            )
            print("[DEBUG] Calibration complete!", flush=True)
            # Convert back to original dtype after calibration
            model = model.to(dtype=original_dtype)
            print(f"[DEBUG] Model back to {original_dtype}", flush=True)

            # Convert BitLinear -> BitLinearSalient with calibrated indices
            model = convert_bitlinear_to_salient(
                model,
                salient_ratio=salient_ratio,
                salient_indices=salient_indices,
                saliency_scores=saliency_scores,
            )

            # Log salient statistics
            salient_stats = get_salient_stats(model)
            logger.info(
                f"Salient: {salient_stats['num_salient_layers']} layers, "
                f"{salient_stats['total_salient_columns']} salient columns "
                f"({salient_stats['num_calibrated_layers']} calibrated), "
                f"avg ratio={salient_stats['average_salient_ratio']*100:.2f}%"
            )
            # Model stays on GPU - Lightning handles device placement

        except ImportError as e:
            logger.error(f"Salient requires wf_arch and wf_data packages: {e}")
            raise

    # ===========================================================================
    # STEP 2: Handle LoRA (Low-Rank Adaptation) if enabled
    # Applied AFTER salient so LoRA can wrap BitLinearSalient layers
    # This is the NEW composable wrapper pattern (replaces legacy BitLinearLRC)
    # ===========================================================================
    print("[DEBUG] Salient setup complete, starting LoRA setup...", flush=True)
    lora_cfg = cfg.training.get("lora", {})
    lora_enabled = lora_cfg.get("enabled", False)
    print(f"[DEBUG] LoRA enabled: {lora_enabled}", flush=True)

    # Also check legacy lrc.enabled for backward compatibility
    lrc_cfg = cfg.training.get("lrc", {})
    lrc_enabled = lrc_cfg.get("enabled", False)

    if lora_enabled:
        # New LoRA adapter pattern (recommended)
        try:
            from wf_arch import LoRAConfig, add_lora_to_model, freeze_base_model, get_lora_stats

            rank_percentage = lora_cfg.get("rank_percentage", 0.1)
            rank = lora_cfg.get("rank", None)
            init_method = lora_cfg.get("init_method", "kaiming")
            alpha = lora_cfg.get("alpha", 1.0)
            dropout = lora_cfg.get("dropout", 0.0)
            target_modules = lora_cfg.get("target_modules", None)
            quantized = lora_cfg.get("quantized", False)
            quant_bits = lora_cfg.get("quant_bits", 4)
            quant_group_size = lora_cfg.get("quant_group_size", 32)
            freeze_base = lora_cfg.get("freeze_base", True)  # Default: freeze base, train only LoRA

            # Build LoRA config
            lora_config = LoRAConfig(
                rank=rank,
                rank_percentage=rank_percentage,
                alpha=alpha,
                dropout=dropout,
                init_method=init_method,
                quantized=quantized,
                quant_bits=quant_bits,
                quant_group_size=quant_group_size,
                target_modules=target_modules,
            )

            print(
                f"[DEBUG] LoRA: Adding adapters (rank={rank or f'{rank_percentage*100:.0f}%'}, "
                f"init={init_method}, alpha={alpha}, freeze_base={freeze_base})", flush=True
            )
            logger.info(
                f"LoRA: Adding adapters (rank={rank or f'{rank_percentage*100:.0f}%'}, "
                f"init={init_method}, alpha={alpha})"
            )

            # Add LoRA wrappers to model
            print("[DEBUG] Calling add_lora_to_model...", flush=True)
            model = add_lora_to_model(model, lora_config)
            print("[DEBUG] add_lora_to_model complete!", flush=True)

            # Optionally freeze base parameters (train only LoRA)
            if freeze_base:
                freeze_stats = freeze_base_model(model)
                logger.info(
                    f"LoRA: Trainable={freeze_stats['trainable']:,}, "
                    f"Frozen={freeze_stats['frozen']:,}"
                )
            else:
                # Count all trainable params
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"LoRA: Full model trainable ({trainable:,} params) - base NOT frozen")

            # Log LoRA statistics
            lora_stats = get_lora_stats(model)
            print(
                f"[DEBUG] LoRA: {lora_stats['num_lora_layers']} layers, "
                f"avg_rank={lora_stats['average_rank']:.1f}, "
                f"params={lora_stats['total_lora_params']:,}", flush=True
            )
            logger.info(
                f"LoRA: {lora_stats['num_lora_layers']} layers, "
                f"avg_rank={lora_stats['average_rank']:.1f}, "
                f"params={lora_stats['total_lora_params']:,}"
            )

            # Enable input require grads for frozen embedding support
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                logger.info("LoRA: Enabled input_require_grads for frozen embedding support")
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                if hasattr(model, "get_input_embeddings"):
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                    logger.info("LoRA: Added require_grad hook to embedding layer")

        except ImportError as e:
            logger.error(f"LoRA requires wf_arch package: {e}")
            raise

    elif lrc_enabled:
        # Legacy LRC path (backward compatibility)
        # Deprecated: Use lora.enabled instead
        logger.warning(
            "DEPRECATED: lrc.enabled is deprecated. Use lora.enabled instead. "
            "The old BitLinearLRC has been moved to wf_arch.layers._legacy"
        )
        try:
            from wf_arch import QLRCConfig, convert_bitlinear_to_lrc, freeze_model_except_lrc

            rank_percentage = lrc_cfg.get("rank_percentage", 0.1)
            init_method = lrc_cfg.get("init_method", "zeros")
            keep_original_weight = lrc_cfg.get("keep_original_weight", True)
            trainable_weight = lrc_cfg.get("trainable_weight", False)

            # Parse QLRC config
            qlrc_cfg = lrc_cfg.get("qlrc", {})
            qlrc_config = None
            if qlrc_cfg.get("enabled", False):
                qlrc_config = QLRCConfig(
                    enabled=True,
                    bits=qlrc_cfg.get("bits", 4),
                    group_size=qlrc_cfg.get("group_size", 32),
                )

            logger.info(
                f"LRC (legacy): Converting BitLinear → BitLinearLRC "
                f"(rank={rank_percentage*100:.0f}%, init={init_method})"
            )

            model = convert_bitlinear_to_lrc(
                model,
                rank_percentage=rank_percentage,
                init_method=init_method,
                keep_original_weight=keep_original_weight,
                trainable_weight=trainable_weight,
                qlrc_config=qlrc_config,
            )

            if not trainable_weight:
                freeze_stats = freeze_model_except_lrc(model)
                logger.info(
                    f"LRC: Trainable={freeze_stats['trainable']:,}, "
                    f"Frozen={freeze_stats['frozen']:,}"
                )

                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                    logger.info("LRC: Enabled input_require_grads for frozen embedding support")
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    if hasattr(model, "get_input_embeddings"):
                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                        logger.info("LRC: Added require_grad hook to embedding layer")

        except ImportError as e:
            logger.error(f"LRC requires wf_arch package: {e}")
            raise

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    logger.info(
        f"Model: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable), "
        f"{param_memory_mb:.1f}MB"
    )

    # Enable gradient checkpointing if configured (saves significant VRAM)
    # CRITICAL: Must use use_reentrant=False for LRC training!
    # use_reentrant=True (old default) breaks with frozen embeddings because it doesn't
    # preserve requires_grad state during recomputation.
    # See: https://pytorch.org/docs/stable/checkpoint.html
    memory_cfg = cfg.training.get("memory", {})
    gc_enabled = memory_cfg.get("gradient_checkpointing", False)
    if gc_enabled:
        if hasattr(model, "gradient_checkpointing_enable"):
            # Use use_reentrant=False for compatibility with LRC (frozen embeddings)
            # HuggingFace models support gradient_checkpointing_kwargs since transformers 4.35+
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                logger.info("Gradient checkpointing enabled (use_reentrant=False)")
            except TypeError:
                # Older transformers version doesn't support kwargs
                model.gradient_checkpointing_enable()
                logger.warning("Gradient checkpointing enabled (legacy) - upgrade transformers to 4.35+ for LRC")
        else:
            logger.warning("Model does not support gradient_checkpointing_enable()")

    # Apply torch.compile if enabled (38% speedup on most hardware)
    torch_compile_cfg = cfg.training.get("torch_compile", {})
    if torch_compile_cfg.get("enabled", False):
        compile_mode = torch_compile_cfg.get("mode", "default")
        fullgraph = torch_compile_cfg.get("fullgraph", False)

        try:
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_compile_cache")
            model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph)
            logger.info(f"torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e} - continuing without")

    return model, tokenizer


def _get_distill_status(cfg: DictConfig) -> tuple[bool, bool, float]:
    """Check distillation configuration status.

    Returns:
        Tuple of (needs_teacher, needs_lazy_load, initial_weight):
        - needs_teacher: True if distill is configured and will be used at some point
        - needs_lazy_load: True if distill weight starts at 0 but becomes non-zero later
        - initial_weight: The initial distill weight (0 if curriculum starts with 0)
    """
    distill_cfg = cfg.training.get("objectives", {}).get("distill", {})

    # Check if distill objective is enabled
    if not distill_cfg.get("enabled", False):
        return False, False, 0.0

    # Check if any sub-component is enabled
    components = ["hidden", "logits", "attention", "lrc"]
    any_component_enabled = any(
        distill_cfg.get(comp, {}).get("enabled", False) for comp in components
    )
    if not any_component_enabled:
        return False, False, 0.0

    # Check base weight
    base_weight = distill_cfg.get("weight", 1.0)

    # Check curriculum phases for distill weights
    curriculum_cfg = cfg.training.get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        phases = curriculum_cfg.get("phases", [])
        if phases:
            # Get initial weight from first phase
            first_phase = phases[0]
            initial_weight = first_phase.get("objectives", {}).get("distill", base_weight)

            # Get max weight across all phases
            max_curriculum_weight = 0.0
            phase_with_max = None
            for phase in phases:
                phase_objectives = phase.get("objectives", {})
                phase_weight = phase_objectives.get("distill", base_weight)
                if phase_weight > max_curriculum_weight:
                    max_curriculum_weight = phase_weight
                    phase_with_max = phase.get("name", "unknown")

            if max_curriculum_weight <= 0:
                logger.info("Distill weight is 0 in all curriculum phases - no teacher needed")
                return False, False, 0.0

            # Check if we need lazy loading (starts at 0, becomes non-zero later)
            if initial_weight <= 0 and max_curriculum_weight > 0:
                logger.info(
                    f"Distill weight starts at 0 but reaches {max_curriculum_weight} "
                    f"in phase '{phase_with_max}' - will lazy load teacher when needed"
                )
                return True, True, initial_weight

            return True, False, initial_weight

    # No curriculum - use base weight
    if base_weight <= 0:
        logger.info("Distill objective weight is 0 - no teacher needed")
        return False, False, 0.0

    return True, False, base_weight


def prepare_teacher_config(cfg: DictConfig) -> dict | None:
    """Prepare teacher configuration for lazy loading.

    Returns a config dict that can be passed to WrinkleFreeLightningModule
    for lazy teacher loading, or None if teacher is not needed.

    Args:
        cfg: Hydra config

    Returns:
        Teacher config dict or None
    """
    needs_teacher, _, _ = _get_distill_status(cfg)
    if not needs_teacher:
        return None

    teacher_cfg = cfg.training.get("teacher", {})
    distill_cfg = cfg.training.get("objectives", {}).get("distill", {})

    # Determine teacher model name
    teacher_model_name = teacher_cfg.get("model_name")
    if not teacher_model_name:
        # Fall back to student model name (distill from original fp16 weights)
        teacher_model_name = cfg.model.get("pretrained_name") or cfg.model.get("name")

    if not teacher_model_name:
        logger.warning("No teacher model name specified and no student model name found")
        return None

    # Determine if attention distillation needs eager attention
    attention_enabled = distill_cfg.get("attention", {}).get("enabled", False)

    return {
        "model_name": teacher_model_name,
        "fp16": teacher_cfg.get("fp16", True),
        "offload_to_cpu": teacher_cfg.get("offload_to_cpu", False),
        "load_in_4bit": teacher_cfg.get("load_in_4bit", False),
        "use_flash_attention": teacher_cfg.get("use_flash_attention", not attention_enabled),
        "use_eager_attention": teacher_cfg.get("use_eager_attention", attention_enabled),
        "distill_cfg": dict(distill_cfg) if distill_cfg else {},
    }


def load_teacher_model(cfg: DictConfig) -> HiddenStateTeacher | None:
    """Load teacher model immediately if distillation needs it from the start.

    Skips loading if:
    - distill objective is disabled or weight is 0
    - all distill sub-components are disabled
    - curriculum has distill weight 0 in the initial phase (will lazy load later)

    Args:
        cfg: Hydra config

    Returns:
        HiddenStateTeacher or None if not needed immediately
    """
    needs_teacher, needs_lazy_load, initial_weight = _get_distill_status(cfg)

    if not needs_teacher:
        return None

    if needs_lazy_load:
        logger.info("Teacher will be lazy loaded when distill weight becomes non-zero")
        return None

    # Load teacher immediately
    teacher_config = prepare_teacher_config(cfg)
    if teacher_config is None:
        return None

    logger.info(f"Loading teacher model: {teacher_config['model_name']}")

    teacher = HiddenStateTeacher(
        model_name_or_path=teacher_config["model_name"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        load_in_fp16=teacher_config["fp16"],
        offload_to_cpu=teacher_config["offload_to_cpu"],
        load_in_4bit=teacher_config["load_in_4bit"],
        use_flash_attention=teacher_config["use_flash_attention"],
        use_eager_attention=teacher_config["use_eager_attention"],
    )
    return teacher


def create_callbacks(cfg: DictConfig) -> list:
    """Create Lightning callbacks from config."""
    callbacks = []

    # Run manager for GCS auto-resume with fingerprinting
    skip_recovery = cfg.get("skip_recovery", False)
    run_manager_cb = RunManagerCallback(
        config=cfg,
        skip_recovery=skip_recovery,
        skip_completed=cfg.get("resume", {}).get("skip_completed", True),
    )
    callbacks.append(run_manager_cb)

    # Note: Auto batch size scaling is now handled via Tuner.scale_batch_size()
    # before trainer.fit() - see main() function. This is the standard Lightning approach.

    # Checkpointing - save every N steps
    save_interval = cfg.training.checkpoint.get("save_interval", 500)
    checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint dir: {checkpoint_dir}, save_interval: {save_interval}")

    callbacks.append(
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="step_{step:06d}",
            save_top_k=-1,  # Keep all checkpoints
            every_n_train_steps=save_interval,
            save_last=True,
            verbose=True,  # Log when saving
        )
    )

    # GCS upload with DLM config
    if cfg.get("gcs", {}).get("enabled", False):
        # Build DLM config for inference compatibility
        dlm_cfg = cfg.training.get("objectives", {}).get("dlm", {})
        dlm_config = None
        if dlm_cfg.get("enabled", False):
            dlm_config = {
                "mask_token_id": dlm_cfg.get("mask_token_id", 0),
                "mask_prob": dlm_cfg.get("mask_prob", 0.15),
                "ignore_index": dlm_cfg.get("ignore_index", -100),
                "training_method": cfg.training.get("stage", "unified-dlm"),
            }
            logger.info(f"DLM config for inference: {dlm_config}")

        callbacks.append(
            GCSCheckpointCallback(
                bucket=cfg.gcs.bucket,
                experiment_name=cfg.get("experiment_name", "default"),
                stage="lightning",
                dlm_config=dlm_config,
            )
        )

    # ZClip adaptive gradient clipping (only if gradient_clipping is a dict with type=zclip)
    zclip_cfg = cfg.training.get("gradient_clipping", {})
    if isinstance(zclip_cfg, dict) and zclip_cfg.get("type", "fixed") == "zclip":
        callbacks.append(
            ZClipCallback(
                z_threshold=zclip_cfg.get("z_threshold", 3.0),
                ema_decay=zclip_cfg.get("ema_decay", 0.99),
                enabled=True,
            )
        )

    # QK clipping
    qk_cfg = cfg.training.get("qk_clip", {})
    if qk_cfg.get("enabled", False):
        callbacks.append(
            QKClipCallback(
                threshold=qk_cfg.get("threshold", 1.0),
                alpha=qk_cfg.get("alpha", 0.99),
            )
        )

    # Lambda warmup for gradual quantization
    lambda_cfg = cfg.training.get("lambda_warmup", {})
    if lambda_cfg.get("enabled", False):
        callbacks.append(
            LambdaWarmupCallback(
                warmup_steps=lambda_cfg.get("warmup_steps", 1000),
                schedule=lambda_cfg.get("schedule", "linear"),
            )
        )

    # Token counting
    if cfg.training.get("total_tokens", 0) > 0:
        callbacks.append(
            TokenCountCallback(
                max_tokens=cfg.training.total_tokens,
                seq_length=cfg.training.max_seq_length,
            )
        )

    # Meta-optimization outer loop (supersedes influence when enabled)
    meta_opt_cfg = cfg.training.get("meta_optimization", {})
    if meta_opt_cfg.get("enabled", False):
        # Build nested configs for LDC-MTL and ODM
        ldc_mtl_cfg = meta_opt_cfg.get("ldc_mtl", {})
        odm_cfg = meta_opt_cfg.get("odm", {})

        from wf_train.meta.config import LDCMTLConfig, ODMConfig

        ldc_mtl_config = LDCMTLConfig(
            enabled=ldc_mtl_cfg.get("enabled", True),
            lambda_penalty=ldc_mtl_cfg.get("lambda_penalty", 0.1),
            hidden_dim=ldc_mtl_cfg.get("hidden_dim", 32),
            router_lr=ldc_mtl_cfg.get("router_lr", 1e-3),
        )

        odm_config = ODMConfig(
            enabled=odm_cfg.get("enabled", True),
            reward_smoothing=odm_cfg.get("reward_smoothing", 0.9),
            warmup_ratio=odm_cfg.get("warmup_ratio", 0.01),
            min_weight=odm_cfg.get("min_weight", 0.05),
            max_weight=odm_cfg.get("max_weight", 0.60),
        )

        meta_config = MetaOptimizationConfig(
            enabled=True,
            ldc_mtl=ldc_mtl_config,
            odm=odm_config,
            log_interval=meta_opt_cfg.get("log_interval", 100),
        )

        callbacks.append(MetaOptimizerCallback(config=meta_config))
        logger.info(
            f"Meta-optimization enabled: "
            f"ldc_mtl={meta_config.ldc_mtl.enabled}, "
            f"odm={meta_config.odm.enabled}, "
            f"log_interval={meta_config.log_interval}"
        )

    # LR monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Progress bar - disabled for non-TTY environments to avoid potential hangs
    # callbacks.append(RichProgressBar())

    return callbacks


def create_trainer(cfg: DictConfig, callbacks: list, max_steps: int) -> pl.Trainer:
    """Create Lightning Trainer from config."""
    # Determine strategy based on number of available GPUs
    # NOTE: WORLD_SIZE isn't set yet (Lightning sets it later), so use cuda.device_count()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    logger.info(f"Detected {num_gpus} GPUs for training")

    if num_gpus > 1:
        strategy_name = cfg.distributed.get("strategy", "ddp")
        if strategy_name == "fsdp":
            # Create proper FSDPStrategy with mixed precision for bfloat16 training
            # Reference: https://lightning.ai/docs/pytorch/stable/api/lightning_fabric.strategies.FSDPStrategy.html
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

            # FSDP mixed precision config (bfloat16 compute, fp32 reduce for stability)
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,  # Reduce in bf16 to match params
                buffer_dtype=torch.bfloat16,
            )

            # Get sharding strategy from config (default to FULL_SHARD = ZeRO-3)
            fsdp_cfg = cfg.distributed.get("fsdp", {})
            sharding_strategy_str = fsdp_cfg.get("sharding_strategy", "FULL_SHARD")
            sharding_strategy = getattr(ShardingStrategy, sharding_strategy_str)

            strategy = FSDPStrategy(
                sharding_strategy=sharding_strategy,
                mixed_precision=mp_policy,
                activation_checkpointing_policy={torch.nn.TransformerEncoderLayer}
                    if fsdp_cfg.get("activation_checkpointing", {}).get("enabled", False)
                    else None,
                limit_all_gathers=fsdp_cfg.get("limit_all_gathers", True),
            )
            logger.info(f"Using FSDPStrategy with {sharding_strategy_str} sharding and bf16 mixed precision")
        else:
            strategy = strategy_name
    else:
        strategy = "auto"

    # Loggers - always CSV for local metrics, optionally WandB
    loggers = []

    # CSV logger for local metrics (easy to parse)
    csv_logger = CSVLogger(
        save_dir=cfg.output_dir,
        name="metrics",
    )
    loggers.append(csv_logger)

    # WandB logger (optional)
    if cfg.training.logging.wandb.get("enabled", False):
        # Generate unique run ID to avoid conflicts
        import uuid
        run_id = f"{cfg.get('experiment_name', 'run')}-{uuid.uuid4().hex[:8]}"
        wandb_logger = WandbLogger(
            project=cfg.training.logging.wandb.get("project", "wrinklefree"),
            name=cfg.get("experiment_name"),
            save_dir=cfg.output_dir,
            id=run_id,  # Unique ID ensures fresh run
        )
        loggers.append(wandb_logger)

    # Handle gradient_clipping as either float or dict
    grad_clip_cfg = cfg.training.get("gradient_clipping", 1.0)
    if isinstance(grad_clip_cfg, (int, float)):
        gradient_clip_val = float(grad_clip_cfg)
    elif grad_clip_cfg.get("type", "fixed") == "fixed":
        gradient_clip_val = grad_clip_cfg.get("max_norm", 1.0)
    else:
        gradient_clip_val = None  # ZClip callback handles it

    # Validation config
    val_cfg = cfg.training.get("validation", {})
    val_enabled = val_cfg.get("enabled", False)

    trainer = pl.Trainer(
        max_steps=max_steps,
        max_epochs=9999,  # Large value - training stops at max_steps anyway
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        precision="bf16-mixed",
        strategy=strategy,
        devices="auto",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=cfg.training.logging.get("log_interval", 10),
        # Validation settings (C4 perplexity every N steps)
        val_check_interval=val_cfg.get("val_check_interval", 500) if val_enabled else None,
        limit_val_batches=val_cfg.get("limit_val_batches", 50) if val_enabled else 0,
        num_sanity_val_steps=0,  # Disable sanity check to avoid potential hangs
        gradient_clip_val=gradient_clip_val,
        enable_checkpointing=True,
        default_root_dir=cfg.output_dir,
    )

    return trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    print("[DEBUG] main() started!", flush=True)
    # Get rank for distributed
    rank = int(os.environ.get("RANK", 0))
    print(f"[DEBUG] Setting up logging for rank {rank}...", flush=True)
    setup_logging(rank=rank, output_dir=Path(cfg.output_dir))

    # Enable TF32 for Ampere+ GPUs (A100, H100, RTX 30xx/40xx)
    # TF32 accelerates bf16 matmul by using reduced-precision accumulation internally.
    # This gives 10-20% speedup with negligible accuracy impact for LLM training.
    # See: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if rank == 0:
        logger.info("TF32 enabled for CUDA matmul and cuDNN")

    # Helper for GPU memory debugging - use print() for reliable output
    def log_gpu_memory(label: str) -> None:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logger.debug(f"GPU MEM {label}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Log config
    if rank == 0:
        logger.info("=" * 60)
        logger.info("WrinkleFree Lightning Training")
        logger.info("=" * 60)
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute max_steps from total_tokens if not specified
    max_steps = cfg.training.get("max_steps")
    if max_steps is None:
        total_tokens = cfg.training.get("total_tokens")
        if total_tokens:
            batch_size = cfg.training.get("batch_size", 32)
            seq_length = cfg.training.get("max_seq_length", 512)
            grad_accum = cfg.training.get("gradient_accumulation_steps", 1)
            tokens_per_step = batch_size * seq_length * grad_accum
            max_steps = int(total_tokens / tokens_per_step)
            logger.info(
                f"Computed max_steps={max_steps:,} from total_tokens={total_tokens:,} "
                f"(batch={batch_size}, seq_len={seq_length}, grad_accum={grad_accum})"
            )
        else:
            max_steps = 10000  # Default fallback
            logger.warning("Neither max_steps nor total_tokens specified, using default max_steps=10000")

    # Load model and tokenizer
    log_gpu_memory("before model load")
    model, tokenizer = load_model_and_tokenizer(cfg)
    log_gpu_memory("after model load")

    # Load teacher if needed immediately, or prepare config for lazy loading
    teacher = load_teacher_model(cfg)
    log_gpu_memory("after teacher check")
    teacher_cfg = prepare_teacher_config(cfg) if teacher is None else None

    # Add batch size reduction factor to teacher config for lazy loading
    if teacher_cfg is not None:
        # When teacher is lazy loaded, reduce batch size by this factor
        # Default 0.5 means batch size is halved (teacher ~doubles memory usage)
        teacher_cfg["batch_size_factor"] = cfg.training.get("teacher", {}).get(
            "batch_size_factor", 0.5
        )

    # Create objective manager
    objective_manager = create_objective_manager(
        config=cfg.training,
        total_steps=max_steps,
    )

    # Create data module with validation support
    val_cfg = cfg.training.get("validation", {})
    val_enabled = val_cfg.get("enabled", False)

    # Determine num_workers: from config, env var, or default
    num_workers = cfg.data.get("num_workers", None)
    if num_workers is None:
        num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "4"))

    datamodule = WrinkleFreeDataModule(
        tokenizer=tokenizer,
        batch_size=cfg.training.batch_size,
        max_length=cfg.training.max_seq_length,
        config_name=cfg.data.get("config_name", "default"),
        with_probes=False,  # Probe dataloaders only used by legacy influence tracking
        packed=cfg.training.get("packing", {}).get("enabled", True),
        num_workers=num_workers,
        # Validation (C4 perplexity)
        val_config_name=val_cfg.get("config_name") if val_enabled else None,
        val_batch_size=val_cfg.get("batch_size", 8),
    )

    # Create Lightning module
    # Handle gradient_clipping as either float or dict with max_norm
    grad_clip_cfg = cfg.training.get("gradient_clipping", 1.0)
    if isinstance(grad_clip_cfg, (int, float)):
        gradient_clipping = float(grad_clip_cfg)
    else:
        gradient_clipping = grad_clip_cfg.get("max_norm", 1.0)

    module = WrinkleFreeLightningModule(
        model=model,
        objective_manager=objective_manager,
        teacher_model=teacher.model if teacher else None,
        teacher_cfg=teacher_cfg,  # For lazy loading
        optimizer_cfg=cfg.training.get("optimizer", {}),
        scheduler_cfg=cfg.training.get("scheduler", {}),
        gradient_clipping=gradient_clipping,
        resume_cfg=cfg.training.get("resume", {}),
    )

    # Store datamodule reference for batch size adjustment during lazy teacher loading
    module._datamodule = datamodule

    # Create callbacks
    callbacks = create_callbacks(cfg)

    # Create trainer
    trainer = create_trainer(cfg, callbacks, max_steps)

    # Check for resume checkpoint (Lightning .ckpt file)
    resume_ckpt_path = find_lightning_resume_checkpoint(cfg)
    if resume_ckpt_path:
        logger.info(f"Resuming training from: {resume_ckpt_path}")

    # Auto batch size scaling using Lightning's Tuner (fast mode)
    # Uses steps_per_trial=1 for quick probing
    # NOTE: Not supported for DDP/FSDP!
    auto_batch_enabled = cfg.training.get("auto_batch_size", False)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if auto_batch_enabled and world_size > 1:
        print(f"\n{'='*60}")
        print(f"WARNING: auto_batch_size is NOT supported with DDP/FSDP!")
        print(f"  WORLD_SIZE={world_size}, using batch_size={cfg.training.get('batch_size', 32)}")
        print(f"{'='*60}\n")
        logger.warning(
            "auto_batch_size is NOT supported with DDP/FSDP distributed training. "
            "Using configured batch_size=%d instead.",
            cfg.training.get("batch_size", 32),
        )
        auto_batch_enabled = False

    if auto_batch_enabled:
        from pytorch_lightning.tuner import Tuner

        max_val = cfg.training.get("auto_batch_size_max_val", 128)
        init_val = cfg.training.get("batch_size", 32)

        print(f"\n{'='*60}")
        print(f"AUTO BATCH SIZE: Lightning Tuner (init={init_val}, max={max_val})")
        print(f"{'='*60}\n")

        tuner = Tuner(trainer)
        tuner.scale_batch_size(
            module,
            datamodule=datamodule,
            mode="binsearch",  # Binary search for optimal
            init_val=init_val,
            max_trials=10,  # Fewer trials (was 25)
            steps_per_trial=1,  # Fast: 1 step per trial (was 5)
            batch_arg_name="batch_size",
        )
        print(f"\n{'='*60}")
        print(f"AUTO BATCH SIZE: Found optimal batch size: {datamodule.batch_size}")
        print(f"{'='*60}\n")
        logger.info(f"Auto batch size: {datamodule.batch_size}")

    # Train!
    log_gpu_memory("before trainer.fit()")
    logger.info("Starting training...")
    trainer.fit(module, datamodule, ckpt_path=resume_ckpt_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
