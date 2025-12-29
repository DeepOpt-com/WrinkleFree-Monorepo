#!/usr/bin/env python3
"""Main training entry point for WrinkleFree BitNet training."""

import logging
import os
import sys
import traceback
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class CheaperTrainingSearchPathPlugin(SearchPathPlugin):
    """Add CheaperTraining configs to Hydra search path.

    This allows 1.58Quant to use data configs from CheaperTraining
    (e.g., data=mixed_pretrain) without duplicating them.
    """

    def manipulate_search_path(self, search_path):
        # Add CheaperTraining configs directory
        cheapertraining_configs = Path(__file__).parent.parent.parent / "WrinkleFree-CheaperTraining" / "configs"
        if cheapertraining_configs.exists():
            search_path.append(
                provider="cheapertraining",
                path=f"file://{cheapertraining_configs}",
            )


# Register the plugin before Hydra initializes
Plugins.instance().register(CheaperTrainingSearchPathPlugin)

from wrinklefree.data import create_finetune_dataloader, create_pretrain_dataloader, create_pretraining_dataloader
from wrinklefree.training import (
    cleanup_distributed,
    download_checkpoint_from_gcs,
    run_stage1,
    run_stage1_9,
    run_stage2,
    run_stage3,
    setup_distributed,
)
from wrinklefree.utils import (
    AuditLogger,
    CredentialsError,
    RunManager,
    RunStatus,
    create_run_manager,
    generate_fingerprint,
)
from wrinklefree.utils.gpu_utils import (
    estimate_starting_batch_size,
    log_gpu_info,
)

logger = logging.getLogger(__name__)


def update_wandb_metadata(run_manager: RunManager | None, rank: int) -> None:
    """Update run manager with wandb URL if available.

    Should be called after trainer initialization when wandb.run exists.
    """
    if run_manager is None or rank != 0:
        return

    try:
        import wandb
        if wandb.run and wandb.run.url:
            run_manager.update_status(
                RunStatus.RUNNING,
                wandb_run_id=wandb.run.id,
                wandb_url=wandb.run.url,
            )
            logger.info(f"Updated metadata with W&B URL: {wandb.run.url}")
    except Exception as e:
        logger.debug(f"Could not update wandb metadata: {e}")


def setup_logging(rank: int, output_dir: Path) -> None:
    """Setup logging configuration."""
    log_level = logging.INFO if rank == 0 else logging.WARNING

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log") if rank == 0 else logging.NullHandler(),
        ],
    )


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hub_repo_id(cfg: DictConfig) -> str | None:
    """Get HF Hub repo_id from config."""
    hub_config = None
    if hasattr(cfg, "checkpoint") and hasattr(cfg.checkpoint, "hub"):
        hub_config = cfg.checkpoint.hub
    elif hasattr(cfg, "training") and hasattr(cfg.training, "checkpoint"):
        checkpoint_cfg = cfg.training.checkpoint
        if hasattr(checkpoint_cfg, "hub"):
            hub_config = checkpoint_cfg.hub

    if hub_config and getattr(hub_config, "enabled", False):
        return getattr(hub_config, "repo_id", None)
    return None


def get_gcs_bucket(cfg: DictConfig) -> str | None:
    """Get GCS bucket name from config or environment."""
    import os

    # Check environment variable first (set by smoke_test.yaml)
    bucket = os.environ.get("GCS_BUCKET")
    if bucket:
        print(f"[DEBUG get_gcs_bucket] Using bucket from env: {bucket}")
        return bucket

    # Check top-level gcs config (preferred location in config.yaml)
    gcs_config = cfg.get("gcs", {})
    print(f"[DEBUG get_gcs_bucket] gcs_config: {gcs_config}")
    if gcs_config.get("enabled", False):
        bucket = gcs_config.get("bucket")
        print(f"[DEBUG get_gcs_bucket] Returning bucket: {bucket}")
        return bucket

    # Legacy: check checkpoint.gcs or training.checkpoint.gcs
    gcs_config = None
    if hasattr(cfg, "checkpoint") and hasattr(cfg.checkpoint, "gcs"):
        gcs_config = cfg.checkpoint.gcs
    elif hasattr(cfg, "training") and hasattr(cfg.training, "checkpoint"):
        checkpoint_cfg = cfg.training.checkpoint
        if hasattr(checkpoint_cfg, "gcs"):
            gcs_config = checkpoint_cfg.gcs

    if gcs_config and getattr(gcs_config, "enabled", False):
        return getattr(gcs_config, "bucket", None)
    return None


def get_or_download_checkpoint(
    local_path: Path,
    hub_repo_id: str | None,
    stage: str,
    cache_dir: Path,
    gcs_bucket: str | None = None,
    gcs_prefix: str = "checkpoints",
) -> Path | None:
    """
    Get checkpoint from local path or GCS.

    Priority: local > GCS

    Args:
        local_path: Local checkpoint path to check first
        hub_repo_id: Deprecated, not used (kept for backwards compatibility)
        stage: Stage name to look for
        cache_dir: Where to cache downloaded checkpoints
        gcs_bucket: GCS bucket name (e.g., "wrinklefree-checkpoints")
        gcs_prefix: Prefix path in GCS bucket (default: "checkpoints")

    Returns:
        Path to checkpoint, or None if not found
    """
    print(f"[DEBUG get_or_download] Checking for {stage} checkpoint...")
    print(f"[DEBUG get_or_download] local_path={local_path}, gcs_bucket={gcs_bucket}, gcs_prefix={gcs_prefix}")

    # Check local first
    if local_path.exists():
        print(f"[DEBUG get_or_download] ✓ Found local checkpoint: {local_path}")
        logger.info(f"Using local checkpoint: {local_path}")
        return local_path
    else:
        print(f"[DEBUG get_or_download] Local path does not exist: {local_path}")

    # Try GCS
    if gcs_bucket:
        print(f"[DEBUG get_or_download] Checking GCS: gs://{gcs_bucket}/{gcs_prefix}/{stage}/")
        logger.info(f"Checking GCS: gs://{gcs_bucket}/{gcs_prefix}/{stage}/")
        gcs_path = download_checkpoint_from_gcs(
            bucket_name=gcs_bucket,
            stage=stage,
            local_dir=cache_dir / "gcs" / stage,
            prefix=gcs_prefix,
        )
        if gcs_path:
            print(f"[DEBUG get_or_download] ✓ Downloaded from GCS: {gcs_path}")
            return gcs_path
        else:
            print(f"[DEBUG get_or_download] ✗ GCS download returned None")
    else:
        print(f"[DEBUG get_or_download] ✗ No GCS bucket configured")

    print(f"[DEBUG get_or_download] ✗ No checkpoint found for {stage}")
    return None


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main training function.

    Runs the appropriate training stage based on configuration.

    Usage:
        # Stage 1: SubLN insertion
        uv run python scripts/train.py training=stage1_subln model=llama_7b

        # Stage 2: Continue pre-training
        uv run python scripts/train.py training=stage2_pretrain model=llama_7b data=falcon

        # Stage 3: Distillation fine-tuning
        uv run python scripts/train.py training=stage3_distill model=llama_7b data=downstream

        # Skip auto-resume from GCS
        uv run python scripts/train.py training=stage2_pretrain +skip_recovery=true
    """
    # Setup
    rank, local_rank, world_size = setup_distributed()
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(rank, output_dir)
    set_seed(cfg.seed)

    # Auto-detect GPU and set batch size if not explicitly configured
    if rank == 0:
        log_gpu_info()

    # Auto batch size: estimate based on GPU VRAM if training.auto_batch_size is True (default)
    auto_batch = getattr(cfg.training, "auto_batch_size", True)
    if auto_batch and torch.cuda.is_available():
        # Get model name and stage for estimation
        model_name = cfg.model.get("name", cfg.get("model_name", "unknown"))
        stage = cfg.training.get("stage", "stage2")

        estimated_batch = estimate_starting_batch_size(model_name, stage)
        current_batch = cfg.training.get("batch_size", 64)

        if rank == 0:
            print(f"[AUTO-BATCH] model={model_name}, stage={stage}, estimated={estimated_batch}, current={current_batch}")

        # Only reduce batch size, never increase (user may have set it lower intentionally)
        if estimated_batch < current_batch:
            # Calculate gradient accumulation to maintain effective batch size
            effective_batch = current_batch * cfg.training.get("gradient_accumulation_steps", 1)
            new_grad_accum = max(1, effective_batch // estimated_batch)

            if rank == 0:
                logger.info(
                    f"Auto-batch: Reducing batch_size {current_batch} -> {estimated_batch} "
                    f"(grad_accum: {cfg.training.get('gradient_accumulation_steps', 1)} -> {new_grad_accum})"
                )

            # Update config in-place
            with open_dict(cfg):
                cfg.training.batch_size = estimated_batch
                cfg.training.gradient_accumulation_steps = new_grad_accum

    if rank == 0:
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"World size: {world_size}")

    # =========================================================================
    # FAIL LOUDLY if GCS is not enabled (checkpoints must be synced to GCS)
    # =========================================================================
    gcs_config = cfg.get("gcs", {})
    if not gcs_config.get("enabled", False):
        error_msg = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! FATAL ERROR: GCS is not enabled!
!!!
!!! All training runs MUST sync checkpoints to GCS for:
!!!   - Reproducibility and recovery
!!!   - Cross-machine checkpoint access
!!!   - Experiment tracking
!!!
!!! To fix, set gcs.enabled=true in your config or override:
!!!   uv run python scripts/train.py ... gcs.enabled=true gcs.bucket=wrinklefree-checkpoints
!!!
!!! If you really want to run without GCS (NOT recommended), use:
!!!   uv run python scripts/train.py ... +allow_no_gcs=true
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
        if not cfg.get("allow_no_gcs", False):
            logger.error(error_msg)
            raise RuntimeError("GCS must be enabled for training. Set gcs.enabled=true or +allow_no_gcs=true to override.")
        else:
            logger.warning("!!! WARNING: Running without GCS sync. Checkpoints will NOT be backed up! !!!")

    # =========================================================================
    # Run Fingerprinting & Auto-Resume System
    # =========================================================================
    audit_logger = AuditLogger(enabled=(rank == 0))
    run_manager = None
    fingerprint = None
    resume_checkpoint_path = None
    resume_wandb_id = None

    # Check if recovery should be skipped (CLI flag: +skip_recovery=true)
    skip_recovery = cfg.get("skip_recovery", False)

    if rank == 0:
        # Generate fingerprint from config + git state
        fingerprint, fp_metadata = generate_fingerprint(cfg)
        logger.info(f"Run fingerprint: {fingerprint[:16]}...")

        # Warn loudly if git is dirty
        if fp_metadata.get("git_dirty", False):
            audit_logger.log_dirty_git(
                fingerprint=fingerprint,
                git_commit=fp_metadata.get("git_commit", "unknown"),
                message="Training with uncommitted changes - results may not be reproducible",
            )

        # Initialize RunManager for GCS integration (if enabled)
        if not skip_recovery:
            gcs_config = OmegaConf.to_container(cfg.get("gcs", {}), resolve=True)
            if gcs_config.get("enabled", False):
                try:
                    run_manager = RunManager(
                        fingerprint=fingerprint,
                        gcs_bucket=gcs_config.get("bucket", "wrinklefree-checkpoints"),
                        audit_logger=audit_logger,
                        fingerprint_metadata=fp_metadata,
                        gcs_prefix=gcs_config.get("experiment_prefix", "experiments"),
                        local_cache_dir=output_dir / ".fingerprint_cache",
                        rank=rank,
                    )

                    # Check for existing run and resume
                    should_resume, ckpt_path, wandb_id = run_manager.check_and_resume()

                    if run_manager.is_completed():
                        skip_completed = cfg.get("resume", {}).get("skip_completed", True)
                        if skip_completed:
                            logger.info(f"✓ Run {fingerprint[:8]} already COMPLETED. Exiting.")
                            return

                    if should_resume and ckpt_path:
                        resume_checkpoint_path = ckpt_path
                        resume_wandb_id = wandb_id
                        logger.info(f"✓ Resuming from checkpoint: {ckpt_path}")
                        logger.info(f"  WandB run ID: {wandb_id}")

                except CredentialsError as e:
                    # FAIL LOUDLY - do not continue without GCS if it's enabled
                    logger.error(f"GCS credentials error: {e}")
                    raise
        else:
            logger.info("⚠ Skip recovery mode: GCS auto-resume disabled")

    # Get checkpoint sources (HuggingFace Hub, GCS)
    hub_repo_id = get_hub_repo_id(cfg)
    gcs_bucket = get_gcs_bucket(cfg)
    if gcs_bucket and rank == 0:
        logger.info(f"GCS checkpoint bucket: {gcs_bucket}")

    # Get training stage
    stage = cfg.training.stage

    try:
        # Update run status to RUNNING (if run_manager available)
        if run_manager and rank == 0:
            run_manager.update_status(RunStatus.RUNNING)

        if stage == "subln_insertion":
            # Stage 1: Convert model to BitNet with SubLN
            logger.info("Running Stage 1: SubLN Insertion")

            model, tokenizer = run_stage1(
                pretrained_model_name=cfg.model.teacher.pretrained,
                output_dir=output_dir / "stage1_checkpoint",
                hidden_size=cfg.model.hidden_size,
                intermediate_size=cfg.model.intermediate_size,
                exclude_layers=cfg.training.conversion.get("exclude_layers", []),
                save_format=cfg.training.get("save_format", "safetensors"),
            )

            logger.info("Stage 1 complete!")

        elif stage == "layerwise_distillation":
            # Stage 1.9: Layer-wise distillation
            logger.info("Running Stage 1.9: Layer-wise Distillation")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.teacher.pretrained,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create dataloader from CheaperTraining config
            config_name = cfg.data.get("config_name", "mixed_pretrain")
            logger.info(f"Loading data config '{config_name}' from CheaperTraining")
            train_dataloader, _, _ = create_pretraining_dataloader(
                tokenizer=tokenizer,
                batch_size=cfg.training.batch_size,
                max_length=cfg.training.max_seq_length,
                config_name=config_name,
                with_probes=False,  # No probes for stage 1.9
                seed=cfg.seed,
            )

            # Load model from stage 1 (local, Hub, or GCS)
            stage1_local_path = output_dir / "stage1_checkpoint"
            stage1_path = get_or_download_checkpoint(
                local_path=stage1_local_path,
                hub_repo_id=hub_repo_id,
                stage="stage1_checkpoint",
                cache_dir=output_dir / ".hub_cache",
                gcs_bucket=gcs_bucket,
                gcs_prefix=f"checkpoints/{cfg.experiment_name}",
            )

            if stage1_path:
                from safetensors.torch import load_file
                from wrinklefree.training.stage1 import convert_model_to_bitnet

                # Load base HuggingFace model
                logger.info(f"Loading base model from {cfg.model.teacher.pretrained}")
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model.teacher.pretrained,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )

                # Convert to BitNet (adds SubLN + BitLinear)
                logger.info("Converting to BitNet architecture...")
                model = convert_model_to_bitnet(
                    model,
                    hidden_size=cfg.model.hidden_size,
                    intermediate_size=cfg.model.intermediate_size,
                )

                # Load Stage 1 weights
                logger.info(f"Loading Stage 1 weights from {stage1_path}")
                safetensors_path = stage1_path / "model.safetensors"
                if not safetensors_path.exists():
                    # Hub downloads may have checkpoint.pt instead
                    safetensors_path = stage1_path / "checkpoint.pt"
                if safetensors_path.suffix == ".safetensors":
                    state_dict = load_file(safetensors_path)
                else:
                    ckpt = torch.load(safetensors_path, map_location="cpu")
                    state_dict = ckpt.get("model_state_dict", ckpt)
                # Cast state_dict tensors to bfloat16 before loading (handles tied weights correctly)
                state_dict = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state_dict.items()}
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded Stage 1 checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
                # Ensure model is bfloat16 after loading checkpoint
                model = model.to(torch.bfloat16)
            else:
                logger.warning("Stage 1 checkpoint not found locally or on Hub, running stage 1 first")
                model, tokenizer = run_stage1(
                    pretrained_model_name=cfg.model.teacher.pretrained,
                    output_dir=stage1_local_path,
                    hidden_size=cfg.model.hidden_size,
                    intermediate_size=cfg.model.intermediate_size,
                )

            # Run stage 1.9
            model = run_stage1_9(
                student_model=model,
                teacher_model_name=cfg.model.teacher.pretrained,
                train_dataloader=train_dataloader,
                eval_dataloader=None,
                config=cfg,
                layerwise_config=cfg.training.layerwise,
                output_dir=output_dir / "stage1_9_checkpoint",
                run_manager=run_manager,
                experiment_name=cfg.experiment_name,
            )

            # Update metadata with wandb URL (after trainer initialized wandb)
            update_wandb_metadata(run_manager, rank)

            logger.info("Stage 1.9 complete!")

        elif stage == "continue_pretrain":
            # Stage 2: Continue pre-training
            logger.info("Running Stage 2: Continue Pre-training")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.teacher.pretrained,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create dataloader from CheaperTraining config
            # Data configs live in CheaperTraining/configs/data/ - 1.58Quant doesn't manage data
            config_name = cfg.data.get("config_name", "mixed_pretrain")
            influence_enabled = cfg.training.get("influence", {}).get("enabled", False)
            logger.info(f"Loading data config '{config_name}' from CheaperTraining (probes={influence_enabled})")

            train_dataloader, mixed_dataset, probe_dataloaders = create_pretraining_dataloader(
                tokenizer=tokenizer,
                batch_size=cfg.training.batch_size,
                max_length=cfg.training.max_seq_length,
                config_name=config_name,
                with_probes=influence_enabled,
                seed=cfg.seed,
            )

            # Extract first probe loader for legacy interface (stage2 trainer expects single loader)
            probe_dataloader = None
            if probe_dataloaders:
                probe_dataloader = next(iter(probe_dataloaders.values()))
                logger.info(f"Created {len(probe_dataloaders)} probe dataloaders for influence")

            # ============================================================
            # CHECKPOINT LOADING STRATEGY:
            # 1. Check for explicit resume checkpoint FIRST
            # 2. If resuming, just create model architecture (skip stage1_9 download!)
            # 3. If not resuming, load stage1_9 weights as starting point
            # ============================================================

            # Check for resume checkpoint FIRST (before downloading anything)
            resume_checkpoint_env = os.environ.get("RESUME_CHECKPOINT")
            resume_checkpoint_cfg = cfg.get("resume", {}).get("checkpoint_path")
            resume_checkpoint = resume_checkpoint_env or resume_checkpoint_cfg
            resume_from = None

            if resume_checkpoint:
                resume_path = str(resume_checkpoint)
                print(f"[RESUME] Found resume checkpoint: {resume_path}")
                logger.info(f"Found resume checkpoint: {resume_path}")

                # Download from GCS if needed
                if resume_path.startswith("gs://"):
                    print(f"[RESUME] Downloading from GCS...")
                    import subprocess
                    local_resume_dir = output_dir / ".resume_cache"
                    local_resume_dir.mkdir(parents=True, exist_ok=True)
                    local_resume_file = local_resume_dir / "checkpoint.pt"

                    result = subprocess.run(
                        ["gcloud", "storage", "cp", resume_path, str(local_resume_file)],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0 and local_resume_file.exists():
                        resume_from = local_resume_file
                        print(f"[RESUME] ✓ Downloaded to: {resume_from}")
                        logger.info(f"Downloaded resume checkpoint to {resume_from}")
                    else:
                        print(f"[RESUME] ✗ Download failed: {result.stderr}")
                        logger.error(f"Failed to download checkpoint: {result.stderr}")
                        raise RuntimeError(f"Failed to download resume checkpoint from {resume_path}")
                else:
                    resume_from = Path(resume_path)
                    if not resume_from.exists():
                        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
                    print(f"[RESUME] Using local checkpoint: {resume_from}")

                # When resuming, just create model architecture (weights will be loaded from resume checkpoint)
                print(f"[RESUME] Creating model architecture (skipping stage1_9 download)...")
                from wrinklefree.training.stage1 import convert_model_to_bitnet

                model = AutoModelForCausalLM.from_pretrained(
                    cfg.model.teacher.pretrained,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                model = convert_model_to_bitnet(
                    model,
                    hidden_size=cfg.model.hidden_size,
                    intermediate_size=cfg.model.intermediate_size,
                )
                model = model.to(torch.bfloat16)
                print(f"[RESUME] ✓ Model architecture created, will load weights from resume checkpoint")

            else:
                # No resume checkpoint - load from stage1_9 or stage1
                print(f"[STAGE2] No resume checkpoint, loading from stage1_9...")

                # Load model from stage 1 or stage 1.9 (local, Hub, or GCS)
                cache_dir = output_dir / ".hub_cache"

                # Try stage 1.9 first (if available), then fall back to stage 1
                print(f"[DEBUG Stage 2] Looking for Stage 1.9 checkpoint...")
                print(f"[DEBUG Stage 2] gcs_bucket={gcs_bucket}, gcs_prefix=checkpoints/{cfg.experiment_name}")
                stage1_9_path = get_or_download_checkpoint(
                    local_path=output_dir / "stage1_9_checkpoint",
                    hub_repo_id=hub_repo_id,
                    stage="stage1_9_checkpoint",
                    cache_dir=cache_dir,
                    gcs_bucket=gcs_bucket,
                    gcs_prefix=f"checkpoints/{cfg.experiment_name}",
                )
                print(f"[DEBUG Stage 2] stage1_9_path={stage1_9_path}")

                if stage1_9_path:
                    # Load from stage 1.9 checkpoint
                    logger.info(f"Loading from Stage 1.9 checkpoint: {stage1_9_path}")
                    # Try multiple possible checkpoint paths
                    possible_paths = [
                        stage1_9_path / "checkpoint.pt",
                        stage1_9_path / "checkpoints" / "final" / "checkpoint.pt",
                        stage1_9_path / "checkpoints" / "latest" / "checkpoint.pt",
                    ]
                    checkpoint_file = None
                    for path in possible_paths:
                        logger.info(f"  Checking path: {path} exists={path.exists()}")
                        if path.exists():
                            checkpoint_file = path
                            logger.info(f"  Found checkpoint at: {checkpoint_file}")
                            break
                    if checkpoint_file:
                        print(f"[DEBUG Stage 2] Loading checkpoint from: {checkpoint_file}")
                        checkpoint = torch.load(checkpoint_file, map_location="cpu")
                        from wrinklefree.training.stage1 import convert_model_to_bitnet

                        model = AutoModelForCausalLM.from_pretrained(
                            cfg.model.teacher.pretrained,
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True,
                        )
                        model = convert_model_to_bitnet(
                            model,
                            hidden_size=cfg.model.hidden_size,
                            intermediate_size=cfg.model.intermediate_size,
                        )
                        state_dict = checkpoint.get("model_state_dict", checkpoint)
                        # Cast state_dict tensors to bfloat16 before loading (handles tied weights correctly)
                        state_dict = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state_dict.items()}
                        missing, unexpected = model.load_state_dict(state_dict, strict=False)
                        print(f"[DEBUG Stage 2] ✓ Loaded Stage 1.9 checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
                        logger.info(f"Loaded Stage 1.9 checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
                        # Ensure model is bfloat16 after loading checkpoint
                        model = model.to(torch.bfloat16)
                    else:
                        print(f"[DEBUG Stage 2] ✗ Stage 1.9 checkpoint dir exists but no checkpoint.pt found!")
                        logger.warning(f"Stage 1.9 checkpoint dir exists but no checkpoint.pt found!")
                        stage1_9_path = None  # Fall back to stage 1

                if not stage1_9_path:
                    print(f"[DEBUG Stage 2] Falling back to Stage 1 checkpoint...")
                    # Try stage 1
                    logger.info("Falling back to Stage 1 checkpoint...")
                    stage1_path = get_or_download_checkpoint(
                        local_path=output_dir / "stage1_checkpoint",
                        hub_repo_id=hub_repo_id,
                        stage="stage1_checkpoint",
                        cache_dir=cache_dir,
                        gcs_bucket=gcs_bucket,
                        gcs_prefix=f"checkpoints/{cfg.experiment_name}",
                    )

                    if stage1_path:
                        from safetensors.torch import load_file
                        from wrinklefree.training.stage1 import convert_model_to_bitnet

                        # Load base HuggingFace model
                        logger.info(f"Loading base model from {cfg.model.teacher.pretrained}")
                        model = AutoModelForCausalLM.from_pretrained(
                            cfg.model.teacher.pretrained,
                            torch_dtype=torch.bfloat16,
                            trust_remote_code=True,
                        )

                        # Convert to BitNet (adds SubLN + BitLinear)
                        logger.info("Converting to BitNet architecture...")
                        model = convert_model_to_bitnet(
                            model,
                            hidden_size=cfg.model.hidden_size,
                            intermediate_size=cfg.model.intermediate_size,
                        )

                        # Load Stage 1 weights
                        logger.info(f"Loading Stage 1 weights from {stage1_path}")
                        safetensors_path = stage1_path / "model.safetensors"
                        if not safetensors_path.exists():
                            safetensors_path = stage1_path / "checkpoint.pt"
                        if safetensors_path.suffix == ".safetensors":
                            state_dict = load_file(safetensors_path)
                        else:
                            ckpt = torch.load(safetensors_path, map_location="cpu")
                            state_dict = ckpt.get("model_state_dict", ckpt)
                        # Cast state_dict tensors to bfloat16 before loading (handles tied weights correctly)
                        state_dict = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state_dict.items()}
                        missing, unexpected = model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Loaded Stage 1 checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
                        # Ensure model is bfloat16 after loading checkpoint
                        model = model.to(torch.bfloat16)
                    else:
                        logger.warning("No prior checkpoint found, running stage 1 first")
                        stage1_local_path = output_dir / "stage1_checkpoint"
                        model, tokenizer = run_stage1(
                            pretrained_model_name=cfg.model.teacher.pretrained,
                            output_dir=stage1_local_path,
                            hidden_size=cfg.model.hidden_size,
                            intermediate_size=cfg.model.intermediate_size,
                        )

            # Run stage 2
            model = run_stage2(
                model=model,
                train_dataloader=train_dataloader,
                config=cfg,
                output_dir=output_dir / "stage2_checkpoint",
                probe_dataloader=probe_dataloader,  # For influence-based data selection
                run_manager=run_manager,
                experiment_name=cfg.experiment_name,
                resume_from=resume_from,
            )

            # Update metadata with wandb URL (after trainer initialized wandb)
            update_wandb_metadata(run_manager, rank)

            logger.info("Stage 2 complete!")

        elif stage == "distillation":
            # Stage 3: Distillation fine-tuning
            logger.info("Running Stage 3: Distillation Fine-tuning")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model.teacher.pretrained,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create dataloaders
            train_dataloader = create_finetune_dataloader(
                dataset_path=cfg.data.dataset.path,
                tokenizer=tokenizer,
                batch_size=cfg.training.batch_size,
                max_length=cfg.training.max_seq_length,
                dataset_name=cfg.data.dataset.get("name"),
                text_column=cfg.data.preprocessing.get("text_column", "sentence"),
                label_column=cfg.data.preprocessing.get("label_column", "label"),
                num_workers=cfg.data.dataloader.get("num_workers", 2),
                is_generation=cfg.data.preprocessing.get("is_generation", False),
            )

            eval_dataloader = None
            if hasattr(cfg.data, "eval_dataset"):
                eval_dataloader = create_finetune_dataloader(
                    dataset_path=cfg.data.eval_dataset.path,
                    tokenizer=tokenizer,
                    batch_size=cfg.training.batch_size,
                    max_length=cfg.training.max_seq_length,
                    dataset_name=cfg.data.eval_dataset.get("name"),
                    split=cfg.data.eval_dataset.get("split", "validation"),
                    text_column=cfg.data.preprocessing.get("text_column", "sentence"),
                    label_column=cfg.data.preprocessing.get("label_column", "label"),
                    is_generation=cfg.data.preprocessing.get("is_generation", False),
                    shuffle=False,
                )

            # Load model from stage 2 (local, Hub, or GCS)
            cache_dir = output_dir / ".hub_cache"

            stage2_path = get_or_download_checkpoint(
                local_path=output_dir / "stage2_checkpoint" / "checkpoints" / "final",
                hub_repo_id=hub_repo_id,
                stage="stage2_checkpoint",
                cache_dir=cache_dir,
                gcs_bucket=gcs_bucket,
                gcs_prefix=f"checkpoints/{cfg.experiment_name}",
            )

            if stage2_path:
                # Try multiple possible checkpoint paths (GCS preserves directory structure)
                possible_paths = [
                    stage2_path / "checkpoint.pt",
                    stage2_path / "checkpoints" / "final" / "checkpoint.pt",
                    stage2_path / "checkpoints" / "latest" / "checkpoint.pt",
                ]
                checkpoint_file = None
                for path in possible_paths:
                    print(f"[DEBUG Stage 3] Checking path: {path} exists={path.exists()}")
                    if path.exists():
                        checkpoint_file = path
                        break

                # Fallback: glob for any .pt file
                if checkpoint_file is None:
                    for f in stage2_path.glob("**/*.pt"):
                        print(f"[DEBUG Stage 3] Found via glob: {f}")
                        checkpoint_file = f
                        break

                if checkpoint_file and checkpoint_file.exists():
                    checkpoint = torch.load(checkpoint_file, map_location="cpu")
                    from wrinklefree.models import BitNetLlama, BitNetConfig

                    # Disable flash attention if attention distillation is enabled
                    # Flash/SDPA doesn't return attention weights
                    needs_attention_weights = cfg.distillation.gamma_attention > 0
                    use_flash = cfg.model.get("use_flash_attention", True)
                    if needs_attention_weights and use_flash:
                        print("[DEBUG Stage 3] Disabling flash attention (required for attention distillation)")
                        use_flash = False

                    model_config = BitNetConfig(
                        vocab_size=cfg.model.vocab_size,
                        hidden_size=cfg.model.hidden_size,
                        intermediate_size=cfg.model.intermediate_size,
                        num_hidden_layers=cfg.model.num_hidden_layers,
                        num_attention_heads=cfg.model.num_attention_heads,
                        num_kv_heads=cfg.model.num_kv_heads,
                        use_flash_attention=use_flash,
                    )
                    model = BitNetLlama(model_config)
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                    model.load_state_dict(state_dict, strict=False)
                    print(f"[DEBUG Stage 3] ✓ Loaded Stage 2 checkpoint from {checkpoint_file}")
                    logger.info(f"Loaded Stage 2 checkpoint from {checkpoint_file}")
                else:
                    # List what's actually in the directory for debugging
                    contents = list(stage2_path.glob("**/*"))
                    print(f"[DEBUG Stage 3] ✗ No checkpoint found! Directory contents:")
                    for c in contents[:20]:
                        print(f"  - {c}")
                    raise FileNotFoundError(f"No checkpoint.pt found in {stage2_path}. Checked: {possible_paths}")
            else:
                raise FileNotFoundError(
                    "Stage 2 checkpoint not found locally or on Hub. Run stage 2 first."
                )

            # Run stage 3
            model = run_stage3(
                student_model=model,
                teacher_model_name=cfg.model.teacher.pretrained,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                config=cfg,
                distill_config=cfg.distillation,
                output_dir=output_dir / "stage3_checkpoint",
                run_manager=run_manager,
                experiment_name=cfg.experiment_name,
            )

            # Update metadata with wandb URL (after trainer initialized wandb)
            update_wandb_metadata(run_manager, rank)

            logger.info("Stage 3 complete!")

        else:
            raise ValueError(f"Unknown training stage: {stage}")

        # Update run status to COMPLETED on success
        if run_manager and rank == 0:
            run_manager.update_status(RunStatus.COMPLETED)
            logger.info(f"✓ Run {fingerprint[:8]} marked as COMPLETED")

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        if rank == 0:
            audit_logger.log_training_interrupted(
                fingerprint=fingerprint or "unknown",
                global_step=0,  # TODO: pass actual step from training loop
                message="Training interrupted by user (Ctrl+C)",
            )
            if run_manager:
                run_manager.update_status(RunStatus.INTERRUPTED)
        raise

    except Exception as e:
        # Log failure and update status
        if rank == 0:
            audit_logger.log_training_failed(
                fingerprint=fingerprint or "unknown",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            if run_manager:
                run_manager.update_status(RunStatus.FAILED, error_message=str(e))
        raise

    finally:
        cleanup_distributed()


def main_cli():
    """CLI entry point."""
    main()


if __name__ == "__main__":
    main_cli()
