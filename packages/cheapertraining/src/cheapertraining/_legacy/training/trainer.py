"""Main trainer orchestration.

Handles multi-stage training pipeline, checkpointing, and distributed setup.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cheapertraining._legacy.models import MobileLLM, MobileLLMConfig
from cheapertraining.training.optimizer import create_optimizer
from cheapertraining._legacy.training.scheduler import create_scheduler
from cheapertraining._legacy.training.stages.base import StageConfig, TrainingStage


@dataclass
class InfluenceTrainerConfig:
    """Configuration for influence functions in trainer."""

    # Enable influence-based training (default: True)
    enabled: bool = True

    # Phase II: Mixture weight optimization
    enable_mixture_optimization: bool = True
    weight_update_interval: int = 10000

    # Phase III: Self-boosting filter
    enable_self_boosting: bool = True
    influence_threshold: float = 0.0
    recompute_interval: int = 1000

    # Probe set settings
    probe_set_size: int = 10000
    probe_batch_size: int = 32

    # DataInf settings
    lambda_reg: float = 1e-4


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    output_dir: str = "./outputs"
    experiment_name: str = "cheapertraining"
    seed: int = 42

    # Logging
    log_interval: int = 100
    wandb_enabled: bool = True
    wandb_project: str = "cheapertraining"

    # Checkpointing
    checkpoint_interval: int = 1000
    keep_last_n: int = 3
    resume_from: Optional[str] = None

    # Influence functions (enabled by default)
    influence: InfluenceTrainerConfig = field(default_factory=InfluenceTrainerConfig)


class Trainer:
    """Main trainer class for orchestrating training stages.

    Handles:
    - Model initialization
    - Distributed training setup
    - Multi-stage training pipeline
    - Checkpointing and resumption
    - Logging (WandB, console)
    - Influence-based data optimization (enabled by default)
    """

    def __init__(
        self,
        model_config: Union[MobileLLMConfig, DictConfig, dict],
        trainer_config: Union[TrainerConfig, DictConfig, dict],
        device: Optional[torch.device] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize trainer.

        Args:
            model_config: Model configuration
            trainer_config: Trainer configuration
            device: Device to train on (auto-detected if None)
            tokenizer: Tokenizer for influence probe set creation
        """
        # Handle config types
        if isinstance(model_config, (DictConfig, dict)):
            self.model_config = MobileLLMConfig(**dict(model_config))
        else:
            self.model_config = model_config

        if isinstance(trainer_config, (DictConfig, dict)):
            trainer_dict = dict(trainer_config)
            # Handle nested influence config
            if "influence" in trainer_dict and isinstance(trainer_dict["influence"], (DictConfig, dict)):
                trainer_dict["influence"] = InfluenceTrainerConfig(**dict(trainer_dict["influence"]))
            self.trainer_config = TrainerConfig(**trainer_dict)
        else:
            self.trainer_config = trainer_config

        self.tokenizer = tokenizer

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Distributed training info
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Initialize model
        self.model = self._init_model()

        # Setup output directory
        self.output_dir = Path(self.trainer_config.output_dir) / self.trainer_config.experiment_name
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_stage: Optional[TrainingStage] = None
        self.completed_stages: list[str] = []

        # Influence function components (initialized lazily)
        self._influence_calculator: Optional[Any] = None
        self._self_boosting_filter: Optional[Any] = None
        self._mixture_calculator: Optional[Any] = None
        self._probe_dataloader: Optional[DataLoader] = None
        self._influence_initialized: bool = False

    def _init_model(self) -> MobileLLM:
        """Initialize model and move to device."""
        model = MobileLLM(self.model_config)

        if self.rank == 0:
            num_params = model.num_parameters()
            print(f"Model initialized with {num_params:,} parameters ({num_params/1e6:.1f}M)")

        model = model.to(self.device)
        return model

    def setup_influence(
        self,
        probe_dataloader: Optional[DataLoader] = None,
        mixed_dataset: Optional[Any] = None,
    ):
        """Initialize influence function components.

        Called automatically by setup_stage if influence is enabled.
        Can also be called manually for custom probe sets.

        Args:
            probe_dataloader: DataLoader for probe set (auto-created if None)
            mixed_dataset: MixedDataset for weight updates (optional)
        """
        influence_config = self.trainer_config.influence
        if not influence_config.enabled:
            return

        if self._influence_initialized:
            return

        from cheapertraining.influence import (
            InfluenceConfig,
            SelfBoostingConfig,
            create_influence_calculator,
            create_self_boosting_filter,
            create_mixture_calculator,
        )

        if self.rank == 0:
            print("\n" + "=" * 50)
            print("Initializing Influence Functions (MobileLLM-R1)")
            print("=" * 50)

        # Create influence config
        inf_config = InfluenceConfig(
            lambda_reg=influence_config.lambda_reg,
            batch_size=influence_config.probe_batch_size,
        )

        # Use provided probe dataloader or create one
        if probe_dataloader is not None:
            self._probe_dataloader = probe_dataloader
        else:
            # Create a simple probe set from the training data
            self._probe_dataloader = self._create_default_probe_dataloader()

        if self._probe_dataloader is None:
            if self.rank == 0:
                print("Warning: No probe dataloader available. Influence functions disabled.")
            return

        # Initialize influence calculator
        self._influence_calculator = create_influence_calculator(self.model, inf_config)

        if self.rank == 0:
            print("Caching probe set gradients...")
        self._influence_calculator.cache_probe_gradients(
            self._probe_dataloader,
            show_progress=(self.rank == 0),
        )

        # Initialize self-boosting filter if enabled
        if influence_config.enable_self_boosting:
            sb_config = SelfBoostingConfig(
                influence_threshold=influence_config.influence_threshold,
                recompute_interval=influence_config.recompute_interval,
            )
            self._self_boosting_filter = create_self_boosting_filter(
                self.model,
                self._probe_dataloader,
                sb_config,
            )
            self._self_boosting_filter._probe_cached = True  # Already cached above
            self._self_boosting_filter.influence_calculator = self._influence_calculator

            if self.rank == 0:
                print(f"Self-boosting filter enabled (threshold={influence_config.influence_threshold})")

        # Initialize mixture calculator if enabled
        if influence_config.enable_mixture_optimization and mixed_dataset is not None:
            self._mixture_calculator = create_mixture_calculator(
                self.model,
                self._probe_dataloader,
            )
            self._mixture_calculator._probe_cached = True
            self._mixture_calculator.influence_calculator = self._influence_calculator

            if self.rank == 0:
                print(f"Mixture optimization enabled (update interval={influence_config.weight_update_interval})")

        self._influence_initialized = True

        if self.rank == 0:
            print("=" * 50 + "\n")

    def _create_default_probe_dataloader(self) -> Optional[DataLoader]:
        """Create a default probe dataloader from training data.

        Returns None if tokenizer is not available.
        """
        if self.tokenizer is None:
            return None

        from cheapertraining.influence.config import ProbeSetConfig
        from cheapertraining._legacy.influence.probe_set import ProbeDataset

        # Create a simple probe set with synthetic quality scores
        # In production, you'd use actual data sources
        influence_config = self.trainer_config.influence

        if self.rank == 0:
            print(f"Creating default probe set (size={influence_config.probe_set_size})...")

        # For now, return None - probe set should be provided by user
        # or created from the training dataloader
        return None

    def set_probe_dataloader(self, probe_dataloader: DataLoader):
        """Set the probe dataloader for influence calculations.

        Args:
            probe_dataloader: DataLoader containing probe set samples
        """
        self._probe_dataloader = probe_dataloader
        if self._influence_initialized:
            # Re-cache gradients with new probe set
            if self._influence_calculator is not None:
                self._influence_calculator.cache_probe_gradients(
                    probe_dataloader,
                    show_progress=(self.rank == 0),
                )

    def setup_stage(
        self,
        stage_config: Union[StageConfig, DictConfig, dict],
        dataloader: DataLoader,
        optimizer_config: Optional[dict] = None,
        probe_dataloader: Optional[DataLoader] = None,
        mixed_dataset: Optional[Any] = None,
    ) -> TrainingStage:
        """Set up a training stage.

        Automatically initializes influence functions if enabled in config.

        Args:
            stage_config: Stage configuration
            dataloader: DataLoader for this stage
            optimizer_config: Optional optimizer configuration override
            probe_dataloader: Optional probe set for influence (auto-sampled if None)
            mixed_dataset: Optional MixedDataset for dynamic weight updates

        Returns:
            Configured TrainingStage
        """
        from cheapertraining._legacy.training.stages.pretrain import PretrainStage
        from cheapertraining._legacy.training.stages.midtrain import MidtrainStage
        from cheapertraining._legacy.training.stages.posttrain import PosttrainSFTStage

        # Convert config if needed
        if isinstance(stage_config, (DictConfig, dict)):
            stage_config = StageConfig(**dict(stage_config))

        # Create optimizer
        optimizer = create_optimizer(
            self.model,
            learning_rate=stage_config.learning_rate,
            weight_decay=stage_config.weight_decay,
            **(optimizer_config or {}),
        )

        # Calculate total steps
        if stage_config.num_steps > 0:
            total_steps = stage_config.num_steps
        else:
            # Epoch-based: calculate from dataloader
            total_steps = len(dataloader) * stage_config.num_epochs // stage_config.gradient_accumulation_steps

        # Create scheduler
        scheduler = create_scheduler(
            optimizer,
            scheduler_type=stage_config.scheduler_type,
            warmup_steps=stage_config.warmup_steps,
            warmup_ratio=stage_config.warmup_ratio,
            total_steps=total_steps,
            min_lr_ratio=stage_config.lr_decay_ratio,
        )

        # Auto-create probe dataloader from training data if not provided
        if probe_dataloader is None and self.trainer_config.influence.enabled:
            probe_dataloader = self._sample_probe_from_dataloader(dataloader)

        # Initialize influence functions if enabled
        if self.trainer_config.influence.enabled and probe_dataloader is not None:
            self.setup_influence(probe_dataloader, mixed_dataset)

        # Get influence filter if available
        influence_filter = self._self_boosting_filter if self.trainer_config.influence.enable_self_boosting else None

        # Select stage class based on name
        stage_name = stage_config.name.lower()
        if "pretrain" in stage_name:
            StageClass = PretrainStage
        elif "midtrain" in stage_name:
            StageClass = MidtrainStage
        elif "posttrain" in stage_name or "sft" in stage_name:
            StageClass = PosttrainSFTStage
        else:
            raise ValueError(f"Unknown stage type: {stage_name}")

        stage = StageClass(
            config=stage_config,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=dataloader,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            influence_filter=influence_filter,
        )

        # Enable gradient checkpointing if configured
        if stage_config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(mode=stage_config.gradient_checkpointing_mode)
            if self.rank == 0:
                print(f"Gradient checkpointing enabled (mode={stage_config.gradient_checkpointing_mode})")
        else:
            self.model.gradient_checkpointing_disable()

        self.current_stage = stage
        return stage

    def _sample_probe_from_dataloader(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
    ) -> Optional[DataLoader]:
        """Sample a probe set from the training dataloader.

        Args:
            dataloader: Training dataloader to sample from
            num_samples: Number of samples (default: from config)

        Returns:
            DataLoader for probe set, or None if sampling fails
        """
        num_samples = num_samples or self.trainer_config.influence.probe_set_size
        batch_size = self.trainer_config.influence.probe_batch_size

        if self.rank == 0:
            print(f"Sampling {num_samples} probe samples from training data...")

        try:
            samples = []
            for batch in dataloader:
                # Collect samples until we have enough
                batch_size_actual = batch["input_ids"].size(0)
                for i in range(batch_size_actual):
                    sample = {k: v[i] if hasattr(v, '__getitem__') else v for k, v in batch.items()}
                    samples.append(sample)
                    if len(samples) >= num_samples:
                        break
                if len(samples) >= num_samples:
                    break

            if len(samples) < num_samples:
                if self.rank == 0:
                    print(f"Warning: Only collected {len(samples)} samples (requested {num_samples})")

            if not samples:
                return None

            # Create a simple dataset from collected samples
            from torch.utils.data import Dataset

            class ProbeDatasetSimple(Dataset):
                def __init__(self, samples):
                    self.samples = samples

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return self.samples[idx]

            probe_dataset = ProbeDatasetSimple(samples)
            return DataLoader(
                probe_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

        except Exception as e:
            if self.rank == 0:
                print(f"Warning: Failed to sample probe set: {e}")
            return None

    def save_checkpoint(self, path: Optional[str] = None, stage: Optional[TrainingStage] = None):
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint (auto-generated if None)
            stage: Training stage to save (uses current_stage if None)
        """
        if self.rank != 0:
            return

        stage = stage or self.current_stage
        if stage is None:
            return

        if path is None:
            path = self.output_dir / f"checkpoint_step{stage.global_step}.pt"

        checkpoint = {
            "model_config": self.model_config,
            "trainer_config": self.trainer_config,
            "stage_state": stage.state_dict(),
            "completed_stages": self.completed_stages,
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        if "stage_state" in checkpoint:
            self.model.load_state_dict(checkpoint["stage_state"]["model_state_dict"])

        self.completed_stages = checkpoint.get("completed_stages", [])

        if self.rank == 0:
            print(f"Loaded checkpoint from {path}")

        return checkpoint

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            self.output_dir.glob("checkpoint_step*.pt"),
            key=lambda p: int(p.stem.split("step")[1]),
        )

        while len(checkpoints) > self.trainer_config.keep_last_n:
            old_ckpt = checkpoints.pop(0)
            old_ckpt.unlink()

    @classmethod
    def from_config(cls, config: Union[DictConfig, dict, str], tokenizer: Optional[Any] = None) -> "Trainer":
        """Create trainer from Hydra config.

        Args:
            config: Hydra config, dict, or path to config
            tokenizer: Optional tokenizer for influence functions

        Returns:
            Configured Trainer
        """
        if isinstance(config, str):
            config = OmegaConf.load(config)
        elif isinstance(config, dict):
            config = OmegaConf.create(config)

        model_config = config.get("model", {})

        # Build influence config from Hydra config
        influence_cfg = config.get("influence", {})
        influence_config = {
            "enabled": influence_cfg.get("enabled", True),
            "enable_mixture_optimization": influence_cfg.get("enable_mixture_optimization", True),
            "enable_self_boosting": influence_cfg.get("enable_self_boosting", True),
            "weight_update_interval": influence_cfg.get("weight_update_interval", 10000),
            "influence_threshold": influence_cfg.get("influence_threshold", 0.0),
            "recompute_interval": influence_cfg.get("recompute_interval", 1000),
            "probe_set_size": influence_cfg.get("probe_set_size", 10000),
            "probe_batch_size": influence_cfg.get("probe_batch_size", 32),
            "lambda_reg": influence_cfg.get("lambda_reg", 1e-4),
        }

        trainer_config = {
            "output_dir": config.get("output_dir", "./outputs"),
            "experiment_name": config.get("experiment_name", "cheapertraining"),
            "seed": config.get("seed", 42),
            "log_interval": config.get("logging", {}).get("log_interval", 100),
            "wandb_enabled": config.get("logging", {}).get("wandb", {}).get("enabled", True),
            "wandb_project": config.get("logging", {}).get("wandb", {}).get("project", "cheapertraining"),
            "checkpoint_interval": config.get("checkpoint", {}).get("save_interval", 1000),
            "keep_last_n": config.get("checkpoint", {}).get("keep_last_n", 3),
            "resume_from": config.get("checkpoint", {}).get("resume_from"),
            "influence": influence_config,
        }

        return cls(model_config=model_config, trainer_config=trainer_config, tokenizer=tokenizer)
