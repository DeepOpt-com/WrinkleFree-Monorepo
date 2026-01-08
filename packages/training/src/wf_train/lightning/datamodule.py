"""WrinkleFree Lightning DataModule.

Wraps the existing dataloader creation from wf_data package.
Supports both pretraining (text) and SFT (chat) data configs.
"""

import logging
from typing import Any, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# SFT data configs that require chat template formatting
SFT_DATA_CONFIGS = {"sft_nemotron"}


class WrinkleFreeDataModule(pl.LightningDataModule):
    """Lightning DataModule for WrinkleFree training.

    Wraps the existing create_pretraining_dataloader from wf_data.

    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Training batch size per device
        max_length: Maximum sequence length
        config_name: Data config name (e.g., "mixed_pretrain", "fineweb")
        with_probes: Whether to create probe dataloaders for influence
        world_size: Number of distributed processes
        rank: Current process rank
        packed: Whether to use sequence packing
        num_workers: DataLoader num_workers
        val_config_name: Optional separate config for validation
        val_batch_size: Optional different batch size for validation
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32,
        max_length: int = 2048,
        config_name: str = "default",
        with_probes: bool = False,
        world_size: int = 1,
        rank: int = 0,
        packed: bool = True,
        num_workers: int = 4,
        val_config_name: Optional[str] = None,
        val_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.config_name = config_name
        self.with_probes = with_probes
        self.world_size = world_size
        self.rank = rank
        self.packed = packed
        self.num_workers = num_workers
        self.val_config_name = val_config_name
        self.val_batch_size = val_batch_size or batch_size

        # Will be set in setup()
        self.train_dataloader_instance = None
        self.val_dataloader_instance = None
        self.mixed_dataset = None
        self.probe_dataloaders = None

        # Note: We skip save_hyperparameters() to avoid omegaconf types in
        # checkpoints, which cause PyTorch 2.6+ weights_only=True loading issues.
        # Config is already managed by Hydra.

    def setup(self, stage: Optional[str] = None):
        """Create dataloaders."""
        if stage == "fit" or stage is None:
            logger.info(
                f"Creating train dataloader: config={self.config_name}, "
                f"batch_size={self.batch_size}, max_length={self.max_length}"
            )

            # Check if this is an SFT config
            if self.config_name in SFT_DATA_CONFIGS:
                train_dl, mixed_dataset, probe_loaders = self._create_sft_dataloader()
            else:
                train_dl, mixed_dataset, probe_loaders = self._create_pretrain_dataloader()

            self.train_dataloader_instance = train_dl
            self.mixed_dataset = mixed_dataset
            self.probe_dataloaders = probe_loaders

            # Create validation dataloader if config specified
            if self.val_config_name:
                logger.info(f"Creating val dataloader: config={self.val_config_name}")
                if self.val_config_name in SFT_DATA_CONFIGS:
                    val_dl, _, _ = self._create_sft_dataloader(
                        config_name=self.val_config_name,
                        batch_size=self.val_batch_size,
                    )
                else:
                    from wf_train.data import create_pretraining_dataloader

                    val_dl, _, _ = create_pretraining_dataloader(
                        tokenizer=self.tokenizer,
                        batch_size=self.val_batch_size,
                        max_length=self.max_length,
                        config_name=self.val_config_name,
                        with_probes=False,
                        world_size=self.world_size,
                        rank=self.rank,
                        packed=self.packed,
                    )
                self.val_dataloader_instance = val_dl

    def _create_pretrain_dataloader(
        self,
        config_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> tuple[DataLoader, Any, Optional[dict]]:
        """Create pretraining dataloader for text data."""
        from wf_train.data import create_pretraining_dataloader

        return create_pretraining_dataloader(
            tokenizer=self.tokenizer,
            batch_size=batch_size or self.batch_size,
            max_length=self.max_length,
            config_name=config_name or self.config_name,
            with_probes=self.with_probes,
            world_size=self.world_size,
            rank=self.rank,
            packed=self.packed,
        )

    def _create_sft_dataloader(
        self,
        config_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> tuple[DataLoader, None, None]:
        """Create SFT dataloader with chat template formatting.

        Uses nvidia/Llama-Nemotron-Post-Training-Dataset with Qwen chat template.
        Labels are masked so only assistant responses contribute to loss.
        """
        from wf_data.data import load_data_config, create_sft_dataloader, SFTConfig

        config_name = config_name or self.config_name
        config = load_data_config(config_name)
        sft_config = config.get("sft", {})

        logger.info(
            f"Creating SFT dataloader: path={sft_config.get('path')}, "
            f"subset={sft_config.get('subset')}"
        )

        sft_cfg = SFTConfig(
            path=sft_config.get("path", "nvidia/Llama-Nemotron-Post-Training-Dataset"),
            subset=sft_config.get("subset", "SFT"),
            split=sft_config.get("split", "train"),
            input_column=sft_config.get("input_column", "input"),
            output_column=sft_config.get("output_column", "output"),
            system_prompt_column=sft_config.get("system_prompt_column", "system_prompt"),
            max_length=self.max_length,
        )

        dataloader = create_sft_dataloader(
            tokenizer=self.tokenizer,
            batch_size=batch_size or self.batch_size,
            config=sft_cfg,
            max_length=self.max_length,
            rank=self.rank,
            world_size=self.world_size,
            num_workers=self.num_workers,
            packed=sft_config.get("packed", True),
        )

        # SFT doesn't support mixed dataset or probes
        return dataloader, None, None

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataloader_instance is None:
            raise RuntimeError("Call setup() before accessing train_dataloader")
        return self.train_dataloader_instance

    def val_dataloader(self):
        """Return validation dataloader if configured, empty list otherwise."""
        # Return empty list (not None) to indicate no validation
        # None causes issues with BatchSizeFinder
        if self.val_dataloader_instance is None:
            return []
        return self.val_dataloader_instance

    def get_mixed_dataset(self):
        """Get the MixedDataset for influence-aware training."""
        return self.mixed_dataset

    def get_probe_dataloaders(self):
        """Get probe dataloaders for influence computation."""
        return self.probe_dataloaders

    def update_batch_size(self, new_batch_size: int):
        """Update batch size (called by BatchSizeFinder).

        Note: This requires recreating the dataloader.
        """
        if new_batch_size != self.batch_size:
            logger.info(f"Updating batch size: {self.batch_size} -> {new_batch_size}")
            self.batch_size = new_batch_size
            # Force recreation of dataloaders
            self.train_dataloader_instance = None
            self.setup(stage="fit")

    def update_data_config(self, new_config_name: str):
        """Update data config for curriculum-based data switching.

        Called when curriculum phase changes to a different data_config.
        This recreates the training dataloader with the new config.

        Supports switching between pretraining and SFT configs.

        Args:
            new_config_name: New data config name (e.g., "fineweb", "sft_nemotron")
        """
        if new_config_name != self.config_name:
            old_is_sft = self.config_name in SFT_DATA_CONFIGS
            new_is_sft = new_config_name in SFT_DATA_CONFIGS

            mode_change = ""
            if old_is_sft and not new_is_sft:
                mode_change = " (SFT -> Pretrain)"
            elif not old_is_sft and new_is_sft:
                mode_change = " (Pretrain -> SFT)"

            logger.info(
                f"[CURRICULUM] Switching data config: {self.config_name} -> {new_config_name}{mode_change}"
            )
            self.config_name = new_config_name
            # Force recreation of training dataloader with new config
            self.train_dataloader_instance = None
            self.setup(stage="fit")
