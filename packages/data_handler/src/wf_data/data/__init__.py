"""Data loading - active components.

This module provides the unified data loading interface for WrinkleFree.

Primary entry points:
- create_pretraining_dataloader: High-level convenience function (RECOMMENDED)
- create_dataloader: Unified factory for training dataloaders
- create_probe_dataloaders: Factory for influence probe dataloaders
- load_data_config: Load YAML configs from CheaperTraining

Consumer repos (Fairy2, 1.58Quant) should use create_pretraining_dataloader()
and NOT maintain their own data YAML configs.

Legacy components (TokenizerWrapper, datasets) have been moved to
cheapertraining._legacy.data
"""

from typing import Any

from torch.utils.data import DataLoader

from wf_data.data.mixing import (
    DatasetMixture,
    DomainProbeConfig,
    DomainProbeDataset,
    MixedDataset,
    PackedDataset,
    create_domain_probe_loaders,
    create_mixed_dataloader,
    create_mixed_dataset,
)

from wf_data.data.factory import (
    create_dataloader,
    create_probe_dataloaders,
    get_dataset_weights_from_config,
)

from wf_data.data.packing import (
    TokenPacker,
    batch_tokenize,
    compute_position_ids,
    pack_token_buffer,
)

from wf_data.data.config_loader import (
    load_data_config,
    list_available_configs,
    get_config_path,
)


def create_pretraining_dataloader(
    tokenizer: Any,
    batch_size: int,
    max_length: int = 2048,
    config_name: str = "mixed_pretrain",
    with_probes: bool = True,
    **kwargs,
) -> tuple[DataLoader, MixedDataset | None, dict[str, DataLoader] | None]:
    """High-level convenience function for pretraining with named config.

    This is the RECOMMENDED way for consumer repos (Fairy2, 1.58Quant) to
    create dataloaders. The YAML config lives in CheaperTraining - consumers
    don't need their own data configs.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        batch_size: Batch size for training.
        max_length: Maximum sequence length (default: 2048).
        config_name: Name of YAML file in configs/data/ (default: "mixed_pretrain").
                     Use list_available_configs() to see options.
        with_probes: Whether to create probe dataloaders for influence (default: True).
        **kwargs: Additional args passed to create_dataloader (seed, rank, world_size, etc.)

    Returns:
        tuple: (train_dataloader, mixed_dataset, probe_dataloaders)
            - train_dataloader: DataLoader for training
            - mixed_dataset: MixedDataset for influence weight updates (or None)
            - probe_dataloaders: dict of probe DataLoaders for influence (or None)

    Example:
        from wf_data.data import create_pretraining_dataloader

        train_dl, mixed_dataset, probes = create_pretraining_dataloader(
            tokenizer,
            batch_size=32,
            max_length=2048,
            seed=42,
        )
    """
    config = load_data_config(config_name)

    train_dl, mixed = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        **kwargs,
    )

    probes = None
    if with_probes and "probe" in config:
        probes = create_probe_dataloaders(
            probe_config=config["probe"],
            tokenizer=tokenizer,
            max_length=max_length,
        )

    return train_dl, mixed, probes

__all__ = [
    # High-level API (RECOMMENDED for consumer repos)
    "create_pretraining_dataloader",
    "load_data_config",
    "list_available_configs",
    "get_config_path",
    # Lower-level API
    "create_dataloader",
    "create_probe_dataloaders",
    "get_dataset_weights_from_config",
    # Dataset classes
    "DatasetMixture",
    "MixedDataset",
    "PackedDataset",
    "DomainProbeConfig",
    "DomainProbeDataset",
    # Packing utilities
    "TokenPacker",
    "batch_tokenize",
    "compute_position_ids",
    "pack_token_buffer",
    # Legacy API (still supported)
    "create_mixed_dataloader",
    "create_mixed_dataset",
    "create_domain_probe_loaders",
]
