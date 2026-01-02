"""Data loading utilities for BitNet training.

This module provides data loading for WrinkleFree training.
Requires the data_handler package to be installed.

RECOMMENDED: Use the high-level API:
    from wrinklefree.data import create_pretraining_dataloader
    train_dl, mixed, probes = create_pretraining_dataloader(tokenizer, batch_size=32)
"""

# Import from data_handler (required)
try:
    from data_handler.data import (
        # High-level API (RECOMMENDED)
        create_pretraining_dataloader,
        load_data_config,
        list_available_configs,
        # Lower-level API
        create_dataloader,
        create_probe_dataloaders,
        get_dataset_weights_from_config,
        MixedDataset,
        PackedDataset,
        DatasetMixture,
        TokenPacker,
        batch_tokenize,
        compute_position_ids,
    )
    from data_handler.influence import (
        InfluenceTracker,
        create_influence_tracker,
    )
except ImportError as e:
    raise ImportError(
        "data_handler package is required. Install it with: "
        "uv add data_handler (from the monorepo)"
    ) from e

# Import local finetune datasets (SFT-specific)
from wrinklefree.data.finetune_dataset import (
    FinetuneDataset,
    InstructDataset,
    create_finetune_dataloader,
)


def create_pretrain_dataloader(
    dataset_path: str = None,
    tokenizer=None,
    batch_size: int = 32,
    max_length: int = 2048,
    dataset_name: str = None,
    text_column: str = "text",
    num_workers: int = 4,
    seed: int = 42,
    config=None,
    **kwargs,
):
    """Create pre-training dataloader (delegates to data_handler).

    Supports both legacy API (dataset_path, ...) and new API (config).
    """
    if config is not None:
        dataloader, _ = create_dataloader(
            config=config,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            seed=seed,
            **kwargs,
        )
        return dataloader

    # Legacy API - construct config
    config = {
        "dataset": {
            "path": dataset_path,
            "name": dataset_name,
            "split": kwargs.pop("split", "train"),
        },
        "preprocessing": {
            "text_column": text_column,
        },
    }
    dataloader, _ = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        seed=seed,
        **kwargs,
    )
    return dataloader


def create_mixed_dataloader(
    sources=None,
    tokenizer=None,
    batch_size: int = 32,
    max_length: int = 2048,
    num_workers: int = 4,
    seed: int = 42,
    config=None,
    **kwargs,
):
    """Create mixed pre-training dataloader (delegates to data_handler)."""
    if config is not None:
        dataloader, mixed_dataset = create_dataloader(
            config=config,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            seed=seed,
            **kwargs,
        )
        return dataloader, mixed_dataset

    # Legacy API - construct config from sources
    config = {"sources": sources or []}
    dataloader, mixed_dataset = create_dataloader(
        config=config,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        seed=seed,
        **kwargs,
    )
    return dataloader, mixed_dataset


def create_probe_dataloader(
    path: str = None,
    tokenizer=None,
    subset: str = None,
    split: str = "train",
    size: int = 1000,
    max_length: int = 2048,
    text_column: str = "text",
    probe_config=None,
    **kwargs,
):
    """Create probe dataloader (delegates to data_handler)."""
    if probe_config is not None:
        loaders = create_probe_dataloaders(
            probe_config=probe_config,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    else:
        # Legacy API - construct probe config
        probe_config = {
            "path": path,
            "subset": subset,
            "split": split,
            "size": size,
            "text_column": text_column,
        }
        loaders = create_probe_dataloaders(
            probe_config=probe_config,
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    # Return first loader for backward compat (single probe mode)
    return next(iter(loaders.values())) if loaders else None


__all__ = [
    # High-level API (RECOMMENDED)
    "create_pretraining_dataloader",
    "load_data_config",
    "list_available_configs",
    # Lower-level API (from data_handler)
    "create_dataloader",
    "create_probe_dataloaders",
    "InfluenceTracker",
    "create_influence_tracker",
    "get_dataset_weights_from_config",
    # Dataset classes
    "MixedDataset",
    "PackedDataset",
    "DatasetMixture",
    "TokenPacker",
    "batch_tokenize",
    "compute_position_ids",
    # Legacy factory functions (delegate to data_handler)
    "create_pretrain_dataloader",
    "create_mixed_dataloader",
    "create_probe_dataloader",
    # Finetune
    "FinetuneDataset",
    "InstructDataset",
    "create_finetune_dataloader",
]
