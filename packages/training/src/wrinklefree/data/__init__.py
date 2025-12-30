"""Data loading utilities for BitNet training.

This module provides data loading for WrinkleFree training.

When cheapertraining is installed, uses the unified data layer from CheaperTraining.
Otherwise, falls back to DEPRECATED local implementations with a warning.

RECOMMENDED: Use the high-level API:
    from wrinklefree.data import create_pretraining_dataloader
    train_dl, mixed, probes = create_pretraining_dataloader(tokenizer, batch_size=32)

Install cheapertraining:
    pip install -e ../WrinkleFree-CheaperTraining
"""

import warnings

# Try to import from data_handler (preferred)
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
    _CHEAPERTRAINING_AVAILABLE = True
except ImportError:
    _CHEAPERTRAINING_AVAILABLE = False

# Always import local finetune datasets (SFT-specific, not deprecated)
from wrinklefree.data.finetune_dataset import (
    FinetuneDataset,
    InstructDataset,
    create_finetune_dataloader,
)

# Import DEPRECATED legacy pretrain datasets (only if cheapertraining not available)
if not _CHEAPERTRAINING_AVAILABLE:
    from wrinklefree.data._legacy import (
        PretrainDataset,
        PackedPretrainDataset,
        StreamingPretrainDataset,
        MixedPretrainDataset,
        create_pretrain_dataloader as _legacy_create_pretrain_dataloader,
        create_mixed_dataloader as _legacy_create_mixed_dataloader,
        create_probe_dataloader as _legacy_create_probe_dataloader,
        _show_deprecation_warning,
    )

# Set up factory functions based on availability
if _CHEAPERTRAINING_AVAILABLE:
    # Use cheapertraining's unified factory with legacy API translation

    def create_pretrain_dataloader(
        dataset_path: str = None,
        tokenizer=None,
        batch_size: int = 32,
        max_length: int = 2048,
        dataset_name: str = None,
        text_column: str = "text",
        num_workers: int = 4,
        seed: int = 42,
        # New API: pass config directly
        config=None,
        **kwargs,
    ):
        """Create pre-training dataloader (delegates to cheapertraining).

        Supports both legacy API (dataset_path, ...) and new API (config).
        """
        if config is not None:
            # New API - pass directly
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
        # New API
        config=None,
        **kwargs,
    ):
        """Create mixed pre-training dataloader (delegates to cheapertraining)."""
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
        # New API
        probe_config=None,
        **kwargs,
    ):
        """Create probe dataloader (delegates to cheapertraining)."""
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

    # Import legacy classes for backward compat (but they delegate to cheapertraining)
    from wrinklefree.data._legacy import (
        PretrainDataset,
        PackedPretrainDataset,
        StreamingPretrainDataset,
        MixedPretrainDataset,
    )

else:
    # DEPRECATED: Fall back to legacy implementations with loud warning
    def create_pretrain_dataloader(*args, **kwargs):
        """DEPRECATED: Create pre-training dataloader."""
        _show_deprecation_warning()
        return _legacy_create_pretrain_dataloader(*args, **kwargs)

    def create_mixed_dataloader(*args, **kwargs):
        """DEPRECATED: Create mixed pre-training dataloader."""
        _show_deprecation_warning()
        return _legacy_create_mixed_dataloader(*args, **kwargs)

    def create_probe_dataloader(*args, **kwargs):
        """DEPRECATED: Create probe dataloader."""
        _show_deprecation_warning()
        return _legacy_create_probe_dataloader(*args, **kwargs)

    # Stub classes when cheapertraining not available
    class MixedDataset:
        """Stub - install cheapertraining for full functionality."""
        def __init__(self, *args, **kwargs):
            _show_deprecation_warning()
            raise ImportError("MixedDataset requires cheapertraining. Run: pip install -e ../WrinkleFree-CheaperTraining")

    class PackedDataset:
        """Stub - install cheapertraining for full functionality."""
        def __init__(self, *args, **kwargs):
            _show_deprecation_warning()
            raise ImportError("PackedDataset requires cheapertraining. Run: pip install -e ../WrinkleFree-CheaperTraining")

    class DatasetMixture:
        """Stub - install cheapertraining for full functionality."""
        def __init__(self, *args, **kwargs):
            _show_deprecation_warning()
            raise ImportError("DatasetMixture requires cheapertraining. Run: pip install -e ../WrinkleFree-CheaperTraining")

    class InfluenceTracker:
        """Stub - install cheapertraining for full functionality."""
        def __init__(self, *args, **kwargs):
            _show_deprecation_warning()
            raise ImportError("InfluenceTracker requires cheapertraining. Run: pip install -e ../WrinkleFree-CheaperTraining")

    def create_influence_tracker(*args, **kwargs):
        _show_deprecation_warning()
        raise ImportError("create_influence_tracker requires cheapertraining. Run: pip install -e ../WrinkleFree-CheaperTraining")

    def create_dataloader(*args, **kwargs):
        _show_deprecation_warning()
        raise ImportError("create_dataloader requires cheapertraining. Run: pip install -e ../WrinkleFree-CheaperTraining")

    def create_probe_dataloaders(*args, **kwargs):
        _show_deprecation_warning()
        raise ImportError("create_probe_dataloaders requires cheapertraining. Run: pip install -e ../WrinkleFree-CheaperTraining")

    def get_dataset_weights_from_config(*args, **kwargs):
        _show_deprecation_warning()
        raise ImportError("get_dataset_weights_from_config requires cheapertraining. Run: pip install -e ../WrinkleFree-CheaperTraining")


def is_cheapertraining_available() -> bool:
    """Check if cheapertraining is installed."""
    return _CHEAPERTRAINING_AVAILABLE


__all__ = [
    # Status check
    "is_cheapertraining_available",
    # High-level API (RECOMMENDED)
    "create_pretraining_dataloader",
    "load_data_config",
    "list_available_configs",
    # Lower-level API (from data_handler when available)
    "create_dataloader",
    "create_probe_dataloaders",
    "InfluenceTracker",
    "create_influence_tracker",
    "get_dataset_weights_from_config",
    # Dataset classes
    "MixedDataset",
    "PackedDataset",
    "DatasetMixture",
    # Legacy pretrain classes (DEPRECATED)
    "PretrainDataset",
    "StreamingPretrainDataset",
    "PackedPretrainDataset",
    "MixedPretrainDataset",
    # Legacy factory functions (delegate to cheapertraining when available)
    "create_pretrain_dataloader",
    "create_mixed_dataloader",
    "create_probe_dataloader",
    # Finetune (not deprecated)
    "FinetuneDataset",
    "InstructDataset",
    "create_finetune_dataloader",
]
