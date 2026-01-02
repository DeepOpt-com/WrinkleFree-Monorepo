"""Unified data loading factory for WrinkleFree projects.

This module provides the single entry point for all data loading across:
- WrinkleFree-1.58Quant (pre-training, distillation)
- WrinkleFree-CheaperTraining (influence-based training)
- WrinkleFree-DLM-Converter (can import utilities)

Usage:
    from data_handler.data import create_dataloader, create_probe_dataloaders

    # Create training dataloader
    dataloader, mixed_dataset = create_dataloader(
        config=cfg.data,
        tokenizer=tokenizer,
        batch_size=32,
        max_length=2048,
    )

    # Create probe dataloaders for influence calculation
    probe_loaders = create_probe_dataloaders(
        probe_config=cfg.data.probe,
        tokenizer=tokenizer,
        max_length=2048,
    )
"""

import os
from typing import Any, Optional

# Prevent HuggingFace datasets from hanging on multi-core systems
# Reference: https://github.com/huggingface/datasets/issues/6079
if "KMP_AFFINITY" not in os.environ:
    os.environ["KMP_AFFINITY"] = "disabled"

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset


def _get_optimal_num_workers(num_workers: int | None, default: int = 4) -> int:
    """Get optimal num_workers for data loading.

    Args:
        num_workers: Explicitly specified num_workers (takes priority)
        default: Default from config (default: 4, recommended for streaming datasets)

    Returns:
        Number of workers to use for data loading.

    Environment variable override:
        DATA_NUM_WORKERS=0  # Forces num_workers=0 for debugging

    Best practice: 4-8 workers for streaming datasets balances throughput
    and HuggingFace API rate limits.
    Reference: PyTorch forums and performance tuning guides.
    """
    # Environment variable override for debugging
    env_workers = os.environ.get("DATA_NUM_WORKERS")
    if env_workers is not None:
        return int(env_workers)

    if num_workers is not None:
        return num_workers
    return default

from data_handler.data.mixing import (
    DatasetMixture,
    DomainProbeDataset,
    DomainProbeConfig,
    MixedDataset,
    PackedDataset,
    _worker_init_fn,
)


def create_dataloader(
    config: DictConfig | dict,
    tokenizer: Any,
    batch_size: int,
    max_length: int = 2048,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int | None = None,  # Auto-detect: os.cpu_count() - 1, capped at 16
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    packed: bool = True,
) -> tuple[DataLoader, MixedDataset | None]:
    """Create dataloader from configuration.

    This is the unified entry point for all data loading in WrinkleFree.
    Supports two configuration modes:

    1. Multi-source mode (for influence-based training):
       ```yaml
       sources:
         - name: dclm
           path: mlfoundations/dclm-baseline-1.0-parquet
           weight: 0.25
         - name: fineweb_edu
           path: HuggingFaceFW/fineweb-edu
           weight: 0.30
       ```

    2. Single-source mode (backward compatibility):
       ```yaml
       dataset:
         path: HuggingFaceFW/fineweb-edu
         name: sample-10BT
       ```

    Args:
        config: Data configuration (DictConfig or dict)
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        rank: Process rank for distributed training
        world_size: Total number of processes
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker
        pin_memory: Whether to pin memory for GPU transfer
        seed: Random seed for reproducibility
        packed: Whether to use sequence packing

    Returns:
        Tuple of (DataLoader, MixedDataset or None)
        The MixedDataset is returned for influence tracking (None for single-source)
    """
    # Convert to dict if DictConfig
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Get num_workers: explicit param > config.dataloader.num_workers > config.num_workers > default (4)
    dataloader_config = config.get("dataloader", {})
    config_num_workers = dataloader_config.get("num_workers", config.get("num_workers", 4))
    num_workers = _get_optimal_num_workers(num_workers, default=config_num_workers)

    # Check for multi-source mode
    sources = config.get("sources")
    if sources:
        return _create_multi_source_dataloader(
            sources=sources,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            seed=seed,
            packed=packed,
            shuffle_buffer_size=config.get("shuffle_buffer_size", 1000),
        )

    # Single-source mode
    dataset_config = config.get("dataset", {})
    if dataset_config:
        return _create_single_source_dataloader(
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            seed=seed,
            packed=packed,
            text_column=config.get("preprocessing", {}).get("text_column", "text"),
        )

    raise ValueError(
        "Data config must have either 'sources' (multi-source) or 'dataset' (single-source)"
    )


def _create_multi_source_dataloader(
    sources: list[dict],
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    rank: int,
    world_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    seed: int,
    packed: bool,
    shuffle_buffer_size: int,
) -> tuple[DataLoader, MixedDataset]:
    """Create dataloader for multi-source mixed dataset."""
    # Convert source dicts to DatasetMixture objects
    mixtures = []
    for source in sources:
        mixtures.append(
            DatasetMixture(
                name=source.get("name", source["path"]),
                weight=source.get("weight", 1.0),
                path=source["path"],
                subset=source.get("subset"),
                split=source.get("split", "train"),
                text_column=source.get("text_column", "text"),
            )
        )

    # Create mixed dataset
    mixed_dataset = MixedDataset(
        mixtures=mixtures,
        seed=seed,
        streaming=True,
        rank=rank,
        world_size=world_size,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    # Wrap with packing if enabled
    if packed:
        dataset = PackedDataset(
            dataset=mixed_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenize_batch_size=256,
        )
    else:
        dataset = mixed_dataset

    # Note: We support num_workers > 0 for streaming datasets
    # The _worker_init_fn handles per-worker seeding for reproducibility
    # Each worker gets different data via the streaming shuffle buffer
    # persistent_workers avoids worker spawn overhead between epochs

    import logging
    logging.getLogger(__name__).info(
        f"Creating dataloader with num_workers={num_workers} (CPU count: {os.cpu_count()})"
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Avoid worker spawn overhead
    )

    return dataloader, mixed_dataset


def _create_single_source_dataloader(
    dataset_config: dict,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    rank: int,
    world_size: int,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    seed: int,
    packed: bool,
    text_column: str,
) -> tuple[DataLoader, None]:
    """Create dataloader for single dataset source."""
    import time
    import logging
    from datasets import load_dataset

    logger = logging.getLogger(__name__)

    # Load dataset
    path = dataset_config["path"]
    name = dataset_config.get("name")
    split = dataset_config.get("split", "train")
    data_files = dataset_config.get("data_files")  # Optional: explicit file paths

    logger.info(f"Loading single-source dataset: {path} (subset={name}, split={split})")
    start = time.time()

    # Use data_files if specified (fixes HuggingFace C4 streaming issues)
    # Reference: https://github.com/huggingface/datasets/issues/5574
    load_kwargs = {
        "path": path,
        "split": split,
        "streaming": True,
    }
    if data_files:
        load_kwargs["data_files"] = data_files
        logger.info(f"Using explicit data_files: {data_files}")
    elif name:
        load_kwargs["name"] = name

    ds = load_dataset(**load_kwargs)
    elapsed = time.time() - start
    logger.info(f"Dataset loaded in {elapsed:.1f}s")

    # Apply distributed sharding
    if world_size > 1:
        from datasets.distributed import split_dataset_by_node
        ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

    # Apply shuffle
    ds = ds.shuffle(seed=seed + rank, buffer_size=10000)

    # Create wrapper dataset that yields dicts with 'text' key
    class SingleSourceDataset(IterableDataset):
        def __init__(self, hf_dataset, text_col: str):
            self.dataset = hf_dataset
            self.text_col = text_col

        def __iter__(self):
            for item in self.dataset:
                text = item.get(self.text_col, item.get("text", ""))
                if text:
                    yield {"text": text}

    single_dataset = SingleSourceDataset(ds, text_column)

    # Wrap with packing
    if packed:
        dataset = PackedDataset(
            dataset=single_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            tokenize_batch_size=256,
        )
    else:
        dataset = single_dataset

    # Note: We support num_workers > 0 for streaming datasets
    # The _worker_init_fn handles per-worker seeding for reproducibility

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Avoid worker spawn overhead
    )

    return dataloader, None


def create_probe_dataloaders(
    probe_config: DictConfig | dict,
    tokenizer: Any,
    max_length: int = 2048,
    batch_size: int = 4,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create probe dataloaders for multi-domain influence calculation.

    Supports two probe configuration modes:

    1. Multi-domain mode (MobileLLM-R1 style):
       ```yaml
       probe:
         domains:
           web_edu:
             path: HuggingFaceFW/fineweb-edu
             samples: 200
           code:
             path: nick007x/github-code-2025
             samples: 200
       ```

    2. Single probe mode (backward compatibility):
       ```yaml
       probe:
         path: HuggingFaceFW/fineweb-edu
         subset: sample-10BT
         size: 1000
       ```

    Args:
        probe_config: Probe configuration
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for probe evaluation
        seed: Random seed

    Returns:
        Dictionary mapping domain names to DataLoaders
        For single probe mode, returns {"default": DataLoader}
    """
    if isinstance(probe_config, DictConfig):
        probe_config = OmegaConf.to_container(probe_config, resolve=True)

    # Check for multi-domain mode
    domains = probe_config.get("domains")
    if domains:
        return _create_multi_domain_probes(
            domains=domains,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            seed=seed,
        )

    # Single probe mode
    if probe_config.get("path"):
        return _create_single_probe(
            probe_config=probe_config,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
            seed=seed,
        )

    raise ValueError(
        "Probe config must have either 'domains' (multi-domain) or 'path' (single probe)"
    )


def _create_multi_domain_probes(
    domains: dict,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    """Create probe dataloaders for multiple domains."""
    domain_loaders = {}

    for domain, domain_cfg in domains.items():
        if not isinstance(domain_cfg, dict):
            continue

        probe_config = DomainProbeConfig(
            domain=domain,
            path=domain_cfg.get("path", ""),
            subset=domain_cfg.get("subset"),
            split=domain_cfg.get("split", "train"),
            samples=domain_cfg.get("samples", 2000),
            text_column=domain_cfg.get("text_column", "text"),
        )

        # Create probe dataset
        probe_dataset = DomainProbeDataset(
            config=probe_config,
            streaming=True,
            seed=seed,
        )

        # Wrap with packing
        packed_dataset = PackedDataset(
            dataset=probe_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        # Create DataLoader
        loader = DataLoader(
            packed_dataset,
            batch_size=batch_size,
            num_workers=0,  # Single worker for probes
            pin_memory=True,
        )

        domain_loaders[domain] = loader

    return domain_loaders


def _create_single_probe(
    probe_config: dict,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    """Create a single probe dataloader."""
    config = DomainProbeConfig(
        domain="default",
        path=probe_config["path"],
        subset=probe_config.get("subset"),
        split=probe_config.get("split", "train"),
        samples=probe_config.get("size", 1000),
        text_column=probe_config.get("text_column", "text"),
    )

    probe_dataset = DomainProbeDataset(
        config=config,
        streaming=True,
        seed=seed,
    )

    packed_dataset = PackedDataset(
        dataset=probe_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    loader = DataLoader(
        packed_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )

    return {"default": loader}


def get_dataset_weights_from_config(config: DictConfig | dict) -> dict[str, float]:
    """Extract initial dataset weights from configuration.

    Useful for logging initial weights before influence updates.

    Args:
        config: Data configuration with 'sources'

    Returns:
        Dictionary mapping source names to weights
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    sources = config.get("sources", [])
    weights = {}
    total = 0.0

    for source in sources:
        name = source.get("name", source["path"])
        weight = source.get("weight", 1.0)
        weights[name] = weight
        total += weight

    # Normalize
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights
