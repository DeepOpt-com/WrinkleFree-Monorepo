"""Dataset mixing utilities for pretraining.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) Table 1
Implements weighted mixing of multiple datasets.
"""

import collections
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional, List, Any, Union

import torch
from torch.utils.data import IterableDataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class DatasetMixture:
    """Configuration for a single dataset in the mixture."""

    name: str
    weight: float  # Sampling weight (will be normalized)
    path: str  # HuggingFace dataset path or local path
    subset: Optional[str] = None
    split: str = "train"
    text_column: Union[str, List[str]] = "text"  # Single column or list to concatenate
    text_separator: str = "\n\n"  # Separator between concatenated columns


class MixedDataset(IterableDataset):
    """Mixed dataset with weighted sampling from multiple sources.

    Supports streaming for large datasets.
    Reference: MobileLLM-R1 uses ~2T tokens per phase with specific mixing ratios.
    """

    def __init__(
        self,
        mixtures: List[DatasetMixture],
        seed: int = 42,
        streaming: bool = True,
        rank: int = 0,
        world_size: int = 1,
        shuffle_buffer_size: int = 10000,
        homogeneous_batch_size: int = 0,
    ):
        """Initialize mixed dataset.

        Args:
            mixtures: List of dataset mixtures with weights
            seed: Random seed for reproducibility
            streaming: Whether to use streaming mode (recommended for large datasets)
            rank: Process rank for distributed training
            world_size: Total number of processes for distributed training
            shuffle_buffer_size: Size of shuffle buffer for streaming datasets
            homogeneous_batch_size: If > 0, yield this many consecutive samples from
                the same domain before sampling a new domain. This implements the
                ODM paper's homogeneous batch sampling strategy (arxiv:2312.02406).
        """
        # Filter out mixtures with weight=0 (disabled datasets)
        self.mixtures = [m for m in mixtures if m.weight > 0]
        self.seed = seed
        self.streaming = streaming
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.homogeneous_batch_size = homogeneous_batch_size

        # Normalize weights (only among active mixtures)
        total_weight = sum(m.weight for m in self.mixtures)
        self.normalized_weights = torch.tensor(
            [m.weight / total_weight for m in self.mixtures]
        )

        self._datasets = None
        self._iterators = None
        self._epoch = 0  # Track epoch for proper iterator reset

        # Sampling statistics tracking
        self._sample_counts: dict[str, int] = {m.name: 0 for m in mixtures}
        self._exhausted_datasets: set[str] = set()

    def update_weights_from_influence(self, weights: dict[str, float]):
        """Update mixture weights based on influence calculation.

        This method allows dynamic weight adjustment during training
        based on influence function analysis.

        Args:
            weights: Dictionary mapping mixture names to new weights
        """
        new_weights = []
        for mixture in self.mixtures:
            if mixture.name in weights:
                new_weights.append(weights[mixture.name])
            else:
                # Keep original weight if not specified
                new_weights.append(mixture.weight)

        total = sum(new_weights)
        if total > 0:
            self.normalized_weights = torch.tensor([w / total for w in new_weights])

    def get_current_weights(self) -> dict[str, float]:
        """Get current mixture weights.

        Returns:
            Dictionary mapping mixture names to current weights
        """
        return {
            m.name: self.normalized_weights[i].item()
            for i, m in enumerate(self.mixtures)
        }

    def get_sampling_stats(self) -> dict:
        """Return actual sampling statistics for debugging.

        Returns:
            Dictionary with counts, observed_weights, configured_weights,
            exhausted datasets, and total samples.
        """
        total = sum(self._sample_counts.values())
        return {
            "counts": dict(self._sample_counts),
            "observed_weights": {
                k: v / total for k, v in self._sample_counts.items()
            }
            if total > 0
            else {},
            "configured_weights": self.get_current_weights(),
            "exhausted": list(self._exhausted_datasets),
            "total_samples": total,
        }

    def reset_stats(self) -> None:
        """Reset sampling statistics. Call at start of new epoch."""
        self._sample_counts = {m.name: 0 for m in self.mixtures}
        self._exhausted_datasets = set()

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for proper shuffling with streaming datasets.

        This must be called at the start of each epoch to ensure different
        samples are seen. Without this, the same shuffle buffer order is used.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        self._epoch = epoch
        # Force iterator recreation on next __iter__ call
        self._iterators = None

    def get_source_loaders(
        self,
        tokenizer: Any,
        batch_size: int = 4,
        max_length: int = 2048,
        samples_per_source: int = 1000,
    ) -> dict[str, DataLoader]:
        """Create individual DataLoaders for each source for influence computation.

        Used by InfluenceDistillation to compute per-source influence scores
        and optimize mixture weights.

        Args:
            tokenizer: Tokenizer for encoding text
            batch_size: Batch size for each loader
            max_length: Maximum sequence length
            samples_per_source: Number of samples per source dataset

        Returns:
            Dict mapping source names to DataLoaders
        """
        loaders = {}
        for mixture in self.mixtures:
            # Create a probe-style dataset for each source
            probe_config = DomainProbeConfig(
                domain=mixture.name,
                path=mixture.path,
                subset=mixture.subset,
                split=mixture.split,
                samples=samples_per_source,
                text_column=mixture.text_column,
            )
            probe_ds = DomainProbeDataset(probe_config)
            packed_ds = PackedDataset(probe_ds, tokenizer, max_length)
            loaders[mixture.name] = DataLoader(packed_ds, batch_size=batch_size)

        return loaders

    def _load_datasets(self):
        """Lazy load datasets with distributed sharding support."""
        import os
        import time
        from datasets import load_dataset

        # Disable KMP affinity to prevent hangs on multi-core systems
        # Reference: https://github.com/huggingface/datasets/issues/6079
        if "KMP_AFFINITY" not in os.environ:
            os.environ["KMP_AFFINITY"] = "disabled"

        self._datasets = []
        self._iterators = []

        # Filter out mixtures with weight=0
        active_mixtures = [m for m in self.mixtures if m.weight > 0]
        total = len(active_mixtures)
        for i, mixture in enumerate(active_mixtures):
            logger.info(f"Loading dataset {i+1}/{total}: {mixture.name} ({mixture.path})")
            start = time.time()
            ds = load_dataset(
                mixture.path,
                name=mixture.subset,
                split=mixture.split,
                streaming=self.streaming,
            )
            elapsed = time.time() - start
            logger.info(f"Dataset {mixture.name} loaded in {elapsed:.1f}s")

            # Apply distributed sharding if multi-GPU
            if self.world_size > 1 and self.streaming:
                from datasets.distributed import split_dataset_by_node

                ds = split_dataset_by_node(
                    ds, rank=self.rank, world_size=self.world_size
                )

            # Apply shuffle buffer for streaming datasets
            # Include epoch in seed to ensure different samples each epoch
            if self.streaming and self.shuffle_buffer_size > 0:
                ds = ds.shuffle(
                    seed=self.seed + self.rank + self._epoch * 1000,
                    buffer_size=self.shuffle_buffer_size,
                )

            self._datasets.append(ds)
            self._iterators.append(iter(ds))

    def _get_sample(self, idx: int) -> dict | None:
        """Get a sample from the specified dataset.

        Args:
            idx: Dataset index

        Returns:
            Sample dictionary, or None if dataset is exhausted
        """
        try:
            sample = next(self._iterators[idx])
        except StopIteration:
            # Dataset exhausted - mark it and return None (don't restart!)
            dataset_name = self.mixtures[idx].name
            self._exhausted_datasets.add(dataset_name)
            logger.info(
                f"Dataset '{dataset_name}' exhausted after "
                f"{self._sample_counts[dataset_name]} samples"
            )
            return None

        # Track sample count
        self._sample_counts[self.mixtures[idx].name] += 1

        # Extract text using the configured column(s)
        text_col = self.mixtures[idx].text_column
        text = None

        if isinstance(text_col, list):
            # Concatenate multiple columns
            parts = [sample.get(col, "") for col in text_col if sample.get(col)]
            if parts:
                text = self.mixtures[idx].text_separator.join(parts)
        elif text_col in sample:
            text = sample[text_col]

        # Fallback if text not found - but warn loudly!
        if not text:
            available_keys = list(sample.keys())
            logger.warning(
                f"Configured text_column '{text_col}' not found in sample from "
                f"'{self.mixtures[idx].name}'. Available keys: {available_keys}. "
                f"Attempting fallback..."
            )
            if "content" in sample:
                text = sample["content"]
            else:
                # Try to find any text-like field
                for key in ["text", "content", "document", "passage"]:
                    if key in sample:
                        text = sample[key]
                        break
                else:
                    # Last resort - stringify the sample (but warn!)
                    logger.error(
                        f"No text field found in sample from '{self.mixtures[idx].name}'. "
                        f"Stringifying entire sample as fallback. Keys: {available_keys}"
                    )
                    text = str(sample)

        return {
            "text": text,
            "source": self.mixtures[idx].name,
        }

    def __iter__(self) -> Iterator[dict]:
        """Iterate over mixed samples.

        When a dataset is exhausted, it is excluded from future sampling
        (weights are renormalized among remaining datasets).
        Iteration stops when all datasets are exhausted.

        If homogeneous_batch_size > 0, yields consecutive samples from the
        same domain before sampling a new domain (ODM paper style).
        """
        # Always reload datasets/iterators to ensure fresh iteration
        # This fixes the bug where iterators were exhausted but not reset
        # Also ensures epoch-varying shuffle seeds take effect
        self._load_datasets()

        # Reset stats for new epoch
        self.reset_stats()

        # Use epoch-varying seed for sampling order reproducibility
        rng = torch.Generator().manual_seed(self.seed + self._epoch * 1000)

        # Increment epoch for next iteration
        self._epoch += 1

        # Track which datasets are still active (not exhausted)
        active_mask = torch.ones(len(self.mixtures), dtype=torch.bool)

        # For homogeneous batch mode
        current_domain_idx = None
        samples_from_current_domain = 0

        while active_mask.any():
            # Renormalize weights among active datasets only
            masked_weights = self.normalized_weights * active_mask.float()
            weight_sum = masked_weights.sum()
            if weight_sum == 0:
                break  # All datasets exhausted
            masked_weights = masked_weights / weight_sum

            # Determine which dataset to sample from
            if self.homogeneous_batch_size > 0:
                # ODM-style homogeneous batches: yield N samples from same domain
                if (current_domain_idx is None or
                    samples_from_current_domain >= self.homogeneous_batch_size or
                    not active_mask[current_domain_idx]):
                    # Sample a new domain
                    current_domain_idx = torch.multinomial(
                        masked_weights, 1, generator=rng
                    ).item()
                    samples_from_current_domain = 0
                idx = current_domain_idx
            else:
                # Standard mixed sampling
                idx = torch.multinomial(masked_weights, 1, generator=rng).item()

            sample = self._get_sample(idx)
            if sample is None:
                # Dataset exhausted - mark as inactive
                active_mask[idx] = False
                remaining = active_mask.sum().item()
                logger.info(
                    f"Remaining active datasets: {remaining}/{len(self.mixtures)}"
                )
                # Force new domain selection on next iteration
                if self.homogeneous_batch_size > 0:
                    current_domain_idx = None
            else:
                samples_from_current_domain += 1
                yield sample

        # Log final stats when epoch completes
        stats = self.get_sampling_stats()
        logger.info(f"Epoch complete. Sampling stats: {stats}")


class PackedDataset(IterableDataset):
    """Dataset that packs multiple sequences into fixed-length chunks.

    Improves training efficiency by minimizing padding.
    Uses batched tokenization for better performance.

    Generates position_ids that reset at document boundaries for proper
    RoPE positioning when using FlashAttention-based sequence packing.
    See: https://huggingface.co/blog/sirluk/llm-sequence-packing
    """

    def __init__(
        self,
        dataset: IterableDataset,
        tokenizer: Any,
        max_length: int = 2048,
        separator_token_id: Optional[int] = None,
        tokenize_batch_size: int = 256,
    ):
        """Initialize packed dataset.

        Args:
            dataset: Source dataset (yields dicts with 'text' key)
            tokenizer: Tokenizer for encoding
            max_length: Target sequence length
            separator_token_id: Token ID to use between documents (default: EOS)
            tokenize_batch_size: Number of texts to batch tokenize at once (higher = faster but more memory)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator_token_id = separator_token_id or tokenizer.eos_token_id
        self.tokenize_batch_size = tokenize_batch_size

    def _compute_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute position IDs that reset at each document boundary.

        This enables proper RoPE positioning for FlashAttention-based sequence
        packing, where each document should have positions starting from 0.

        Uses vectorized implementation for ~10x speedup over serial loop.

        Args:
            input_ids: Token IDs for the packed sequence

        Returns:
            Position IDs that reset after each separator token
        """
        from data_handler.data.packing import compute_position_ids_vectorized

        return compute_position_ids_vectorized(input_ids, self.separator_token_id)

    def _batch_tokenize(self, texts: List[str]) -> List[List[int]]:
        """Batch tokenize texts for better performance.

        Args:
            texts: List of text strings to tokenize

        Returns:
            List of token ID lists
        """
        if not texts:
            return []

        # Use batch tokenization (much faster than per-doc encoding)
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return encoded["input_ids"]

    def __iter__(self) -> Iterator[dict]:
        """Iterate over packed sequences with batched tokenization.

        Preserves domain/source info from the underlying dataset for ODM.
        Each packed sequence gets the domain of its first document.
        """
        token_buffer: collections.deque = collections.deque()
        text_buffer: List[str] = []
        source_buffer: List[str] = []  # Track sources for each text
        # Track source for current packed chunk - "unknown" if no source info available
        current_chunk_source: str = "unknown"
        # Track source for the NEXT chunk (set when yielding overflow)
        next_chunk_source: Optional[str] = None

        for sample in self.dataset:
            text_buffer.append(sample["text"])
            source_buffer.append(sample.get("source", sample.get("domain", "unknown")))

            # Batch tokenize when buffer is full
            if len(text_buffer) >= self.tokenize_batch_size:
                token_batches = self._batch_tokenize(text_buffer)
                for i, tokens in enumerate(token_batches):
                    # Track source: use pending source from previous yield, or first doc's source
                    if next_chunk_source is not None:
                        current_chunk_source = next_chunk_source
                        next_chunk_source = None
                    elif i < len(source_buffer):
                        # For homogeneous batches, all docs have same source
                        # For mixed batches, use first doc's source
                        current_chunk_source = source_buffer[i]
                    # Add separator between documents
                    if token_buffer:
                        token_buffer.append(self.separator_token_id)
                    token_buffer.extend(tokens)
                    # Track last source added for overflow handling
                    if i < len(source_buffer):
                        next_chunk_source = source_buffer[i]
                text_buffer = []
                source_buffer = []

                # Yield complete chunks
                while len(token_buffer) >= self.max_length:
                    chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    position_ids = self._compute_position_ids(input_ids)
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                        "position_ids": position_ids,
                        "labels": input_ids.clone(),  # For causal LM training
                        "domain": current_chunk_source,  # For ODM - never None
                    }
                    # Next chunk inherits source from overflow tokens
                    current_chunk_source = next_chunk_source if next_chunk_source else current_chunk_source

        # Process remaining texts in buffer
        if text_buffer:
            token_batches = self._batch_tokenize(text_buffer)
            for i, tokens in enumerate(token_batches):
                # Track source: use pending source from previous yield, or first doc's source
                if next_chunk_source is not None:
                    current_chunk_source = next_chunk_source
                    next_chunk_source = None
                elif i < len(source_buffer):
                    current_chunk_source = source_buffer[i]
                if token_buffer:
                    token_buffer.append(self.separator_token_id)
                token_buffer.extend(tokens)
                # Track last source added for overflow handling
                if i < len(source_buffer):
                    next_chunk_source = source_buffer[i]

            # Yield remaining complete chunks
            while len(token_buffer) >= self.max_length:
                chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                position_ids = self._compute_position_ids(input_ids)
                yield {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                    "position_ids": position_ids,
                    "labels": input_ids.clone(),  # For causal LM training
                    "domain": current_chunk_source,  # For ODM - never None
                }
                # Next chunk inherits source from overflow tokens
                current_chunk_source = next_chunk_source if next_chunk_source else current_chunk_source


def create_mixed_dataset(
    config: dict,
    tokenizer: Any,
    packing: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> IterableDataset:
    """Create mixed dataset from configuration.

    Args:
        config: Data configuration dict with 'mixtures' list
        tokenizer: Tokenizer for packing
        packing: Whether to use sequence packing
        rank: Process rank for distributed training
        world_size: Total number of processes

    Returns:
        Configured dataset
    """
    mixtures = [
        DatasetMixture(**m) if isinstance(m, dict) else m
        for m in config.get("mixtures", [])
    ]

    dataset = MixedDataset(
        mixtures=mixtures,
        seed=config.get("seed", 42),
        streaming=config.get("streaming", True),
        rank=rank,
        world_size=world_size,
        shuffle_buffer_size=config.get("shuffle_buffer_size", 10000),
        homogeneous_batch_size=config.get("homogeneous_batch_size", 0),
    )

    if packing:
        dataset = PackedDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            max_length=config.get("max_length", 2048),
            tokenize_batch_size=config.get("tokenize_batch_size", 256),
        )

    return dataset


@dataclass
class DomainProbeConfig:
    """Configuration for a single domain probe."""

    domain: str  # e.g., 'code', 'math', 'knowledge'
    path: str  # HuggingFace dataset path
    subset: Optional[str] = None
    split: str = "train"
    samples: int = 2000
    text_column: Union[str, List[str]] = "text"  # Single column or list to concatenate
    text_separator: str = "\n\n"  # Separator between concatenated columns


class DomainProbeDataset(IterableDataset):
    """Dataset for domain-specific probe samples.

    Used in MobileLLM-R1 style multi-domain influence calculation.
    """

    def __init__(
        self,
        config: DomainProbeConfig,
        streaming: bool = True,
        seed: int = 42,
    ):
        """Initialize domain probe dataset.

        Args:
            config: Domain probe configuration
            streaming: Whether to use streaming mode
            seed: Random seed for reproducibility
        """
        self.config = config
        self.streaming = streaming
        self.seed = seed
        self._dataset = None
        self._samples_yielded = 0

    def _load_dataset(self):
        """Load the probe dataset."""
        from datasets import load_dataset

        ds = load_dataset(
            self.config.path,
            name=self.config.subset,
            split=self.config.split,
            streaming=self.streaming,
        )

        if self.streaming:
            ds = ds.shuffle(seed=self.seed, buffer_size=10000)

        self._dataset = ds

    def __iter__(self) -> Iterator[dict]:
        """Iterate over probe samples up to the configured limit."""
        if self._dataset is None:
            self._load_dataset()

        self._samples_yielded = 0
        text_col = self.config.text_column

        for sample in self._dataset:
            if self._samples_yielded >= self.config.samples:
                break

            # Extract text (supports list of columns for concatenation)
            text = None
            if isinstance(text_col, list):
                # Concatenate multiple columns
                parts = [sample.get(col, "") for col in text_col if sample.get(col)]
                if parts:
                    text = self.config.text_separator.join(parts)
            elif text_col in sample:
                text = sample[text_col]

            # Fallback if text not found
            if not text:
                if "content" in sample:
                    text = sample["content"]
                elif "text" in sample:
                    text = sample["text"]
                else:
                    text = str(sample)

            self._samples_yielded += 1
            yield {
                "text": text,
                "domain": self.config.domain,
            }


def create_domain_probe_loaders(
    config: dict,
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 2048,
    streaming: bool = True,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create DataLoaders for each domain probe.

    Args:
        config: Probe configuration dict with 'domains' mapping
        tokenizer: Tokenizer for encoding
        batch_size: Batch size for probe evaluation
        max_length: Maximum sequence length
        streaming: Whether to use streaming mode
        seed: Random seed

    Returns:
        Dictionary mapping domain names to DataLoaders
    """
    domain_loaders = {}
    domains_config = config.get("domains", {})

    for domain, domain_cfg in domains_config.items():
        if isinstance(domain_cfg, dict):
            probe_config = DomainProbeConfig(
                domain=domain,
                path=domain_cfg.get("path", ""),
                subset=domain_cfg.get("subset"),
                split=domain_cfg.get("split", "train"),
                samples=domain_cfg.get("samples", 2000),
                text_column=domain_cfg.get("text_column", "text"),
                text_separator=domain_cfg.get("text_separator", "\n\n"),
            )
        else:
            # Skip if not a valid config dict
            continue

        # Create probe dataset
        probe_dataset = DomainProbeDataset(
            config=probe_config,
            streaming=streaming,
            seed=seed,
        )

        # Wrap with packing for consistent sequence lengths
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


def get_domain_weights(config: dict) -> dict[str, float]:
    """Extract domain weights from probe configuration.

    Args:
        config: Probe configuration dict

    Returns:
        Dictionary mapping domain names to weights
    """
    weights = config.get("domain_weights", {})

    # Ensure all domains have weights
    domains = config.get("domains", {})
    if isinstance(domains, dict):
        n_domains = len(domains)
        default_weight = 1.0 / max(n_domains, 1)

        for domain in domains:
            if domain not in weights:
                weights[domain] = default_weight

    return weights


def _worker_init_fn(worker_id: int) -> None:
    """Initialize worker with unique random seed.

    This ensures each DataLoader worker processes different data
    when using IterableDataset.

    Args:
        worker_id: The worker ID assigned by DataLoader
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset

        # Handle nested datasets (e.g., PackedDataset wrapping MixedDataset)
        inner_dataset = dataset
        while hasattr(inner_dataset, "dataset"):
            inner_dataset = inner_dataset.dataset

        # Offset the seed per worker to ensure different data
        if hasattr(inner_dataset, "seed"):
            inner_dataset.seed = inner_dataset.seed + worker_id

        # Force reload of datasets with new seed
        if hasattr(inner_dataset, "_datasets"):
            inner_dataset._datasets = None
            inner_dataset._iterators = None


def create_mixed_dataloader(
    config: dict,
    tokenizer: Any,
    batch_size: int,
    packing: bool = True,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create DataLoader for mixed dataset with optimized settings.

    Args:
        config: Data configuration dict with 'mixtures' list
        tokenizer: Tokenizer for encoding
        batch_size: Batch size
        packing: Whether to use sequence packing
        rank: Process rank for distributed training
        world_size: Total number of processes
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs

    Returns:
        Configured DataLoader
    """
    dataset = create_mixed_dataset(
        config=config,
        tokenizer=tokenizer,
        packing=packing,
        rank=rank,
        world_size=world_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
