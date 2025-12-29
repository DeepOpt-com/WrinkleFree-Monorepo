"""Pre-training dataset for Stage 2 continue pre-training.

Provides a unified PretrainDataset class that supports:
- Single source (e.g., "HuggingFaceFW/fineweb") or multi-source with weights
- Sequence packing with position_ids reset at document boundaries
- Distributed training with sharding
"""

import collections
import logging
from typing import Dict, Iterator, List, Optional

import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PretrainDataset(IterableDataset):
    """
    Unified pre-training dataset with packing and multi-source support.

    Supports both single-source and multi-source modes:
    - Single source: Provide dataset_path (backward compatible with PackedPretrainDataset)
    - Multi-source: Provide sources list with weights (replaces MixedPretrainDataset)

    Features:
    - Sequence packing: Concatenates documents with EOS separators (default)
    - Position IDs: Reset at document boundaries for proper RoPE
    - Distributed: Automatic sharding across GPUs
    - Batched tokenization: For better performance

    Args:
        tokenizer: Tokenizer for text encoding
        max_length: Maximum sequence length
        dataset_path: HuggingFace dataset path (single-source mode)
        dataset_name: Dataset configuration name (single-source mode)
        sources: List of source configs with 'path', 'weight', etc. (multi-source mode)
        split: Dataset split
        text_column: Column name containing text (single-source mode)
        packed: Whether to use sequence packing (default: True)
        seed: Random seed for shuffling
        rank: Process rank for distributed training
        world_size: Total number of processes
        shuffle_buffer_size: Size of shuffle buffer for streaming
        tokenize_batch_size: Number of texts to batch tokenize at once
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        # Single-source API (backward compat)
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        # Multi-source API
        sources: Optional[List[dict]] = None,
        # Common params
        split: str = "train",
        text_column: str = "text",
        packed: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        shuffle_buffer_size: int = 10000,
        tokenize_batch_size: int = 256,
    ):
        # Validate: either dataset_path OR sources, not both
        if dataset_path and sources:
            raise ValueError("Provide dataset_path OR sources, not both")
        if not dataset_path and not sources:
            raise ValueError("Must provide dataset_path or sources")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.text_column = text_column
        self.packed = packed
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.tokenize_batch_size = tokenize_batch_size

        # Source configuration
        self.multi_source = sources is not None
        if self.multi_source:
            self.sources = sources
            self.weights = [s.get("weight", 1.0) for s in sources]
            # Normalize weights to probabilities
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
            self._loaded_datasets: Optional[List] = None
        else:
            self.dataset_path = dataset_path
            self.dataset_name = dataset_name

        # Special tokens
        self.eos_token_id = tokenizer.eos_token_id or 2

        # Lazy-loaded dataset
        self._dataset = None

    def _load_dataset(self):
        """Load dataset lazily with distributed sharding support."""
        if self._dataset is not None:
            return

        if self.multi_source:
            self._load_multi_source()
        else:
            self._load_single_source()

    def _load_single_source(self):
        """Load a single dataset source."""
        logger.info(f"Loading dataset: {self.dataset_path}")

        self._dataset = load_dataset(
            self.dataset_path,
            self.dataset_name,
            split=self.split,
            streaming=True,
        )

        self._apply_distributed_sharding()
        self._apply_shuffle()

        logger.info(f"Loaded streaming dataset: {self.dataset_path}")

    def _load_multi_source(self):
        """Load and interleave multiple dataset sources."""
        logger.info(f"Loading {len(self.sources)} data sources...")

        datasets = []
        for source in self.sources:
            text_col = source.get("text_column", "text")
            logger.info(f"  Loading {source.get('name', source['path'])} (weight={source.get('weight', 1.0)}, text_col={text_col})")
            ds = load_dataset(
                source["path"],
                name=source.get("subset"),
                split=self.split,
                streaming=True,
            )
            # Select only the text column and rename to 'text' for uniform schema
            # This is critical for interleave_datasets to work with heterogeneous sources
            if text_col != "text":
                ds = ds.rename_column(text_col, "text")
            ds = ds.select_columns(["text"])
            datasets.append(ds)

        self._loaded_datasets = datasets

        # Interleave with weights
        self._dataset = interleave_datasets(
            datasets,
            probabilities=self.weights,
            seed=self.seed,
            stopping_strategy="first_exhausted",
        )

        self._apply_distributed_sharding()
        self._apply_shuffle()

        logger.info(f"Created interleaved dataset with {len(self.sources)} sources")

    def _apply_distributed_sharding(self):
        """Apply distributed sharding if multi-GPU."""
        if self.world_size > 1:
            from datasets.distributed import split_dataset_by_node

            self._dataset = split_dataset_by_node(
                self._dataset, rank=self.rank, world_size=self.world_size
            )

    def _apply_shuffle(self):
        """Apply shuffle buffer."""
        if self.shuffle_buffer_size > 0:
            self._dataset = self._dataset.shuffle(
                seed=self.seed + self.rank, buffer_size=self.shuffle_buffer_size
            )

    def _compute_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute position IDs that reset at each document boundary (EOS token).

        This enables proper RoPE positioning for FlashAttention-based sequence
        packing, where each document should have positions starting from 0.

        Args:
            input_ids: Token IDs for the packed sequence

        Returns:
            Position IDs that reset after each EOS token
        """
        position_ids = torch.zeros_like(input_ids)
        pos = 0
        for i, token_id in enumerate(input_ids):
            position_ids[i] = pos
            if token_id == self.eos_token_id:
                pos = 0  # Reset position after EOS
            else:
                pos += 1
        return position_ids

    def _batch_tokenize(self, texts: List[str]) -> List[List[int]]:
        """Batch tokenize texts for better performance."""
        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return encoded["input_ids"]

    def _get_text_column(self, item: dict) -> str:
        """Determine text column name from item (for multi-source)."""
        # Check configured column first
        if self.text_column in item:
            return self.text_column
        # Try common column names
        for col in ["text", "content", "code"]:
            if col in item:
                return col
        # Fallback to first column
        return list(item.keys())[0]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over dataset."""
        self._load_dataset()

        if self.packed:
            yield from self._iter_packed()
        else:
            yield from self._iter_chunked()

    def _iter_packed(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate with sequence packing (documents concatenated with EOS)."""
        token_buffer: collections.deque = collections.deque()
        text_buffer: List[str] = []

        for example in self._dataset:
            # Get text from appropriate column
            text_col = self._get_text_column(example) if self.multi_source else self.text_column
            text = example.get(text_col)

            # Skip invalid items
            if not text or not isinstance(text, str):
                continue

            text_buffer.append(text)

            # Batch tokenize when buffer is full
            if len(text_buffer) >= self.tokenize_batch_size:
                token_batches = self._batch_tokenize(text_buffer)
                for tokens in token_batches:
                    token_buffer.extend(tokens)
                    token_buffer.append(self.eos_token_id)
                text_buffer = []

                # Yield when buffer has enough tokens
                while len(token_buffer) >= self.max_length:
                    chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    labels = input_ids.clone()
                    position_ids = self._compute_position_ids(input_ids)

                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": torch.ones_like(input_ids),
                        "position_ids": position_ids,
                    }

        # Process remaining texts in buffer
        if text_buffer:
            token_batches = self._batch_tokenize(text_buffer)
            for tokens in token_batches:
                token_buffer.extend(tokens)
                token_buffer.append(self.eos_token_id)

            while len(token_buffer) >= self.max_length:
                chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                labels = input_ids.clone()
                position_ids = self._compute_position_ids(input_ids)

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": torch.ones_like(input_ids),
                    "position_ids": position_ids,
                }

    def _iter_chunked(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate without packing (simple chunking, no position_ids)."""
        token_buffer: collections.deque = collections.deque()
        text_buffer: List[str] = []

        for example in self._dataset:
            text_col = self._get_text_column(example) if self.multi_source else self.text_column
            text = example.get(text_col)

            if not text or not isinstance(text, str):
                continue

            text_buffer.append(text)

            # Batch tokenize when buffer is full
            if len(text_buffer) >= self.tokenize_batch_size:
                token_batches = self._batch_tokenize(text_buffer)
                for tokens in token_batches:
                    token_buffer.extend(tokens)
                text_buffer = []

                # Yield chunks
                while len(token_buffer) >= self.max_length:
                    chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    labels = input_ids.clone()

                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": torch.ones_like(input_ids),
                    }

        # Process remaining
        if text_buffer:
            token_batches = self._batch_tokenize(text_buffer)
            for tokens in token_batches:
                token_buffer.extend(tokens)

            while len(token_buffer) >= self.max_length:
                chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                labels = input_ids.clone()

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": torch.ones_like(input_ids),
                }

    def update_weights(self, new_weights: List[float]) -> None:
        """Update mixture weights (for CheaperTraining dynamic reweighting).

        Only valid for multi-source mode.

        Args:
            new_weights: New weights for each source (will be normalized)
        """
        if not self.multi_source:
            raise ValueError("update_weights() only valid for multi-source mode")

        logger.info(f"Updating mixture weights: {new_weights}")

        # Normalize
        total = sum(new_weights)
        self.weights = [w / total for w in new_weights]

        # Recreate interleaved dataset
        if self._loaded_datasets is not None:
            self._dataset = interleave_datasets(
                self._loaded_datasets,
                probabilities=self.weights,
                seed=self.seed,
                stopping_strategy="first_exhausted",
            )
            self._apply_distributed_sharding()
            self._apply_shuffle()

    def get_current_weights(self) -> Dict[str, float]:
        """Get current mixture weights as a dict (for CheaperTraining influence).

        Returns:
            Dict mapping source name to weight
        """
        if not self.multi_source:
            raise ValueError("get_current_weights() only valid for multi-source mode")

        weights_dict = {}
        for i, source in enumerate(self.sources):
            name = source.get("name", source["path"])
            weights_dict[name] = self.weights[i]
        return weights_dict

    def update_weights_from_influence(self, new_weights: Dict[str, float]) -> None:
        """Update weights from influence calculation (for CheaperTraining).

        Args:
            new_weights: Dict mapping source name to new weight
        """
        if not self.multi_source:
            raise ValueError("update_weights_from_influence() only valid for multi-source mode")

        # Convert dict to list in same order as sources
        weights_list = []
        for source in self.sources:
            name = source.get("name", source["path"])
            weights_list.append(new_weights.get(name, self.weights[len(weights_list)]))

        self.update_weights(weights_list)


# Backward compatibility aliases
PackedPretrainDataset = PretrainDataset
StreamingPretrainDataset = PretrainDataset


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

        # Offset the seed per worker to ensure different data
        if hasattr(dataset, "seed"):
            dataset.seed = dataset.seed + worker_id

        # Force reload of dataset with new seed
        if hasattr(dataset, "_dataset"):
            dataset._dataset = None


def create_pretrain_dataloader(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 2048,
    dataset_name: Optional[str] = None,
    split: str = "train",
    text_column: str = "text",
    num_workers: int = 4,
    seed: int = 42,
    packed: bool = True,
    rank: int = 0,
    world_size: int = 1,
    prefetch_factor: int = 2,
    shuffle_buffer_size: int = 10000,
    tokenize_batch_size: int = 256,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Create dataloader for pre-training with optimized settings.

    Args:
        dataset_path: HuggingFace dataset path
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        dataset_name: Dataset config name
        split: Dataset split
        text_column: Text column name
        num_workers: Number of data loading workers
        seed: Random seed
        packed: Whether to use packed sequences (default: True)
        rank: Process rank for distributed training
        world_size: Total number of processes
        prefetch_factor: Number of batches to prefetch per worker
        shuffle_buffer_size: Size of shuffle buffer for streaming datasets
        tokenize_batch_size: Number of texts to batch tokenize at once
        persistent_workers: Keep workers alive between epochs

    Returns:
        DataLoader for pre-training
    """
    dataset = PretrainDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
        text_column=text_column,
        packed=packed,
        seed=seed,
        rank=rank,
        world_size=world_size,
        shuffle_buffer_size=shuffle_buffer_size,
        tokenize_batch_size=tokenize_batch_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


def create_mixed_dataloader(
    sources: List[dict],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 2048,
    num_workers: int = 4,
    packed: bool = True,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    prefetch_factor: int = 2,
    shuffle_buffer_size: int = 10000,
    tokenize_batch_size: int = 256,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Create dataloader for mixed multi-source dataset.

    Args:
        sources: List of source configs with 'path', 'weight', 'name', etc.
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: Number of worker processes
        packed: Whether to use sequence packing (default: True)
        seed: Random seed
        rank: Process rank for distributed training
        world_size: Total number of processes
        prefetch_factor: Number of batches to prefetch per worker
        shuffle_buffer_size: Size of shuffle buffer for streaming datasets
        tokenize_batch_size: Number of texts to batch tokenize at once
        persistent_workers: Keep workers alive between epochs

    Returns:
        DataLoader instance
    """
    dataset = PretrainDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        sources=sources,
        packed=packed,
        seed=seed,
        rank=rank,
        world_size=world_size,
        shuffle_buffer_size=shuffle_buffer_size,
        tokenize_batch_size=tokenize_batch_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
