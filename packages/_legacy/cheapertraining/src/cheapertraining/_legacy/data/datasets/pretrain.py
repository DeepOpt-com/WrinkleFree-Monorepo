"""Pretraining dataset implementations.

Provides datasets for the pretraining phases with sequence packing.
"""

from typing import Iterator, Optional, Any, List

import torch
from torch.utils.data import IterableDataset, DataLoader


class PretrainDataset(IterableDataset):
    """Pretraining dataset with optional packing.

    Wraps a HuggingFace dataset for pretraining use.
    """

    def __init__(
        self,
        hf_dataset: Any,
        tokenizer: Any,
        max_length: int = 2048,
        text_column: str = "text",
        packing: bool = True,
    ):
        """Initialize pretraining dataset.

        Args:
            hf_dataset: HuggingFace dataset (streaming or regular)
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            text_column: Column name containing text
            packing: Whether to pack sequences
        """
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.packing = packing

    def _pack_sequences(self, tokens_iter: Iterator[List[int]]) -> Iterator[List[int]]:
        """Pack token sequences into fixed-length chunks.

        Args:
            tokens_iter: Iterator of token lists

        Yields:
            Fixed-length token lists
        """
        buffer = []
        eos_id = self.tokenizer.eos_token_id

        for tokens in tokens_iter:
            if buffer:
                buffer.append(eos_id)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length:
                yield buffer[:self.max_length]
                buffer = buffer[self.max_length:]

    def __iter__(self) -> Iterator[dict]:
        """Iterate over samples."""
        def token_generator():
            for sample in self.hf_dataset:
                text = sample.get(self.text_column, "")
                if not text:
                    continue

                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=not self.packing,
                    max_length=self.max_length if not self.packing else None,
                )
                yield tokens

        if self.packing:
            for packed_tokens in self._pack_sequences(token_generator()):
                yield {
                    "input_ids": torch.tensor(packed_tokens, dtype=torch.long),
                    "attention_mask": torch.ones(len(packed_tokens), dtype=torch.long),
                }
        else:
            for tokens in token_generator():
                # Pad to max_length
                padding_length = self.max_length - len(tokens)
                if padding_length > 0:
                    tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
                    attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
                else:
                    attention_mask = [1] * len(tokens)

                yield {
                    "input_ids": torch.tensor(tokens[:self.max_length], dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask[:self.max_length], dtype=torch.long),
                }


def create_pretrain_dataloader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for pretraining.

    Args:
        dataset: Pretraining dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Collate function is identity since dataset already returns tensors
        collate_fn=_collate_batch,
    )


def _collate_batch(batch: List[dict]) -> dict:
    """Collate batch of samples.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    input_ids = torch.stack([s["input_ids"] for s in batch])
    attention_mask = torch.stack([s["attention_mask"] for s in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
