"""Shared sequence packing utilities for pre-training datasets.

These utilities handle:
- RoPE-aware position ID computation (reset at document boundaries)
- Batched tokenization for performance
- Sequence packing into fixed-length chunks

Used by both CheaperTraining and WrinkleFree-1.58Quant.
"""

import collections
from typing import Any, Iterator, List

import torch
from torch import Tensor


def compute_position_ids(input_ids: Tensor, separator_token_id: int) -> Tensor:
    """Compute position IDs that reset at each document boundary.

    This enables proper RoPE positioning for FlashAttention-based sequence
    packing, where each document should have positions starting from 0.

    Reference: https://huggingface.co/blog/sirluk/llm-sequence-packing

    Args:
        input_ids: Token IDs for the packed sequence [seq_len]
        separator_token_id: Token ID that marks document boundaries (typically EOS)

    Returns:
        Position IDs that reset after each separator token [seq_len]
    """
    position_ids = torch.zeros_like(input_ids)
    pos = 0
    for i, token_id in enumerate(input_ids):
        position_ids[i] = pos
        if token_id == separator_token_id:
            pos = 0  # Reset position after separator
        else:
            pos += 1
    return position_ids


def compute_position_ids_vectorized(input_ids: Tensor, separator_token_id: int) -> Tensor:
    """Truly vectorized position ID computation (no Python loops).

    Uses segment-based approach:
    1. Find segment starts (pos 0 + positions after separators)
    2. Assign segment IDs via cumsum
    3. Map each position to its segment's start
    4. Position within segment = global_pos - segment_start

    Args:
        input_ids: Token IDs for the packed sequence [seq_len]
        separator_token_id: Token ID that marks document boundaries

    Returns:
        Position IDs that reset after each separator token [seq_len]
    """
    seq_len = input_ids.size(0)
    if seq_len == 0:
        return torch.empty(0, dtype=torch.long, device=input_ids.device)

    device = input_ids.device

    # Step 1: Find segment starts (pos 0 + positions after separators)
    is_separator = input_ids == separator_token_id
    is_seg_start = torch.zeros(seq_len, dtype=torch.bool, device=device)
    is_seg_start[0] = True
    if seq_len > 1:
        is_seg_start[1:] = is_separator[:-1]  # Position after each separator

    # Step 2: Assign segment IDs (cumsum of starts, minus 1 to be 0-indexed)
    segment_ids = torch.cumsum(is_seg_start.long(), dim=0) - 1

    # Step 3: Find the start position of each segment
    seg_start_indices = is_seg_start.nonzero(as_tuple=True)[0]

    # Step 4: Map each position to its segment's start position
    segment_starts = seg_start_indices[segment_ids]

    # Step 5: Position within segment = global position - segment start
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = positions - segment_starts

    return position_ids


def batch_tokenize(
    texts: List[str],
    tokenizer: Any,
    add_special_tokens: bool = False,
) -> List[List[int]]:
    """Batch tokenize texts for better performance.

    Uses HuggingFace's batch encoding which is significantly faster
    than tokenizing documents one at a time.

    Args:
        texts: List of text strings to tokenize
        tokenizer: HuggingFace tokenizer
        add_special_tokens: Whether to add special tokens (BOS/EOS)

    Returns:
        List of token ID lists, one per input text
    """
    if not texts:
        return []

    encoded = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    return encoded["input_ids"]


def pack_token_buffer(
    token_buffer: collections.deque,
    max_length: int,
    separator_token_id: int,
    include_labels: bool = True,
) -> Iterator[dict[str, Tensor]]:
    """Pack tokens from buffer into fixed-length sequences.

    Yields complete sequences of exactly max_length tokens.
    Generates position_ids that reset at document boundaries.

    Args:
        token_buffer: Deque of token IDs to pack
        max_length: Target sequence length
        separator_token_id: Token ID for document boundaries
        include_labels: Whether to include labels in output

    Yields:
        Dicts with input_ids, attention_mask, position_ids, and optionally labels
    """
    while len(token_buffer) >= max_length:
        chunk = [token_buffer.popleft() for _ in range(max_length)]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        position_ids = compute_position_ids(input_ids, separator_token_id)

        result = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(max_length, dtype=torch.long),
            "position_ids": position_ids,
        }

        if include_labels:
            result["labels"] = input_ids.clone()

        yield result


class TokenPacker:
    """Utility class for packing tokenized documents into fixed-length sequences.

    Handles batched tokenization and packing with proper position IDs.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 2048,
        separator_token_id: int | None = None,
        tokenize_batch_size: int = 256,
        include_labels: bool = True,
    ):
        """Initialize token packer.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Target sequence length
            separator_token_id: Token ID for document boundaries (default: EOS)
            tokenize_batch_size: Number of texts to batch tokenize at once
            include_labels: Whether to include labels in output
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.separator_token_id = separator_token_id or tokenizer.eos_token_id
        self.tokenize_batch_size = tokenize_batch_size
        self.include_labels = include_labels

        # Internal buffers
        self._token_buffer: collections.deque = collections.deque()
        self._text_buffer: List[str] = []

    def reset(self):
        """Reset internal buffers."""
        self._token_buffer.clear()
        self._text_buffer.clear()

    def add_text(self, text: str) -> Iterator[dict[str, Tensor]]:
        """Add a text document to the packer.

        Args:
            text: Text document to add

        Yields:
            Complete packed sequences when buffer is full
        """
        self._text_buffer.append(text)

        # Batch tokenize when buffer is full
        if len(self._text_buffer) >= self.tokenize_batch_size:
            yield from self._flush_text_buffer()

    def _flush_text_buffer(self) -> Iterator[dict[str, Tensor]]:
        """Tokenize buffered texts and add to token buffer."""
        if not self._text_buffer:
            return

        token_batches = batch_tokenize(
            self._text_buffer,
            self.tokenizer,
            add_special_tokens=False,
        )

        for tokens in token_batches:
            self._token_buffer.extend(tokens)
            self._token_buffer.append(self.separator_token_id)

        self._text_buffer.clear()

        # Yield complete sequences
        yield from pack_token_buffer(
            self._token_buffer,
            self.max_length,
            self.separator_token_id,
            self.include_labels,
        )

    def flush(self) -> Iterator[dict[str, Tensor]]:
        """Flush remaining texts and yield complete sequences.

        Call this at the end of iteration to process remaining documents.

        Yields:
            Complete packed sequences from remaining buffer
        """
        yield from self._flush_text_buffer()

        # Yield any remaining complete sequences
        yield from pack_token_buffer(
            self._token_buffer,
            self.max_length,
            self.separator_token_id,
            self.include_labels,
        )

    def get_remaining_tokens(self) -> int:
        """Get count of tokens remaining in buffer (not yet yielded)."""
        return len(self._token_buffer)
