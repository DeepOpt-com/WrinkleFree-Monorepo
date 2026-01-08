"""Supervised Fine-Tuning (SFT) dataset with chat template support.

Loads instruction-following datasets and applies chat templates (e.g., Qwen)
with proper label masking so only assistant responses contribute to loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT dataset.

    Attributes:
        path: HuggingFace dataset path
        subset: Dataset subset/config name (e.g., "SFT")
        split: Dataset split (e.g., "train")
        input_column: Column containing conversation turns (list of dicts)
        output_column: Column containing final assistant response
        system_prompt_column: Column containing system prompt (optional)
        max_length: Maximum sequence length
        add_generation_prompt: Whether to add generation prompt after user turn
    """

    path: str = "nvidia/Llama-Nemotron-Post-Training-Dataset"
    subset: str = "SFT"
    split: str = "train"
    input_column: str = "input"
    output_column: str = "output"
    system_prompt_column: str = "system_prompt"
    max_length: int = 2048
    add_generation_prompt: bool = False


class SFTDataset(IterableDataset):
    """Streaming SFT dataset with chat template formatting.

    Loads conversations from HuggingFace datasets and formats them using
    the tokenizer's chat template. Labels are masked so only assistant
    responses contribute to the loss.

    The dataset expects data in the Nemotron format:
    - input: List of {"role": "user/assistant", "content": "..."}
    - output: Final assistant response
    - system_prompt: Optional system prompt

    Args:
        config: SFT configuration
        tokenizer: HuggingFace tokenizer with chat template
        streaming: Whether to use streaming mode
        seed: Random seed for shuffling
        rank: Process rank for distributed training
        world_size: Total number of processes
        shuffle_buffer_size: Size of shuffle buffer for streaming
    """

    def __init__(
        self,
        config: SFTConfig,
        tokenizer: Any,
        streaming: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        shuffle_buffer_size: int = 1000,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.streaming = streaming
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shuffle_buffer_size = shuffle_buffer_size

        # Ensure tokenizer has required attributes
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.warning(f"Setting pad_token_id to eos_token_id ({tokenizer.eos_token_id})")

        self._dataset = None

    def _load_dataset(self):
        """Lazy-load the HuggingFace dataset."""
        if self._dataset is not None:
            return

        from datasets import load_dataset

        logger.info(
            f"Loading SFT dataset: {self.config.path} "
            f"(subset={self.config.subset}, split={self.config.split})"
        )

        self._dataset = load_dataset(
            self.config.path,
            self.config.subset,
            split=self.config.split,
            streaming=self.streaming,
        )

        # Apply distributed sharding
        if self.world_size > 1:
            from datasets.distributed import split_dataset_by_node

            self._dataset = split_dataset_by_node(
                self._dataset, rank=self.rank, world_size=self.world_size
            )

        # Shuffle
        self._dataset = self._dataset.shuffle(
            seed=self.seed + self.rank,
            buffer_size=self.shuffle_buffer_size,
        )

    def _format_conversation(self, example: dict) -> list[dict]:
        """Format example into chat messages.

        Converts Nemotron format to standard chat format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        """
        messages = []

        # Add system prompt if present
        system_prompt = example.get(self.config.system_prompt_column)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation turns from input
        input_turns = example.get(self.config.input_column, [])
        if isinstance(input_turns, list):
            for turn in input_turns:
                if isinstance(turn, dict) and "role" in turn and "content" in turn:
                    messages.append({"role": turn["role"], "content": turn["content"]})

        # Add final assistant response from output
        output = example.get(self.config.output_column)
        if output:
            messages.append({"role": "assistant", "content": output})

        return messages

    def _tokenize_with_labels(self, messages: list[dict]) -> dict[str, torch.Tensor]:
        """Tokenize messages and create labels with instruction masking.

        Uses the tokenizer's chat template to format the conversation,
        then creates labels where:
        - Instruction tokens (system + user) are masked with -100
        - Assistant response tokens have valid labels

        This is done by:
        1. Tokenizing the full conversation
        2. Tokenizing just the prompt (without final assistant response)
        3. Masking labels for the prompt portion
        """
        # Full conversation (prompt + response)
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize full conversation
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            add_special_tokens=False,  # Chat template handles special tokens
        )

        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        # Create labels (copy of input_ids initially)
        labels = input_ids.clone()

        # Find where the assistant response starts by tokenizing just the prompt
        if len(messages) > 0 and messages[-1]["role"] == "assistant":
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,  # Include the assistant prefix
            )

            prompt_tokens = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
                add_special_tokens=False,
            )

            prompt_len = prompt_tokens["input_ids"].size(1)

            # Mask everything before the response (set to -100)
            labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over tokenized examples."""
        self._load_dataset()

        for example in self._dataset:
            try:
                # Format conversation
                messages = self._format_conversation(example)

                if not messages:
                    continue

                # Ensure there's at least one assistant response
                has_assistant = any(m["role"] == "assistant" for m in messages)
                if not has_assistant:
                    continue

                # Tokenize with label masking
                tokenized = self._tokenize_with_labels(messages)

                # Skip if too short (only padding)
                if tokenized["input_ids"].sum() == 0:
                    continue

                yield tokenized

            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                continue


class PackedSFTDataset(IterableDataset):
    """SFT dataset with sequence packing for efficient training.

    Packs multiple SFT examples into sequences of max_length, using
    attention mask to separate examples. Labels are properly aligned
    to maintain the instruction masking.
    """

    def __init__(
        self,
        sft_dataset: SFTDataset,
        max_length: int = 2048,
        pad_token_id: int = 0,
    ):
        self.sft_dataset = sft_dataset
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Pack examples into fixed-length sequences."""
        buffer_input_ids = []
        buffer_labels = []
        buffer_attention_mask = []
        current_length = 0

        for example in self.sft_dataset:
            input_ids = example["input_ids"]
            labels = example["labels"]
            attention_mask = example["attention_mask"]
            seq_len = input_ids.size(0)

            # If this example would overflow, yield current buffer
            if current_length + seq_len > self.max_length and buffer_input_ids:
                yield self._pad_and_return(
                    buffer_input_ids, buffer_labels, buffer_attention_mask
                )
                buffer_input_ids = []
                buffer_labels = []
                buffer_attention_mask = []
                current_length = 0

            # If single example is too long, truncate it
            if seq_len > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                seq_len = self.max_length

            buffer_input_ids.append(input_ids)
            buffer_labels.append(labels)
            buffer_attention_mask.append(attention_mask)
            current_length += seq_len

        # Yield remaining buffer
        if buffer_input_ids:
            yield self._pad_and_return(
                buffer_input_ids, buffer_labels, buffer_attention_mask
            )

    def _pad_and_return(
        self,
        input_ids_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
        attention_mask_list: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Concatenate and pad to max_length."""
        input_ids = torch.cat(input_ids_list)
        labels = torch.cat(labels_list)
        attention_mask = torch.cat(attention_mask_list)

        current_len = input_ids.size(0)
        pad_len = self.max_length - current_len

        if pad_len > 0:
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)]
            )
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)]
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_sft_dataloader(
    tokenizer: Any,
    batch_size: int,
    config: Optional[SFTConfig] = None,
    max_length: int = 2048,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    seed: int = 42,
    packed: bool = True,
) -> "torch.utils.data.DataLoader":
    """Create SFT dataloader with Nemotron dataset.

    Args:
        tokenizer: HuggingFace tokenizer with chat template (e.g., Qwen)
        batch_size: Batch size for training
        config: SFT configuration (defaults to Nemotron dataset)
        max_length: Maximum sequence length
        rank: Process rank for distributed training
        world_size: Total number of processes
        num_workers: Number of data loading workers
        seed: Random seed
        packed: Whether to use sequence packing

    Returns:
        DataLoader yielding batches with input_ids, attention_mask, labels
    """
    from torch.utils.data import DataLoader

    if config is None:
        config = SFTConfig(max_length=max_length)
    else:
        config.max_length = max_length

    # Create base SFT dataset
    sft_dataset = SFTDataset(
        config=config,
        tokenizer=tokenizer,
        streaming=True,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Optionally wrap with packing
    if packed:
        dataset = PackedSFTDataset(
            sft_dataset=sft_dataset,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
    else:
        dataset = sft_dataset

    # Create collate function
    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        """Stack batch of examples."""
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
