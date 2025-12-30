"""SFT dataset implementations.

Provides datasets for supervised fine-tuning with chat templates.
"""

from typing import Iterator, Optional, Any, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader


class SFTDataset(Dataset):
    """Supervised fine-tuning dataset with chat template support.

    Handles instruction-response pairs and creates labels that mask the prompt.
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: Any,
        max_length: int = 4096,
        prompt_column: str = "prompt",
        response_column: str = "response",
        messages_column: Optional[str] = "messages",
    ):
        """Initialize SFT dataset.

        Args:
            samples: List of samples (can be instruction-response or messages format)
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            prompt_column: Column name for prompts
            response_column: Column name for responses
            messages_column: Column name for chat messages (takes precedence)
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.messages_column = messages_column

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        sample = self.samples[idx]

        # Check for messages format first
        if self.messages_column and self.messages_column in sample:
            return self._process_messages(sample[self.messages_column])
        else:
            return self._process_instruction_response(
                sample.get(self.prompt_column, ""),
                sample.get(self.response_column, ""),
            )

    def _process_messages(self, messages: List[Dict]) -> dict:
        """Process chat messages format.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Processed sample
        """
        # Find where assistant responses start for label masking
        prompt_messages = []
        response_start_idx = -1

        for i, msg in enumerate(messages):
            if msg["role"] == "assistant" and response_start_idx == -1:
                response_start_idx = i
                break
            prompt_messages.append(msg)

        # Tokenize full conversation
        if hasattr(self.tokenizer, "apply_chat_template"):
            full_text = self.tokenizer.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = self.tokenizer.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback formatting
            full_text = self._format_messages(messages)
            prompt_text = self._format_messages(prompt_messages, add_generation=True)

        # Tokenize
        full_tokens = self.tokenizer.encode(
            full_text,
            max_length=self.max_length,
            truncation=True,
        )
        prompt_tokens = self.tokenizer.encode(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
        )

        # Create labels with masked prompt
        labels = full_tokens.copy()
        labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)

        # Pad to max_length
        padding_length = self.max_length - len(full_tokens)
        if padding_length > 0:
            full_tokens = full_tokens + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
        else:
            attention_mask = [1] * len(full_tokens)

        return {
            "input_ids": torch.tensor(full_tokens[:self.max_length], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:self.max_length], dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_length], dtype=torch.long),
        }

    def _process_instruction_response(self, prompt: str, response: str) -> dict:
        """Process instruction-response format.

        Args:
            prompt: Instruction/prompt text
            response: Response/completion text

        Returns:
            Processed sample
        """
        # Format as simple prompt-response
        full_text = f"{prompt}\n\n{response}"

        # Tokenize
        prompt_tokens = self.tokenizer.encode(
            f"{prompt}\n\n",
            add_special_tokens=True,
        )
        full_tokens = self.tokenizer.encode(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )

        # Create labels with masked prompt
        labels = full_tokens.copy()
        prompt_len = min(len(prompt_tokens), len(full_tokens))
        labels[:prompt_len] = [-100] * prompt_len

        # Pad to max_length
        padding_length = self.max_length - len(full_tokens)
        if padding_length > 0:
            full_tokens = full_tokens + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = [1] * (self.max_length - padding_length) + [0] * padding_length
        else:
            attention_mask = [1] * len(full_tokens)

        return {
            "input_ids": torch.tensor(full_tokens[:self.max_length], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:self.max_length], dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_length], dtype=torch.long),
        }

    def _format_messages(self, messages: List[Dict], add_generation: bool = False) -> str:
        """Format messages as text (fallback).

        Args:
            messages: List of message dicts
            add_generation: Whether to add generation prompt

        Returns:
            Formatted text
        """
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{role}|>\n{content}\n"

        if add_generation:
            text += "<|assistant|>\n"

        return text


def create_sft_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for SFT.

    Args:
        dataset: SFT dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
