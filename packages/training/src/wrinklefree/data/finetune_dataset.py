"""Fine-tuning dataset for Stage 3 distillation."""

import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class FinetuneDataset(Dataset):
    """
    Dataset for fine-tuning on downstream tasks.

    Handles various task formats:
    - Classification: single text + label
    - Text pair classification: text_a + text_b + label
    - Generation: input text + target text

    Args:
        dataset_path: HuggingFace dataset path
        tokenizer: Tokenizer for text encoding
        max_length: Maximum sequence length
        dataset_name: Dataset configuration name
        split: Dataset split
        text_column: Column containing primary text
        text_pair_column: Column containing secondary text (for pair tasks)
        label_column: Column containing labels
        is_generation: Whether this is a generation task
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        dataset_name: Optional[str] = None,
        split: str = "train",
        text_column: str = "sentence",
        text_pair_column: Optional[str] = None,
        label_column: str = "label",
        is_generation: bool = False,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.text_pair_column = text_pair_column
        self.label_column = label_column
        self.is_generation = is_generation

        # Load dataset
        self.dataset = load_dataset(
            dataset_path,
            dataset_name,
            split=split,
        )

        logger.info(f"Loaded dataset: {dataset_path}/{dataset_name or ''} ({split})")
        logger.info(f"  Examples: {len(self.dataset)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.dataset[idx]

        # Get text
        text = example[self.text_column]

        # Handle text pairs
        if self.text_pair_column and self.text_pair_column in example:
            text_pair = example[self.text_pair_column]
            encoding = self.tokenizer(
                text,
                text_pair,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # Squeeze batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Handle labels
        if self.is_generation:
            # For generation tasks, labels are the same as input_ids
            labels = input_ids.clone()
            # Mask padding
            labels[attention_mask == 0] = -100
            result["labels"] = labels
        else:
            # For classification tasks
            if self.label_column in example:
                result["labels"] = torch.tensor(example[self.label_column], dtype=torch.long)

        return result


class InstructDataset(Dataset):
    """
    Dataset for instruction-following fine-tuning.

    Formats data as instruction-response pairs for causal LM training.

    Args:
        dataset_path: HuggingFace dataset path
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        dataset_name: Dataset config name
        split: Dataset split
        instruction_column: Column with instructions
        response_column: Column with responses
        instruction_template: Template for formatting instructions
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        dataset_name: Optional[str] = None,
        split: str = "train",
        instruction_column: str = "instruction",
        response_column: str = "response",
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}",
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.instruction_template = instruction_template

        self.dataset = load_dataset(
            dataset_path,
            dataset_name,
            split=split,
        )

        logger.info(f"Loaded instruction dataset: {len(self.dataset)} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.dataset[idx]

        instruction = example[self.instruction_column]
        response = example[self.response_column]

        # Format text
        text = self.instruction_template.format(
            instruction=instruction,
            response=response,
        )

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels (mask instruction part)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Find where response starts and mask everything before
        instruction_only = self.instruction_template.split("{response}")[0].format(
            instruction=instruction
        )
        instruction_tokens = self.tokenizer.encode(
            instruction_only, add_special_tokens=False
        )
        instruction_len = len(instruction_tokens)

        # Mask instruction tokens
        labels[:instruction_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_finetune_dataloader(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 512,
    dataset_name: Optional[str] = None,
    split: str = "train",
    text_column: str = "sentence",
    label_column: str = "label",
    num_workers: int = 2,
    shuffle: bool = True,
    is_generation: bool = False,
) -> DataLoader:
    """
    Create dataloader for fine-tuning.

    Args:
        dataset_path: HuggingFace dataset path
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        dataset_name: Dataset config name
        split: Dataset split
        text_column: Text column name
        label_column: Label column name
        num_workers: Number of workers
        shuffle: Whether to shuffle
        is_generation: Whether it's a generation task

    Returns:
        DataLoader for fine-tuning
    """
    dataset = FinetuneDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_length,
        dataset_name=dataset_name,
        split=split,
        text_column=text_column,
        label_column=label_column,
        is_generation=is_generation,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
