"""Mixed dataset utilities for CheaperTraining influence-based data selection.

NOTE: MixedPretrainDataset has been consolidated into PretrainDataset.
Use create_mixed_dataloader() from pretrain_dataset.py for multi-source datasets.
"""

import logging
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

# Re-export from pretrain_dataset for backward compatibility
from wrinklefree.data._legacy.pretrain_dataset import (
    PretrainDataset,
    create_mixed_dataloader,
)

logger = logging.getLogger(__name__)

# Backward compatibility alias
MixedPretrainDataset = PretrainDataset


def create_probe_dataloader(
    path: str,
    tokenizer: PreTrainedTokenizer,
    subset: Optional[str] = None,
    split: str = "train",
    size: int = 1000,
    max_length: int = 2048,
    text_column: str = "text",
) -> DataLoader:
    """
    Create probe dataset for influence calculation.

    Uses streaming to avoid downloading the full dataset, then materializes
    only the samples needed for the probe set.

    Args:
        path: HuggingFace dataset path
        tokenizer: Tokenizer
        subset: Dataset subset name
        split: Split to use
        size: Number of samples
        max_length: Max sequence length
        text_column: Column containing text

    Returns:
        DataLoader for probe set
    """
    logger.info(f"Loading probe dataset (streaming): {path} (size={size})")

    # Use streaming to avoid downloading full dataset
    streaming_dataset = load_dataset(
        path,
        name=subset,
        split=split,
        streaming=True,
    )

    # Take only the samples we need and materialize into a list
    probe_samples = []
    for i, example in enumerate(streaming_dataset):
        if i >= size:
            break
        text = example[text_column]

        # Tokenize
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        probe_samples.append({
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
        })

    logger.info(f"Loaded {len(probe_samples)} probe samples")

    # Create a simple list-based dataset
    class ProbeDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    return DataLoader(ProbeDataset(probe_samples), batch_size=4, shuffle=False)
