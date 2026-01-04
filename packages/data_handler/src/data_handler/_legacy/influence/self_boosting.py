"""Self-boosting filter for Phase III mid-training.

Implements rejection sampling based on influence scores.
Samples with negative influence on the probe set are rejected.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) Phase III
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from math_utils.influence.config import InfluenceConfig, SelfBoostingConfig
from math_utils.influence.datainf import DataInfCalculator, create_influence_calculator


class SelfBoostingFilter:
    """Mid-training self-boosting data filter.

    Uses the current model's influence scores to filter training data,
    keeping only samples with positive influence on the probe set.

    This creates a "self-purifying" cycle where the dataset shrinks
    as the model gets smarter, removing data that would hurt performance.

    Reference: MobileLLM-R1 Phase III - Mid-Training Self-Boosting
    """

    def __init__(
        self,
        model: Any,
        probe_dataloader: DataLoader,
        config: Optional[SelfBoostingConfig] = None,
        influence_config: Optional[InfluenceConfig] = None,
    ):
        """Initialize self-boosting filter.

        Args:
            model: MobileLLM model
            probe_dataloader: DataLoader for probe set
            config: Self-boosting configuration
            influence_config: Influence calculation configuration
        """
        self.model = model
        self.probe_dataloader = probe_dataloader
        self.config = config or SelfBoostingConfig()
        self.influence_config = influence_config or InfluenceConfig()

        # Initialize influence calculator
        self.influence_calculator = create_influence_calculator(
            model, self.influence_config
        )

        # State tracking
        self._current_stage = 0
        self._last_update_step = 0
        self._probe_cached = False

        # Statistics
        self._total_samples_seen = 0
        self._total_samples_kept = 0

    def initialize_probe_set(self, show_progress: bool = True):
        """Initialize or reinitialize probe set gradients.

        Call this at the start and periodically during training
        to keep probe gradients aligned with model state.
        """
        self.influence_calculator.cache_probe_gradients(
            self.probe_dataloader,
            show_progress=show_progress,
        )
        self._probe_cached = True

    def update_model(self, model: Any):
        """Update the model reference.

        Call this when the model has been updated and you want
        to refresh the influence calculations.

        Args:
            model: Updated model
        """
        self.model = model
        # Recreate calculator with new model
        self.influence_calculator = create_influence_calculator(
            model, self.influence_config
        )
        self._probe_cached = False

    def refresh(self, current_step: int, show_progress: bool = False):
        """Refresh probe cache if recompute interval has elapsed.

        Args:
            current_step: Current training step
            show_progress: Whether to show progress

        Returns:
            True if cache was refreshed
        """
        if self.should_update(current_step):
            self.influence_calculator.clear_cache()
            self.initialize_probe_set(show_progress=show_progress)
            self._last_update_step = current_step
            return True
        return False

    def should_update(self, current_step: int) -> bool:
        """Check if influence scores should be recomputed.

        Args:
            current_step: Current training step

        Returns:
            True if recomputation is needed
        """
        return (current_step - self._last_update_step) >= self.config.recompute_interval

    def filter_batch(
        self,
        batch: Dict[str, Tensor],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """Filter a batch based on influence scores.

        Keeps samples with influence > threshold.

        Args:
            batch: Input batch dictionary

        Returns:
            Tuple of (filtered_batch, influence_scores)
        """
        if not self._probe_cached:
            self.initialize_probe_set(show_progress=False)

        device = next(self.model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}

        # Compute influence scores
        influence_scores = self.influence_calculator.compute_batch_influence_aggregated(batch)

        # Determine which samples to keep
        keep_mask = influence_scores > self.config.influence_threshold

        # Update statistics
        self._total_samples_seen += influence_scores.size(0)
        self._total_samples_kept += keep_mask.sum().item()

        # Ensure we keep at least min_batch_size samples
        if keep_mask.sum() < self.config.min_batch_size:
            # Keep top-k by influence score
            top_k = min(self.config.min_batch_size, influence_scores.size(0))
            _, top_indices = torch.topk(influence_scores, top_k)
            keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
            keep_mask[top_indices] = True

        # Filter batch
        filtered_batch = {}
        for key, value in batch.items():
            if isinstance(value, Tensor):
                filtered_batch[key] = value[keep_mask]
            else:
                filtered_batch[key] = value

        return filtered_batch, influence_scores

    def get_rejection_rate(self) -> float:
        """Get current rejection rate.

        Returns:
            Fraction of samples rejected (0-1)
        """
        if self._total_samples_seen == 0:
            return 0.0
        return 1.0 - (self._total_samples_kept / self._total_samples_seen)

    def reset_statistics(self):
        """Reset rejection statistics."""
        self._total_samples_seen = 0
        self._total_samples_kept = 0


class SelfBoostingDataset(IterableDataset):
    """Wrapper dataset that applies self-boosting filtering.

    Wraps a source dataset and filters samples based on influence scores.
    """

    def __init__(
        self,
        source_dataset: IterableDataset,
        filter: SelfBoostingFilter,
        tokenizer: Any = None,
        max_length: int = 2048,
    ):
        """Initialize self-boosting dataset.

        Args:
            source_dataset: Source dataset to filter
            filter: SelfBoostingFilter instance
            tokenizer: Tokenizer for encoding (if source yields text)
            max_length: Maximum sequence length
        """
        self.source_dataset = source_dataset
        self.filter = filter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = filter.config.buffer_size

    def __iter__(self) -> Iterator[dict]:
        """Iterate with rejection sampling.

        Accumulates samples into buffer, filters by influence,
        and yields only high-influence samples.
        """
        buffer: List[dict] = []

        for sample in self.source_dataset:
            # Ensure sample is tokenized
            if "input_ids" not in sample and self.tokenizer is not None:
                sample = self._tokenize_sample(sample)

            buffer.append(sample)

            # Process buffer when full
            if len(buffer) >= self.buffer_size:
                yield from self._process_buffer(buffer)
                buffer = []

        # Process remaining buffer
        if buffer:
            yield from self._process_buffer(buffer)

    def _tokenize_sample(self, sample: dict) -> dict:
        """Tokenize a text sample.

        Args:
            sample: Sample with 'text' field

        Returns:
            Sample with added 'input_ids', 'attention_mask', 'labels'
        """
        text = sample.get("text", "")
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        attention_mask = [1] * len(tokens)
        padding_length = self.max_length - len(tokens)
        if padding_length > 0:
            tokens = tokens + [self.tokenizer.pad_token_id or 0] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        sample["input_ids"] = torch.tensor(tokens, dtype=torch.long)
        sample["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        sample["labels"] = sample["input_ids"].clone()

        return sample

    def _process_buffer(self, buffer: List[dict]) -> Iterator[dict]:
        """Process buffer through filter.

        Args:
            buffer: List of samples

        Yields:
            Filtered samples
        """
        if not buffer:
            return

        # Collate buffer into batch
        batch = self._collate_buffer(buffer)

        # Filter
        filtered_batch, scores = self.filter.filter_batch(batch)

        # Yield individual filtered samples
        batch_size = filtered_batch["input_ids"].size(0)
        for i in range(batch_size):
            yield {
                "input_ids": filtered_batch["input_ids"][i],
                "attention_mask": filtered_batch["attention_mask"][i],
                "labels": filtered_batch.get("labels", filtered_batch["input_ids"])[i],
            }

    def _collate_buffer(self, buffer: List[dict]) -> Dict[str, Tensor]:
        """Collate buffer samples into a batch.

        Args:
            buffer: List of sample dictionaries

        Returns:
            Batched tensor dictionary
        """
        input_ids = torch.stack([s["input_ids"] for s in buffer])
        attention_mask = torch.stack([s["attention_mask"] for s in buffer])
        labels = torch.stack([s.get("labels", s["input_ids"]) for s in buffer])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_self_boosting_filter(
    model: Any,
    probe_dataloader: DataLoader,
    config: Optional[SelfBoostingConfig] = None,
) -> SelfBoostingFilter:
    """Factory function to create a self-boosting filter.

    Args:
        model: MobileLLM model
        probe_dataloader: Probe set DataLoader
        config: Configuration

    Returns:
        Configured SelfBoostingFilter
    """
    return SelfBoostingFilter(model, probe_dataloader, config)
