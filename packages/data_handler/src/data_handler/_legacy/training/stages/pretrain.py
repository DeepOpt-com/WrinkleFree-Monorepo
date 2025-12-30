"""Pretraining stage implementation.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) Section 3.1
- Phase 1: Diverse pretraining with code, math, and web data
- Phase 2: Increased math/code emphasis
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from data_handler._legacy.training.stages.base import TrainingStage, StageConfig


class PretrainStage(TrainingStage):
    """Pretraining stage for language modeling.

    Implements standard causal language modeling loss (next-token prediction).
    """

    def compute_loss(
        self,
        batch: dict[str, Tensor],
    ) -> Tuple[Tensor, dict[str, float]]:
        """Compute language modeling loss.

        Args:
            batch: Dictionary with 'input_ids' and optionally 'attention_mask'

        Returns:
            Tuple of (loss, metrics_dict)
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs["logits"]

        # Shift for next-token prediction
        # logits: (batch, seq, vocab) -> (batch, seq-1, vocab)
        # labels: (batch, seq) -> (batch, seq-1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Create label mask (don't compute loss on padding)
        if attention_mask is not None:
            label_mask = attention_mask[:, 1:].contiguous()
            # Set padding positions to ignore_index
            shift_labels = shift_labels.masked_fill(label_mask == 0, -100)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # Compute metrics
        with torch.no_grad():
            # Token accuracy (non-padding tokens only)
            predictions = shift_logits.argmax(dim=-1)
            valid_mask = shift_labels != -100
            correct = (predictions == shift_labels) & valid_mask
            accuracy = correct.sum().float() / valid_mask.sum().float()

            # Perplexity
            perplexity = torch.exp(loss)

            # Count valid tokens for throughput calculation
            num_tokens = valid_mask.sum().item()

        metrics = {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
            "num_tokens": num_tokens,
        }

        return loss, metrics
