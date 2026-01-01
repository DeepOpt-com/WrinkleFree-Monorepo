"""Post-training SFT stage implementation.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) Section 3.3
- Stage 1: General SFT with instruction-following data
- Stage 2: Reasoning SFT with long chain-of-thought traces
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from cheapertraining._legacy.training.stages.base import TrainingStage, StageConfig


class PosttrainSFTStage(TrainingStage):
    """Post-training supervised fine-tuning stage.

    Implements SFT loss where we only compute loss on assistant/completion tokens,
    not on the prompt/user tokens.
    """

    def compute_loss(
        self,
        batch: dict[str, Tensor],
    ) -> Tuple[Tensor, dict[str, float]]:
        """Compute SFT loss (only on completion tokens).

        Args:
            batch: Dictionary with:
                - 'input_ids': Input token IDs
                - 'labels': Same as input_ids but with -100 for prompt tokens
                - 'attention_mask': Optional attention mask

        Returns:
            Tuple of (loss, metrics_dict)
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)  # If no labels, use input_ids
        attention_mask = batch.get("attention_mask")

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs["logits"]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Apply attention mask to labels if provided
        if attention_mask is not None:
            label_mask = attention_mask[:, 1:].contiguous()
            shift_labels = shift_labels.masked_fill(label_mask == 0, -100)

        # Compute cross-entropy loss (ignoring -100 positions)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        # Compute metrics
        with torch.no_grad():
            # Accuracy on completion tokens only
            valid_mask = shift_labels != -100
            predictions = shift_logits.argmax(dim=-1)
            correct = (predictions == shift_labels) & valid_mask
            accuracy = correct.sum().float() / valid_mask.sum().float() if valid_mask.any() else torch.tensor(0.0)

            # Perplexity
            perplexity = torch.exp(loss)

            # Count completion tokens
            num_completion_tokens = valid_mask.sum().item()

            # Calculate completion ratio (what fraction of sequence is completion)
            total_tokens = shift_labels.numel()
            completion_ratio = num_completion_tokens / total_tokens if total_tokens > 0 else 0.0

        metrics = {
            "accuracy": accuracy.item(),
            "perplexity": perplexity.item(),
            "num_completion_tokens": num_completion_tokens,
            "completion_ratio": completion_ratio,
        }

        return loss, metrics


class ReasoningSFTStage(PosttrainSFTStage):
    """Reasoning-focused SFT stage for long chain-of-thought.

    Same as PosttrainSFTStage but with additional metrics for reasoning.
    Designed for longer sequences (32k context).
    """

    def compute_loss(
        self,
        batch: dict[str, Tensor],
    ) -> Tuple[Tensor, dict[str, float]]:
        """Compute SFT loss with reasoning-specific metrics.

        Args:
            batch: Dictionary with input_ids, labels, attention_mask

        Returns:
            Tuple of (loss, metrics_dict)
        """
        loss, metrics = super().compute_loss(batch)

        # Add reasoning-specific metrics
        with torch.no_grad():
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)

            # Estimate reasoning length (tokens where labels != -100)
            reasoning_mask = labels[:, 1:] != -100
            avg_reasoning_length = reasoning_mask.sum(dim=1).float().mean().item()

            # Sequence length statistics
            seq_lengths = batch.get("attention_mask", torch.ones_like(input_ids)).sum(dim=1)
            avg_seq_length = seq_lengths.float().mean().item()
            max_seq_length = seq_lengths.max().item()

        metrics.update({
            "avg_reasoning_length": avg_reasoning_length,
            "avg_seq_length": avg_seq_length,
            "max_seq_length": max_seq_length,
        })

        return loss, metrics
