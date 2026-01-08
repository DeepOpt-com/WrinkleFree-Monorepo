"""Supervised Fine-Tuning (SFT) objective.

Cross-entropy loss on assistant responses only, with instruction tokens masked.
The data loader handles label masking - this objective simply computes the loss.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from wf_train.objectives.base import Objective, ObjectiveOutput


class SFTObjective(Objective):
    """Supervised Fine-Tuning objective.

    Computes cross-entropy loss on chat-formatted data where instruction
    tokens (system + user messages) are masked with ignore_index.
    Only assistant responses contribute to the loss.

    The data loader is responsible for:
    - Applying the chat template (e.g., Qwen format)
    - Setting labels to -100 for instruction tokens
    - Ensuring only assistant responses have valid labels

    Args:
        ignore_index: Index to ignore in loss computation (default: -100)
        label_smoothing: Label smoothing factor (default: 0.0)
    """

    requires_teacher = False
    requires_hidden_states = False
    modifies_input = False

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    @property
    def name(self) -> str:
        return "sft"

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """Compute SFT cross-entropy loss.

        Args:
            model_outputs: Must contain 'logits' (batch, seq, vocab)
            batch: Must contain 'labels' (batch, seq) with -100 for masked tokens
            teacher_outputs: Not used

        Returns:
            ObjectiveOutput with loss, perplexity, and response token metrics
        """
        logits = model_outputs["logits"]
        # For SFT, labels already have -100 for instruction tokens
        # Unlike continue_pretrain, we don't check _original_labels
        labels = batch["labels"]

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute cross-entropy loss (only on valid response tokens)
        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Compute metrics
        with torch.no_grad():
            perplexity = torch.exp(loss.detach())

            # Count response tokens (non-masked labels)
            valid_mask = shift_labels != self.ignore_index
            response_tokens = valid_mask.sum()
            total_tokens = shift_labels.numel()
            response_ratio = response_tokens.float() / total_tokens

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "perplexity": perplexity,
                "response_tokens": response_tokens,
                "response_ratio": response_ratio,
            },
            ce_loss=loss.detach(),
        )
