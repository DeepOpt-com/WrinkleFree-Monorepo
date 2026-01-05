"""Continue pre-training objective (Stage 2).

Simple cross-entropy language modeling loss on next token prediction.
This is the primary objective for continue pre-training BitNet models.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from wf_train.objectives.base import Objective, ObjectiveOutput


class ContinuePretrainObjective(Objective):
    """Cross-entropy language modeling objective.

    Computes standard next-token prediction loss. This is the core
    objective for continue pre-training.

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
        return "continue_pretrain"

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """Compute cross-entropy language modeling loss.

        Args:
            model_outputs: Must contain 'logits' (batch, seq, vocab)
            batch: Must contain 'labels' (batch, seq)
            teacher_outputs: Not used

        Returns:
            ObjectiveOutput with loss and perplexity metric

        Note:
            When used with DLMObjective (multi-task), the model sees masked
            input_ids but this objective uses the original labels. This trains
            the model to predict next tokens even with masked context.
        """
        logits = model_outputs["logits"]
        # Use original labels even when DLM has masked the input
        labels = batch.get("_original_labels", batch["labels"])

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute cross-entropy loss
        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Compute perplexity for logging
        with torch.no_grad():
            perplexity = torch.exp(loss.detach())

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "perplexity": perplexity,
            },
            ce_loss=loss.detach(),  # For train/loss_unweighted_ce
        )
