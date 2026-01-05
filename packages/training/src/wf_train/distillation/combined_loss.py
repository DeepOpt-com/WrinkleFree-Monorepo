"""Loss functions for continue pre-training (Stage 2).

NOTE: BitDistillLoss and ClassificationDistillLoss have been moved to the
separate `distillation` package. This file now only contains ContinuePretrainLoss
for Stage 2 continue pre-training.

For Stage 3 distillation, use:
    from distillation import BitDistillLoss
"""

import torch
import torch.nn as nn


class ContinuePretrainLoss(nn.Module):
    """
    Loss function for Stage 2 continue pre-training.

    Simple cross-entropy loss on next token prediction without distillation.

    Args:
        ignore_index: Index to ignore in loss computation
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute language modeling loss.

        Args:
            logits: Model logits (batch, seq, vocab)
            labels: Ground truth labels (batch, seq)

        Returns:
            Dictionary with loss value
        """
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return {"loss": loss}
