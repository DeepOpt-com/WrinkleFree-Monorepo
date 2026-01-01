"""DLM (Diffusion Language Model) objective.

Implements block-wise masked language modeling (Fast-dLLM) where tokens
are masked and reconstructed. This objective modifies the input batch
in-place to apply masking.

Reference: Fast-dLLM v2 (https://arxiv.org/abs/2509.26328)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from wrinklefree.objectives.base import Objective, ObjectiveOutput


class DLMObjective(Objective):
    """Block-wise masked language modeling objective (Fast-dLLM).

    This objective applies random masking to input tokens and trains
    the model to reconstruct the masked tokens. Unlike CLM which shifts
    labels by 1, DLM predicts the masked token at the same position.

    Args:
        mask_token_id: Token ID to use for masking
        mask_prob: Probability of masking each token (default: 0.15)
        ignore_index: Label value to ignore in loss (default: -100)
    """

    requires_teacher = False
    requires_hidden_states = False
    modifies_input = True

    def __init__(
        self,
        mask_token_id: int,
        mask_prob: float = 0.15,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index

    @property
    def name(self) -> str:
        return "dlm"

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply random masking to inputs.

        Creates masked_input_ids and dlm_labels in the batch.
        The model will receive masked_input_ids as input and
        dlm_labels as the target for the masked positions.
        """
        input_ids = batch["input_ids"]
        device = input_ids.device

        # Initialize dlm_labels with ignore_index
        dlm_labels = torch.full_like(input_ids, self.ignore_index)

        # Clone input_ids for masking
        masked_input_ids = input_ids.clone()

        # Create random mask
        mask = torch.rand(input_ids.shape, device=device) < self.mask_prob

        # Don't mask first token (BOS) or last token (EOS)
        mask[:, 0] = False
        mask[:, -1] = False

        # Don't mask padding tokens
        if "attention_mask" in batch:
            mask = mask & batch["attention_mask"].bool()

        # Apply masking: replace masked tokens with mask_token_id
        masked_input_ids[mask] = self.mask_token_id

        # Set labels: only care about masked positions
        dlm_labels[mask] = input_ids[mask]

        # Update batch
        batch["input_ids"] = masked_input_ids
        batch["dlm_labels"] = dlm_labels
        batch["_original_input_ids"] = input_ids  # Keep original for debugging
        batch["_original_labels"] = batch.get("labels", input_ids).clone()  # For multi-task

        return batch

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """Compute DLM loss (reconstruction of masked tokens).

        Unlike CLM which predicts next token (shifted by 1),
        DLM predicts the masked token at the same position.
        """
        logits = model_outputs["logits"]
        labels = batch.get("dlm_labels")

        if labels is None:
            raise ValueError(
                "dlm_labels not found in batch. "
                "Did preprocess_batch run? Set modifies_input=True."
            )

        # DLM predicts the masked token at position i (no shift needed)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.ignore_index,
        )

        # Count masked tokens for logging
        num_masked = (labels != self.ignore_index).sum().float()
        total_tokens = labels.numel()

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "num_masked": num_masked,
                "mask_ratio": num_masked / total_tokens,
            },
        )

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, mask_token_id={self.mask_token_id}, "
            f"mask_prob={self.mask_prob}"
        )
