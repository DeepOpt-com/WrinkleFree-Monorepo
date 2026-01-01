"""Combined BitDistill objective (convenience wrapper).

Based on BitDistill: arxiv.org/abs/2510.13998
L = lambda_logits * L_LD + gamma_attention * L_AD

Note: CE loss should be handled by ContinuePretrainObjective separately.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from wrinklefree.objectives.base import Objective, ObjectiveOutput
from wrinklefree.objectives.logits_distill import LogitsDistillationObjective
from wrinklefree.objectives.attention_distill import AttentionRelationDistillationObjective


class BitDistillObjective(Objective):
    """
    Combined BitDistill objective (Equation 13).

    L = lambda_logits * L_LD + gamma_attention * L_AD

    This is a convenience wrapper that combines:
    - LogitsDistillationObjective (KL divergence on logits)
    - AttentionRelationDistillationObjective (A·A^T relation distillation)

    Note: CE loss should be handled by ContinuePretrainObjective separately.

    Args:
        lambda_logits: Weight for logits distillation
        gamma_attention: Weight for attention distillation (0 = disabled)
        temperature: Temperature for KL divergence
        distill_layer: Layer for attention distillation (-1 = last)
        ignore_index: Index to ignore in labels
    """

    requires_teacher = True
    requires_hidden_states = False
    modifies_input = False

    def __init__(
        self,
        lambda_logits: float = 10.0,
        gamma_attention: float = 1e-5,
        temperature: float = 5.0,
        distill_layer: int = -1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.lambda_logits = lambda_logits
        self.gamma_attention = gamma_attention

        self.logits_obj = LogitsDistillationObjective(
            temperature=temperature,
            ignore_index=ignore_index,
            shift_labels=True,  # AR model
        )

        self.attention_obj = AttentionRelationDistillationObjective(
            distill_layer=distill_layer,
            temperature=1.0,
            ignore_index=ignore_index,
        )

    @property
    def name(self) -> str:
        return "bitdistill"

    @property
    def requires_attentions(self) -> bool:
        """Only require attentions if gamma_attention > 0."""
        return self.gamma_attention > 0

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """
        Compute combined BitDistill loss.

        Args:
            model_outputs: Must contain 'logits', optionally 'attentions'
            batch: Must contain 'labels'
            teacher_outputs: Must contain 'logits', optionally 'attentions'

        Returns:
            ObjectiveOutput with combined loss and per-component metrics
        """
        if teacher_outputs is None:
            raise ValueError("BitDistillObjective requires teacher_outputs")

        # Logits distillation (always enabled)
        logits_output = self.logits_obj(model_outputs, batch, teacher_outputs)
        total_loss = self.lambda_logits * logits_output.loss

        metrics = {
            "logits_loss": logits_output.loss.detach(),
            "logits_kl": logits_output.metrics.get("kl_div", logits_output.loss.detach()),
        }

        # Attention distillation (optional)
        if self.gamma_attention > 0:
            attn_output = self.attention_obj(model_outputs, batch, teacher_outputs)
            total_loss = total_loss + self.gamma_attention * attn_output.loss
            metrics["attention_loss"] = attn_output.loss.detach()
        else:
            metrics["attention_loss"] = torch.tensor(0.0, device=total_loss.device)

        return ObjectiveOutput(
            loss=total_loss,
            metrics=metrics,
        )
