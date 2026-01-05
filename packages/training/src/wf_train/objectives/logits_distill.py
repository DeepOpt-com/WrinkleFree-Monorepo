"""Logits distillation objective (KL divergence on teacher/student logits).

Based on BitDistill Equation 13: arxiv.org/abs/2510.13998
L_LD = KL(P_teacher || P_student) * T^2
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from wf_train.objectives.base import Objective, ObjectiveOutput


class LogitsDistillationObjective(Objective):
    """
    KL divergence logits distillation (BitDistill Equation 13).

    L_LD = KL(P_teacher || P_student) * T^2

    Where P = softmax(z / T) with temperature T.

    Higher temperature produces softer probability distributions,
    which transfer more information about relative class probabilities.

    Args:
        temperature: Softmax temperature (higher = softer distributions)
        ignore_index: Index to ignore in labels (for masking)
        shift_labels: Whether to shift logits for next-token prediction (AR models).
            Set to False for DLM models that predict masked tokens.
    """

    requires_teacher = True
    requires_hidden_states = False
    requires_attentions = False
    modifies_input = False

    def __init__(
        self,
        temperature: float = 5.0,
        ignore_index: int = -100,
        shift_labels: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.shift_labels = shift_labels

    @property
    def name(self) -> str:
        return "logits_distill"

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """
        Compute KL divergence loss between teacher and student logits.

        Args:
            model_outputs: Must contain 'logits' (batch, seq, vocab)
            batch: Must contain 'labels' for masking
            teacher_outputs: Must contain 'logits' (batch, seq, vocab)

        Returns:
            ObjectiveOutput with KL divergence loss
        """
        if teacher_outputs is None:
            raise ValueError("LogitsDistillationObjective requires teacher_outputs")

        student_logits = model_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        # Get labels for masking (use original labels if available from DLM)
        labels = batch.get("_original_labels", batch["labels"])

        # Shift for next-token prediction (AR models)
        if self.shift_labels:
            student_logits = student_logits[..., :-1, :].contiguous()
            teacher_logits = teacher_logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        # Create mask for valid positions
        mask = (labels != self.ignore_index).float()

        # Temperature-scaled probabilities
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence: sum_i p_teacher * (log p_teacher - log p_student)
        # Using F.kl_div: expects log(q), p as arguments
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        kl_div = kl_div.sum(dim=-1)  # Sum over vocab dimension

        # Apply mask and normalize
        kl_div = kl_div * mask
        num_valid = mask.sum().clamp(min=1)

        # Temperature^2 scaling (standard for distillation)
        loss = (kl_div.sum() / num_valid) * (self.temperature**2)

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "kl_div": (kl_div.sum() / num_valid).detach(),
                "temperature": torch.tensor(self.temperature, device=loss.device),
            },
        )
