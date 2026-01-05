"""Target Concrete Score (TCS) distillation objective for DLM students.

Based on:
- Apple's TCSM (ICML 2025)
- DDLM: https://openreview.net/forum?id=xfw92pDy2u

Key differences from standard logits distillation:
1. NO logit shifting - DLM operates on masked positions, not next-token prediction
2. Top-K TCS estimation - sparse distribution matching for efficiency
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from wf_train.objectives.base import Objective, ObjectiveOutput


class TCSDistillationObjective(Objective):
    """
    Target Concrete Score (TCS) distillation for DLM students.

    Uses top-K sparse TCS estimation for efficiency.
    NO logit shifting (DLM predicts masked tokens, not next tokens).

    For each position, we:
    1. Get teacher's top-K logits and indices
    2. Gather student's logits at those indices
    3. Compute KL divergence on the sparse distributions

    This is computationally efficient while capturing most of the distribution.

    Args:
        temperature: Temperature for KL divergence
        top_k: Number of top tokens for sparse TCS estimation
        ignore_index: Index to ignore in labels
    """

    requires_teacher = True
    requires_hidden_states = False
    requires_attentions = False
    modifies_input = False

    def __init__(
        self,
        temperature: float = 5.0,
        top_k: int = 100,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.ignore_index = ignore_index

    @property
    def name(self) -> str:
        return "tcs_distill"

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """
        Compute TCS distillation loss with Top-K estimation.

        Args:
            model_outputs: Must contain 'logits' (batch, seq, vocab)
            batch: Must contain 'labels' or 'dlm_labels' for masking
            teacher_outputs: Must contain 'logits'

        Returns:
            ObjectiveOutput with TCS loss
        """
        if teacher_outputs is None:
            raise ValueError("TCSDistillationObjective requires teacher_outputs")

        student_logits = model_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        # Use DLM labels if available, otherwise regular labels
        # DLM labels only have values at masked positions
        labels = batch.get("dlm_labels", batch["labels"])

        # Response mask: positions where we compute loss (not -100)
        response_mask = labels != self.ignore_index

        batch_size, seq_len, vocab_size = student_logits.shape

        # Get teacher's top-K predictions
        teacher_topk_logits, topk_indices = torch.topk(
            teacher_logits, k=min(self.top_k, vocab_size), dim=-1
        )

        # Gather student logits at the same indices
        student_topk_logits = torch.gather(student_logits, dim=-1, index=topk_indices)

        # Temperature-scaled probabilities
        teacher_probs = F.softmax(teacher_topk_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_topk_logits / self.temperature, dim=-1)

        # KL divergence: sum_i p_teacher * (log p_teacher - log p_student)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

        # Mask to response positions only
        kl = kl * response_mask.float()
        num_valid = response_mask.sum().clamp(min=1)

        # Temperature^2 scaling (standard for distillation)
        loss = (kl.sum() / num_valid) * (self.temperature**2)

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "tcs_kl": (kl.sum() / num_valid).detach(),
                "num_masked": response_mask.sum().detach().float(),
            },
        )
