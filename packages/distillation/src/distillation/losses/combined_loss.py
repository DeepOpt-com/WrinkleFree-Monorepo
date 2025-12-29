"""Combined loss function for BitDistill training."""

import torch
import torch.nn as nn

from distillation.losses.attention_loss import (
    AttentionDistillationLoss,
    BitDistillAttentionRelationLoss,
)
from distillation.losses.logits_loss import LogitsDistillationLoss


class BitDistillLoss(nn.Module):
    """
    Combined loss for BitDistill (Equation 13 from arxiv.org/abs/2510.13998).

    L = L_CE + lambda * L_LD + gamma * L_AD

    Where:
    - L_CE: Cross-entropy loss on labels
    - L_LD: Logits distillation loss (KL divergence)
    - L_AD: Attention distillation loss

    For L_AD, uses BitDistill's attention relation distillation (Equation 11):
        R = Softmax(A · Aᵀ / √d_r)
    Distilled at a SINGLE layer for optimization flexibility.

    Recommended coefficients from the paper:
    - Classification tasks: lambda=10, gamma=1e-5
    - Summarization tasks: lambda=1, gamma=1e-3

    Args:
        lambda_logits: Weight for logits distillation loss
        gamma_attention: Weight for attention distillation loss
        temperature: Temperature for logits distillation
        alpha: Alpha for attention distillation
        ignore_index: Index to ignore in CE loss (typically -100)
        use_relation_distill: Use BitDistill A·Aᵀ relations (True) or direct attention (False)
        distill_layer: Layer for attention distillation (-1 = last, BitDistill recommends single layer)
    """

    def __init__(
        self,
        lambda_logits: float = 10.0,
        gamma_attention: float = 1e-5,
        temperature: float = 5.0,
        alpha: float = 1.0,
        ignore_index: int = -100,
        use_relation_distill: bool = True,
        distill_layer: int = -1,
    ):
        super().__init__()
        self.lambda_logits = lambda_logits
        self.gamma_attention = gamma_attention
        self.ignore_index = ignore_index

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.logits_loss = LogitsDistillationLoss(temperature=temperature)

        # BitDistill uses single-layer attention relation distillation
        if use_relation_distill:
            self.attention_loss = BitDistillAttentionRelationLoss(
                alpha=alpha,
                distill_layer=distill_layer,
            )
        else:
            self.attention_loss = AttentionDistillationLoss(alpha=alpha)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_attentions: list[torch.Tensor] | None,
        teacher_attentions: list[torch.Tensor] | None,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined BitDistill loss.

        Args:
            student_logits: Student model logits (batch, seq, vocab)
            teacher_logits: Teacher model logits (batch, seq, vocab)
            student_attentions: List of student attention weights per layer
            teacher_attentions: List of teacher attention weights per layer
            labels: Ground truth labels for CE loss (batch, seq)
            attention_mask: Optional attention mask (batch, seq)

        Returns:
            Dictionary containing:
                - loss: Total combined loss
                - ce_loss: Cross-entropy component
                - logits_distill_loss: Logits distillation component
                - attention_distill_loss: Attention distillation component
        """
        # Cross-entropy loss
        # Shift for next token prediction
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Logits distillation loss
        # Create mask for valid (non-ignored) positions
        label_mask = (shift_labels != self.ignore_index).float() if attention_mask is None else None
        shift_student = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_logits[..., :-1, :].contiguous()
        ld = self.logits_loss(shift_student, shift_teacher, mask=label_mask)

        # Attention distillation loss
        ad = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
        if (
            student_attentions is not None
            and teacher_attentions is not None
            and len(student_attentions) > 0
            and len(teacher_attentions) > 0
        ):
            ad = self.attention_loss(student_attentions, teacher_attentions, attention_mask)

        # Combined loss
        total = ce + self.lambda_logits * ld + self.gamma_attention * ad

        return {
            "loss": total,
            "ce_loss": ce,
            "logits_distill_loss": ld,
            "attention_distill_loss": ad,
        }


class ClassificationDistillLoss(nn.Module):
    """
    Distillation loss for classification tasks.

    Combines classification loss with logits distillation only
    (no attention distillation for classification).

    L = L_CE + lambda * L_LD

    Args:
        lambda_logits: Weight for logits distillation
        temperature: Temperature for distillation
        num_labels: Number of classification labels
    """

    def __init__(
        self,
        lambda_logits: float = 10.0,
        temperature: float = 5.0,
        num_labels: int = 2,
    ):
        super().__init__()
        self.lambda_logits = lambda_logits
        self.num_labels = num_labels

        self.ce_loss = nn.CrossEntropyLoss()
        self.logits_loss = LogitsDistillationLoss(temperature=temperature)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute classification distillation loss.

        Args:
            student_logits: Student predictions (batch, num_labels)
            teacher_logits: Teacher predictions (batch, num_labels)
            labels: Ground truth labels (batch,)

        Returns:
            Dictionary with loss components
        """
        ce = self.ce_loss(student_logits, labels)
        ld = self.logits_loss(student_logits, teacher_logits)

        total = ce + self.lambda_logits * ld

        return {
            "loss": total,
            "ce_loss": ce,
            "logits_distill_loss": ld,
        }
