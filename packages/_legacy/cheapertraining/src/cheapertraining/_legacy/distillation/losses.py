"""Knowledge distillation loss functions.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) Section 3.2
Uses KL divergence to distill knowledge from teacher to student.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KLDivergenceLoss(nn.Module):
    """KL Divergence loss for knowledge distillation.

    Computes KL(teacher || student) to encourage student to match teacher distribution.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "batchmean",
    ):
        """Initialize KL divergence loss.

        Args:
            temperature: Temperature for softening distributions
            reduction: Reduction method ('batchmean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute KL divergence loss.

        Args:
            student_logits: Student model logits (batch, seq, vocab)
            teacher_logits: Teacher model logits (batch, seq, vocab)
            mask: Optional mask for valid positions (batch, seq)

        Returns:
            KL divergence loss
        """
        # Apply temperature
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Compute KL divergence per token
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)  # Sum over vocab dimension

        # Apply mask if provided
        if mask is not None:
            kl_div = kl_div * mask.float()
            if self.reduction == "batchmean":
                return kl_div.sum() / mask.sum() * (self.temperature ** 2)
            elif self.reduction == "sum":
                return kl_div.sum() * (self.temperature ** 2)
            else:
                return kl_div * (self.temperature ** 2)
        else:
            if self.reduction == "batchmean":
                return kl_div.mean() * (self.temperature ** 2)
            elif self.reduction == "sum":
                return kl_div.sum() * (self.temperature ** 2)
            else:
                return kl_div * (self.temperature ** 2)


class DistillationLoss(nn.Module):
    """Combined distillation loss with optional CE component.

    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
    """

    def __init__(
        self,
        temperature: float = 1.0,
        alpha: float = 1.0,
        ignore_index: int = -100,
    ):
        """Initialize distillation loss.

        Args:
            temperature: Temperature for KL divergence
            alpha: Weight for KD loss (1.0 = pure KD, 0.0 = pure CE)
            ignore_index: Label index to ignore in CE loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index

        self.kl_loss = KLDivergenceLoss(temperature=temperature)

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """Compute combined distillation loss.

        Args:
            student_logits: Student model logits (batch, seq, vocab)
            teacher_logits: Teacher model logits (batch, seq, vocab)
            labels: Optional target labels for CE loss
            attention_mask: Optional attention mask

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Create mask from attention_mask if provided
        mask = attention_mask if attention_mask is not None else None

        # KL divergence loss
        kd_loss = self.kl_loss(student_logits, teacher_logits, mask)

        metrics = {"kd_loss": kd_loss.item()}

        # Optional CE loss
        if labels is not None and self.alpha < 1.0:
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=self.ignore_index,
            )
            metrics["ce_loss"] = ce_loss.item()
            total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        else:
            total_loss = kd_loss

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics


class ForwardKL(nn.Module):
    """Forward KL divergence: KL(teacher || student).

    Encourages student to cover all modes of teacher distribution.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """Compute forward KL divergence."""
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl = (teacher_probs * (teacher_probs.log() - student_log_probs)).sum(dim=-1)
        return kl.mean() * (self.temperature ** 2)


class ReverseKL(nn.Module):
    """Reverse KL divergence: KL(student || teacher).

    Encourages student to focus on high-probability modes of teacher.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """Compute reverse KL divergence."""
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)

        kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        return kl.mean() * (self.temperature ** 2)
