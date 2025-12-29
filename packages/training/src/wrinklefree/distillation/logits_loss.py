"""Logits distillation loss for BitDistill."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitsDistillationLoss(nn.Module):
    """
    KL divergence loss between teacher and student logits.

    L_LD = KL(P_teacher || P_student) * T^2

    Where P = softmax(z / T) with temperature T.

    Higher temperature produces softer probability distributions,
    which transfer more information about relative class probabilities.

    Args:
        temperature: Softmax temperature (higher = softer distributions)
        reduction: How to reduce the loss ("mean", "sum", "batchmean", "none")
    """

    def __init__(
        self,
        temperature: float = 5.0,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between teacher and student logits.

        Args:
            student_logits: Student model logits (batch, seq, vocab) or (batch, vocab)
            teacher_logits: Teacher model logits, same shape as student
            mask: Optional mask for valid positions (1 = valid, 0 = ignore)

        Returns:
            KL divergence loss scaled by temperature^2
        """
        # Apply temperature scaling
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence: sum(p * log(p/q)) = sum(p * (log(p) - log(q)))
        # F.kl_div expects log(q) as input, p as target
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        )

        # Sum over vocab dimension
        kl_div = kl_div.sum(dim=-1)

        # Apply mask if provided
        if mask is not None:
            kl_div = kl_div * mask
            if self.reduction == "mean" or self.reduction == "batchmean":
                return kl_div.sum() / mask.sum().clamp(min=1) * (self.temperature ** 2)
            elif self.reduction == "sum":
                return kl_div.sum() * (self.temperature ** 2)
            else:
                return kl_div * (self.temperature ** 2)

        # Reduce
        if self.reduction == "mean":
            loss = kl_div.mean()
        elif self.reduction == "batchmean":
            loss = kl_div.sum() / student_logits.size(0)
        elif self.reduction == "sum":
            loss = kl_div.sum()
        else:
            loss = kl_div

        # Scale by temperature^2 (standard practice for distillation)
        return loss * (self.temperature ** 2)


class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss with soft targets from teacher.

    Alternative to KL divergence that can be more stable in some cases.

    L = -sum(P_teacher * log(P_student))

    Args:
        temperature: Softmax temperature
    """

    def __init__(self, temperature: float = 5.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute soft target cross-entropy.

        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            mask: Optional mask for valid positions

        Returns:
            Soft target cross-entropy loss
        """
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Cross-entropy: -sum(p * log(q))
        loss = -(teacher_probs * student_log_probs).sum(dim=-1)

        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1) * (self.temperature ** 2)

        return loss.mean() * (self.temperature ** 2)
