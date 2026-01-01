"""Mid-training stage with knowledge distillation.

Reference: MobileLLM-R1 paper (arXiv:2509.24945) Section 3.2
- Uses KL divergence loss to distill from teacher model
- Teacher: Llama-3.1-8B-Instruct (or similar)
- Student learns to match teacher's output distribution
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cheapertraining._legacy.training.stages.base import TrainingStage, StageConfig


class MidtrainStage(TrainingStage):
    """Mid-training stage with knowledge distillation.

    Trains student model to match teacher model's output distribution
    using KL divergence loss.
    """

    def __init__(
        self,
        *args,
        teacher_model: Optional[nn.Module] = None,
        temperature: float = 1.0,
        alpha: float = 1.0,
        **kwargs,
    ):
        """Initialize mid-training stage.

        Args:
            *args: Arguments passed to TrainingStage
            teacher_model: Teacher model for distillation (must be provided)
            temperature: Temperature for softmax (higher = softer distributions)
            alpha: Weight for KD loss (1.0 = pure KD, 0.0 = pure CE)
            **kwargs: Keyword arguments passed to TrainingStage
        """
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        if self.teacher_model is not None:
            # Freeze teacher and set to eval mode
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def set_teacher(self, teacher_model: nn.Module):
        """Set teacher model after initialization.

        Args:
            teacher_model: Teacher model for distillation
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        batch: dict[str, Tensor],
    ) -> Tuple[Tensor, dict[str, float]]:
        """Compute knowledge distillation loss.

        Args:
            batch: Dictionary with 'input_ids' and optionally 'attention_mask'

        Returns:
            Tuple of (loss, metrics_dict)
        """
        if self.teacher_model is None:
            raise RuntimeError("Teacher model not set. Call set_teacher() first.")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Student forward pass
        student_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        student_logits = student_outputs["logits"]

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            teacher_logits = teacher_outputs["logits"]

        # Shift for next-token prediction
        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Create mask for valid positions
        if attention_mask is not None:
            valid_mask = attention_mask[:, 1:].contiguous().bool()
        else:
            valid_mask = torch.ones_like(shift_labels, dtype=torch.bool)

        # KL divergence loss
        # KL(P_teacher || P_student) where we use teacher as target
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Compute per-token KL divergence
        kl_div = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)  # Sum over vocab dimension

        # Apply mask and compute mean
        kl_div = kl_div * valid_mask.float()
        kd_loss = kl_div.sum() / valid_mask.sum()

        # Scale by temperature squared (standard practice)
        kd_loss = kd_loss * (self.temperature ** 2)

        # Optionally add cross-entropy loss with labels
        if self.alpha < 1.0:
            # Set padding to ignore_index
            labels = shift_labels.masked_fill(~valid_mask, -100)
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        else:
            ce_loss = torch.tensor(0.0, device=kd_loss.device)
            total_loss = kd_loss

        # Compute metrics
        with torch.no_grad():
            # Student accuracy
            student_preds = student_logits.argmax(dim=-1)
            correct = (student_preds == shift_labels) & valid_mask
            student_acc = correct.sum().float() / valid_mask.sum().float()

            # Agreement with teacher
            teacher_preds = teacher_logits.argmax(dim=-1)
            agreement = (student_preds == teacher_preds) & valid_mask
            teacher_agreement = agreement.sum().float() / valid_mask.sum().float()

            # Perplexity (from CE perspective)
            perplexity = torch.exp(ce_loss) if self.alpha < 1.0 else torch.tensor(0.0)

        metrics = {
            "kd_loss": kd_loss.item(),
            "ce_loss": ce_loss.item(),
            "student_accuracy": student_acc.item(),
            "teacher_agreement": teacher_agreement.item(),
            "perplexity": perplexity.item(),
            "num_tokens": valid_mask.sum().item(),
        }

        return total_loss, metrics
