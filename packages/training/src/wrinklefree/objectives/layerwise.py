"""Layer-wise distillation objective (Stage 1.9).

Aligns student hidden states with teacher at each transformer layer.
This accelerates convergence by providing direct supervision at each layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrinklefree.objectives.base import Objective, ObjectiveOutput


class LayerwiseLossType(Enum):
    """Loss metric types for layer-wise distillation."""

    MSE = "mse"
    MSE_NORMALIZED = "mse_normalized"  # OneBit style
    INNER_PRODUCT = "inner_product"


class LayerwiseDistillationObjective(Objective):
    """Layer-wise hidden state distillation objective.

    Aligns student hidden states to teacher hidden states at each
    transformer layer. Supports multiple loss metrics.

    Args:
        loss_type: Type of loss metric (default: MSE_NORMALIZED)
        layer_weights: Per-layer weight configuration:
            - None: Uniform weights
            - "progressive": Linearly increasing (later layers weighted more)
            - "exponential": Exponentially increasing
            - list[float]: Custom weights
        normalize: Whether to L2-normalize hidden states
    """

    requires_teacher = True
    requires_hidden_states = True
    modifies_input = False

    def __init__(
        self,
        loss_type: LayerwiseLossType | str = LayerwiseLossType.MSE_NORMALIZED,
        layer_weights: Optional[list[float] | str] = None,
        normalize: bool = True,
    ):
        super().__init__()

        if isinstance(loss_type, str):
            loss_type = LayerwiseLossType(loss_type)

        self.loss_type = loss_type
        self.layer_weights_config = layer_weights
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "layerwise_distill"

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """Compute layer-wise distillation loss.

        Args:
            model_outputs: Must contain 'hidden_states' tuple
            batch: Used for attention_mask if available
            teacher_outputs: Must contain 'hidden_states' tuple

        Returns:
            ObjectiveOutput with total loss and per-layer metrics
        """
        if teacher_outputs is None:
            raise ValueError("LayerwiseDistillationObjective requires teacher_outputs")

        student_hidden = model_outputs.get("hidden_states")
        teacher_hidden = teacher_outputs.get("hidden_states")

        if student_hidden is None or teacher_hidden is None:
            raise ValueError(
                "Both student and teacher must output hidden_states. "
                "Set output_hidden_states=True in model config."
            )

        # Skip embedding layer (index 0), use transformer layers only
        student_hidden = student_hidden[1:]
        teacher_hidden = teacher_hidden[1:]

        if len(student_hidden) != len(teacher_hidden):
            raise ValueError(
                f"Layer count mismatch: student={len(student_hidden)}, "
                f"teacher={len(teacher_hidden)}"
            )

        num_layers = len(student_hidden)
        if num_layers == 0:
            device = model_outputs["logits"].device
            return ObjectiveOutput(
                loss=torch.tensor(0.0, device=device),
                metrics={"mean_layer_loss": torch.tensor(0.0, device=device)},
            )

        attention_mask = batch.get("attention_mask")
        weights = self._get_layer_weights(num_layers)
        device = student_hidden[0].device

        layer_losses = []
        total_loss = torch.tensor(0.0, device=device, dtype=student_hidden[0].dtype)

        for idx, (s_hidden, t_hidden) in enumerate(zip(student_hidden, teacher_hidden)):
            layer_loss = self._compute_layer_loss(s_hidden, t_hidden, attention_mask)
            layer_losses.append(layer_loss.detach())
            total_loss = total_loss + weights[idx] * layer_loss

        # Metrics
        mean_layer_loss = torch.stack(layer_losses).mean()

        return ObjectiveOutput(
            loss=total_loss,
            metrics={
                "mean_layer_loss": mean_layer_loss,
                "num_layers": torch.tensor(float(num_layers), device=device),
            },
        )

    def _get_layer_weights(self, num_layers: int) -> list[float]:
        """Get normalized layer weights."""
        if self.layer_weights_config is None:
            return [1.0 / num_layers] * num_layers
        elif self.layer_weights_config == "progressive":
            weights = [(i + 1) for i in range(num_layers)]
            total = sum(weights)
            return [w / total for w in weights]
        elif self.layer_weights_config == "exponential":
            weights = [2**i for i in range(num_layers)]
            total = sum(weights)
            return [w / total for w in weights]
        else:
            if len(self.layer_weights_config) != num_layers:
                raise ValueError(
                    f"layer_weights length ({len(self.layer_weights_config)}) "
                    f"must match num_layers ({num_layers})"
                )
            total = sum(self.layer_weights_config)
            return [w / total for w in self.layer_weights_config]

    def _compute_layer_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss for a single layer."""
        if self.loss_type == LayerwiseLossType.MSE:
            return self._mse_loss(student_h, teacher_h, attention_mask, normalize=False)
        elif self.loss_type == LayerwiseLossType.MSE_NORMALIZED:
            return self._mse_loss(student_h, teacher_h, attention_mask, normalize=True)
        elif self.loss_type == LayerwiseLossType.INNER_PRODUCT:
            return self._inner_product_loss(student_h, teacher_h, attention_mask)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _mse_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        normalize: bool = False,
    ) -> torch.Tensor:
        """MSE loss with optional L2 normalization."""
        if normalize:
            student_h = F.normalize(student_h, p=2, dim=-1)
            teacher_h = F.normalize(teacher_h.detach(), p=2, dim=-1)
        else:
            teacher_h = teacher_h.detach()

        mse = (student_h - teacher_h).pow(2).mean(dim=-1)

        if attention_mask is not None:
            mse = mse * attention_mask.to(mse.dtype)
            return mse.sum() / attention_mask.sum().clamp(min=1)
        return mse.mean()

    def _inner_product_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Negative inner product loss (normalized)."""
        student_h = F.normalize(student_h, p=2, dim=-1)
        teacher_h = F.normalize(teacher_h.detach(), p=2, dim=-1)

        inner = -(student_h * teacher_h).sum(dim=-1)

        if attention_mask is not None:
            inner = inner * attention_mask.to(inner.dtype)
            return inner.sum() / attention_mask.sum().clamp(min=1)
        return inner.mean()
