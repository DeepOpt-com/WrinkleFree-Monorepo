"""Low-Rank Correction reconstruction objective.

Trains LRC adapters (U, V matrices) to minimize the reconstruction error
between the original model and the quantized model with LRC correction.

Based on paper: "Low-Rank Correction for Quantized LLMs" (arxiv 2412.07902)

The loss measures how well the LRC-corrected quantized model approximates
the original (teacher) model at each transformer layer:

    L = sum_l weight_l * ||h_teacher^l - h_student^l||^2

Where:
    - h_teacher^l: Hidden state from layer l of the original (fp16) teacher
    - h_student^l: Hidden state from layer l of the quantized + LRC student

This is equivalent to minimizing the reconstruction error of the LRC paper:
    ||W @ X - W_quant @ Q_a(X) - U @ V^T @ X||^2

because the hidden states capture the cumulative effect of all layers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import torch
import torch.nn.functional as F

from wrinklefree.objectives.base import Objective, ObjectiveOutput


class LRCLossType(Enum):
    """Loss metric types for LRC reconstruction."""

    MSE = "mse"
    MSE_NORMALIZED = "mse_normalized"
    COSINE = "cosine"


class LRCReconstructionObjective(Objective):
    """Layer-wise reconstruction objective for LRC training.

    Computes the reconstruction error between student (quantized + LRC)
    and teacher (original fp16) hidden states at each transformer layer.

    This trains the LRC adapters to compensate for quantization error.

    IMPORTANT: This objective requires:
    - Teacher model (fp16 original) for reconstruction targets
    - Hidden states from both student and teacher

    Args:
        loss_type: Type of loss metric:
            - "mse": Mean squared error
            - "mse_normalized": L2-normalized MSE (OneBit style)
            - "cosine": 1 - cosine_similarity
        layer_weights: Per-layer weight configuration:
            - None: Uniform weights across layers
            - "progressive": Linearly increasing (later layers weighted more)
            - "exponential": Exponentially increasing
            - list[float]: Custom weights
        temperature: Scaling factor for the loss (default: 1.0)
        normalize: Whether to L2-normalize hidden states before computing loss
    """

    requires_teacher = True
    requires_hidden_states = True
    requires_attentions = False
    modifies_input = False

    def __init__(
        self,
        loss_type: LRCLossType | str = LRCLossType.MSE,
        layer_weights: Optional[list[float] | str] = None,
        temperature: float = 1.0,
        normalize: bool = False,
    ):
        super().__init__()

        if isinstance(loss_type, str):
            loss_type = LRCLossType(loss_type)

        self.loss_type = loss_type
        self.layer_weights_config = layer_weights
        self.temperature = temperature
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "lrc_reconstruction"

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """Compute layer-wise reconstruction loss.

        Args:
            model_outputs: Student (quantized + LRC) outputs with hidden_states
            batch: Input batch with attention_mask
            teacher_outputs: Teacher (original fp16) outputs with hidden_states

        Returns:
            ObjectiveOutput with reconstruction loss and metrics
        """
        if teacher_outputs is None:
            raise ValueError("LRCReconstructionObjective requires teacher_outputs")

        student_hidden = model_outputs.get("hidden_states")
        teacher_hidden = teacher_outputs.get("hidden_states")

        if student_hidden is None or teacher_hidden is None:
            raise ValueError(
                "Both student and teacher must output hidden_states. "
                "Set output_hidden_states=True in model config."
            )

        # Skip embedding layer (index 0), only compare transformer layer outputs
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

        for idx, (s_h, t_h) in enumerate(zip(student_hidden, teacher_hidden)):
            layer_loss = self._compute_layer_loss(s_h, t_h, attention_mask)
            layer_losses.append(layer_loss.detach())
            total_loss = total_loss + weights[idx] * layer_loss

        # Apply temperature scaling
        # Note: temperature > 1 reduces loss magnitude (inverse of typical distillation scaling)
        total_loss = total_loss / self.temperature

        mean_layer_loss = torch.stack(layer_losses).mean()

        return ObjectiveOutput(
            loss=total_loss,
            metrics={
                "mean_layer_loss": mean_layer_loss,
                "num_layers": torch.tensor(float(num_layers), device=device),
                "temperature": torch.tensor(self.temperature, device=device),
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
        elif isinstance(self.layer_weights_config, list):
            if len(self.layer_weights_config) != num_layers:
                raise ValueError(
                    f"layer_weights length ({len(self.layer_weights_config)}) "
                    f"must match num_layers ({num_layers})"
                )
            total = sum(self.layer_weights_config)
            if total <= 0:
                raise ValueError(f"layer_weights must sum to > 0, got {total}")
            return [w / total for w in self.layer_weights_config]
        else:
            raise ValueError(f"Unknown layer_weights config: {self.layer_weights_config}")

    def _compute_layer_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute reconstruction loss for a single layer."""
        if self.loss_type == LRCLossType.MSE:
            return self._mse_loss(student_h, teacher_h, attention_mask, normalize=False)
        elif self.loss_type == LRCLossType.MSE_NORMALIZED:
            return self._mse_loss(student_h, teacher_h, attention_mask, normalize=True)
        elif self.loss_type == LRCLossType.COSINE:
            return self._cosine_loss(student_h, teacher_h, attention_mask)
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
        if normalize or self.normalize:
            student_h = F.normalize(student_h, p=2, dim=-1)
            teacher_h = F.normalize(teacher_h.detach(), p=2, dim=-1)
        else:
            teacher_h = teacher_h.detach()

        mse = (student_h - teacher_h).pow(2).mean(dim=-1)

        if attention_mask is not None:
            mse = mse * attention_mask.to(mse.dtype)
            return mse.sum() / attention_mask.sum().clamp(min=1)
        return mse.mean()

    def _cosine_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Cosine distance loss (1 - cosine_similarity)."""
        cos_sim = F.cosine_similarity(student_h, teacher_h.detach(), dim=-1)
        loss = 1.0 - cos_sim

        if attention_mask is not None:
            loss = loss * attention_mask.to(loss.dtype)
            return loss.sum() / attention_mask.sum().clamp(min=1)
        return loss.mean()
