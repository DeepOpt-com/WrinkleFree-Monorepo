"""Layer-wise hidden state distillation loss for Stage 1.9.

This module provides layer-wise distillation to align BitNet hidden states
with the original full-precision teacher model at each transformer layer.

Research basis:
- OneBit (arxiv.org/abs/2402.11295): L2-normalized MSE for scale-invariance
- BitDistill (arxiv.org/abs/2510.13998): Later layers often more important
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerwiseLossType(Enum):
    """Loss metric types for layer-wise distillation."""

    # NOTE: cosine loss removed - mathematically equivalent to mse_normalized
    # For L2-normalized vectors: ||s-t||² = 2 - 2*cos(s,t) = 2*(1-cos(s,t))
    # So mse_normalized = 2 * cosine_loss, making them redundant.
    MSE = "mse"  # ||student_h - teacher_h||^2
    MSE_NORMALIZED = "mse_normalized"  # MSE with L2 normalization (OneBit style)
    KL = "kl"  # KL divergence after projecting to vocab space
    INNER_PRODUCT = "inner_product"  # -<student_h, teacher_h> (normalized)


class LayerwiseDistillationLoss(nn.Module):
    """
    Layer-wise hidden state distillation loss.

    Aligns student hidden states to teacher hidden states at each transformer layer.
    Supports multiple loss metrics configurable via LayerwiseLossType.

    Args:
        loss_type: Type of loss metric to use (default: MSE_NORMALIZED)
        layer_weights: Per-layer weights configuration:
            - None: Uniform weights (1/L for each layer)
            - "progressive": Linearly increasing (later layers weighted more)
            - "exponential": Exponentially increasing weights
            - list[float]: Custom weights per layer
        hidden_size: Hidden dimension (required for KL loss projection)
        vocab_size: Vocabulary size (required for KL loss)
        temperature: Temperature for KL loss softmax (default: 1.0)
        normalize: Whether to L2-normalize hidden states before computing loss
        reduction: Loss reduction method ("mean" or "sum")
    """

    def __init__(
        self,
        loss_type: LayerwiseLossType | str = LayerwiseLossType.MSE_NORMALIZED,
        layer_weights: Optional[list[float] | str] = None,
        hidden_size: Optional[int] = None,
        vocab_size: Optional[int] = None,
        temperature: float = 1.0,
        normalize: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()

        # Handle string loss type from config
        if isinstance(loss_type, str):
            loss_type = LayerwiseLossType(loss_type)

        self.loss_type = loss_type
        self.layer_weights = layer_weights
        self.temperature = temperature
        self.normalize = normalize
        self.reduction = reduction

        # KL loss requires projection to vocab space
        if loss_type == LayerwiseLossType.KL:
            if hidden_size is None or vocab_size is None:
                raise ValueError("KL loss requires hidden_size and vocab_size")
            self.vocab_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.vocab_projection = None

    def forward(
        self,
        student_hidden_states: list[torch.Tensor],
        teacher_hidden_states: list[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute layer-wise distillation loss.

        Args:
            student_hidden_states: List of hidden states per layer,
                each tensor has shape (batch, seq, hidden_dim).
                Should include all transformer layer outputs (not embedding).
            teacher_hidden_states: List of teacher hidden states, same shape.
            attention_mask: Optional mask of shape (batch, seq),
                where 1 = valid token, 0 = padding.

        Returns:
            Dictionary containing:
                - loss: Total weighted loss across all layers
                - layer_losses: List of per-layer loss values (detached, for logging)
                - mean_layer_loss: Mean of layer losses (for logging)
        """
        if len(student_hidden_states) != len(teacher_hidden_states):
            raise ValueError(
                f"Layer count mismatch: student={len(student_hidden_states)}, "
                f"teacher={len(teacher_hidden_states)}"
            )

        num_layers = len(student_hidden_states)
        if num_layers == 0:
            device = "cpu"
            return {
                "loss": torch.tensor(0.0, device=device),
                "layer_losses": [],
                "mean_layer_loss": torch.tensor(0.0, device=device),
            }

        weights = self._get_layer_weights(num_layers)
        device = student_hidden_states[0].device

        layer_losses = []
        total_loss = torch.tensor(0.0, device=device, dtype=student_hidden_states[0].dtype)

        for idx, (s_hidden, t_hidden) in enumerate(
            zip(student_hidden_states, teacher_hidden_states)
        ):
            layer_loss = self._compute_layer_loss(s_hidden, t_hidden, attention_mask)
            layer_losses.append(layer_loss.detach())
            total_loss = total_loss + weights[idx] * layer_loss

        return {
            "loss": total_loss,
            "layer_losses": layer_losses,
            "mean_layer_loss": torch.stack(layer_losses).mean(),
        }

    def _get_layer_weights(self, num_layers: int) -> list[float]:
        """
        Get layer weights based on configuration.

        Args:
            num_layers: Number of transformer layers

        Returns:
            List of normalized weights (sum to 1.0)
        """
        if self.layer_weights is None:
            # Uniform weights
            return [1.0 / num_layers] * num_layers
        elif self.layer_weights == "progressive":
            # Linearly increasing weights (later layers more important)
            # Based on BitDistill findings that later layers matter more
            weights = [(i + 1) for i in range(num_layers)]
            total = sum(weights)
            return [w / total for w in weights]
        elif self.layer_weights == "exponential":
            # Exponentially increasing weights
            weights = [2**i for i in range(num_layers)]
            total = sum(weights)
            return [w / total for w in weights]
        else:
            # User-provided custom weights
            if len(self.layer_weights) != num_layers:
                raise ValueError(
                    f"layer_weights length ({len(self.layer_weights)}) "
                    f"must match num_layers ({num_layers})"
                )
            total = sum(self.layer_weights)
            return [w / total for w in self.layer_weights]

    def _compute_layer_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss for a single layer based on loss_type.

        Args:
            student_h: Student hidden states (batch, seq, hidden)
            teacher_h: Teacher hidden states (batch, seq, hidden)
            attention_mask: Optional mask (batch, seq)

        Returns:
            Scalar loss tensor
        """
        if self.loss_type == LayerwiseLossType.MSE:
            return self._mse_loss(student_h, teacher_h, attention_mask, normalize=False)
        elif self.loss_type == LayerwiseLossType.MSE_NORMALIZED:
            return self._mse_loss(student_h, teacher_h, attention_mask, normalize=True)
        elif self.loss_type == LayerwiseLossType.KL:
            return self._kl_loss(student_h, teacher_h, attention_mask)
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
        """
        MSE loss with optional L2 normalization.

        When normalize=True (OneBit style), hidden states are L2-normalized
        before computing MSE, making the loss scale-invariant.
        """
        if normalize:
            # L2 normalize to unit vectors (OneBit approach)
            student_h = F.normalize(student_h, p=2, dim=-1)
            teacher_h = F.normalize(teacher_h.detach(), p=2, dim=-1)
        else:
            teacher_h = teacher_h.detach()

        # MSE per position: (batch, seq)
        mse = (student_h - teacher_h).pow(2).mean(dim=-1)

        if attention_mask is not None:
            mse = mse * attention_mask.to(mse.dtype)
            return mse.sum() / attention_mask.sum().clamp(min=1)
        return mse.mean()

    def _kl_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        KL divergence after projecting hidden states to vocab space.

        Projects hidden states to vocabulary logits, then computes
        KL divergence between softmax distributions.
        """
        # Project to vocab space
        student_logits = self.vocab_projection(student_h)
        teacher_logits = self.vocab_projection(teacher_h.detach())

        # Apply temperature and compute distributions
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence: sum over vocab dimension
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

        # Scale by temperature squared (standard KD practice)
        if attention_mask is not None:
            kl = kl * attention_mask.to(kl.dtype)
            return kl.sum() / attention_mask.sum().clamp(min=1) * (self.temperature**2)
        return kl.mean() * (self.temperature**2)

    def _inner_product_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Negative inner product loss (normalized).

        Maximizes the inner product between normalized hidden states.
        Loss = -<norm(student), norm(teacher)>
        """
        # Normalize to unit vectors
        student_h = F.normalize(student_h, p=2, dim=-1)
        teacher_h = F.normalize(teacher_h.detach(), p=2, dim=-1)

        # Inner product per position: -<s, t>
        inner = -(student_h * teacher_h).sum(dim=-1)

        if attention_mask is not None:
            inner = inner * attention_mask.to(inner.dtype)
            return inner.sum() / attention_mask.sum().clamp(min=1)
        return inner.mean()
