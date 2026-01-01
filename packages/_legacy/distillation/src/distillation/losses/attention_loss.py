"""Attention distillation loss for BitDistill (MiniLM-style)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDistillationLoss(nn.Module):
    """
    MiniLM-style multi-head attention relation distillation.

    Distills the attention distributions from teacher to student,
    helping the student learn similar attention patterns.

    L_AD = (1 / |L|) * sum_l (alpha / (A * |x|)) * sum_a sum_t KL(R_t || R_s)

    Where:
    - L is the set of layers
    - A is the number of attention heads
    - |x| is the sequence length
    - R is the attention distribution for head a at position t

    Args:
        alpha: Scaling coefficient
        layer_weights: Optional per-layer weights (default: uniform)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        layer_weights: list[float] | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.layer_weights = layer_weights

    def forward(
        self,
        student_attentions: list[torch.Tensor],
        teacher_attentions: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute attention distillation loss.

        Args:
            student_attentions: List of attention weights per layer
                Each tensor has shape (batch, num_heads, seq_len, seq_len)
            teacher_attentions: List of attention weights per layer
                Same shape as student_attentions
            attention_mask: Optional mask of shape (batch, seq_len)
                1 = valid token, 0 = padding

        Returns:
            Attention distillation loss
        """
        if len(student_attentions) != len(teacher_attentions):
            raise ValueError(
                f"Number of layers mismatch: {len(student_attentions)} vs {len(teacher_attentions)}"
            )

        num_layers = len(student_attentions)
        if num_layers == 0:
            return torch.tensor(0.0, device=student_attentions[0].device if student_attentions else "cpu")

        # Get layer weights
        if self.layer_weights is not None:
            weights = self.layer_weights
        else:
            weights = [1.0 / num_layers] * num_layers

        total_loss = 0.0

        for layer_idx, (student_attn, teacher_attn) in enumerate(
            zip(student_attentions, teacher_attentions)
        ):
            layer_loss = self._compute_layer_loss(
                student_attn, teacher_attn, attention_mask
            )
            total_loss = total_loss + weights[layer_idx] * layer_loss

        return self.alpha * total_loss

    def _compute_layer_loss(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss for a single layer."""
        # Fail loudly if attentions are None
        if student_attn is None:
            raise ValueError(
                "Student attention is None. Model may be using Flash/SDPA attention. "
                "Set use_flash_attention=False in model config."
            )
        if teacher_attn is None:
            raise ValueError(
                "Teacher attention is None. Teacher may be using Flash/SDPA attention. "
                "Load with attn_implementation='eager'."
            )

        # student_attn, teacher_attn: (batch, num_heads, seq_len, seq_len)
        batch_size, num_heads, seq_len, _ = student_attn.shape

        # Add small epsilon for numerical stability
        eps = 1e-10
        student_attn = student_attn.clamp(min=eps)
        teacher_attn = teacher_attn.clamp(min=eps)

        # KL divergence: sum(p * log(p/q))
        # We compute per-head, per-query-position KL divergence
        kl_div = teacher_attn * (torch.log(teacher_attn) - torch.log(student_attn))
        kl_div = kl_div.sum(dim=-1)  # Sum over key positions: (batch, heads, seq)

        if attention_mask is not None:
            # Expand mask: (batch, seq) -> (batch, 1, seq)
            mask = attention_mask.unsqueeze(1).float()
            kl_div = kl_div * mask
            valid_count = mask.sum() * num_heads
            return kl_div.sum() / valid_count.clamp(min=1)
        else:
            valid_count = batch_size * num_heads * seq_len
            return kl_div.sum() / max(valid_count, 1)


class AttentionRelationDistillationLoss(nn.Module):
    """
    Attention relation distillation (from MiniLMv2).

    Instead of distilling attention distributions directly, this distills
    the scaled dot-product attention scores (before softmax), which can
    capture more nuanced relationships.

    L = MSE(A_student / sqrt(d), A_teacher / sqrt(d))

    Where A is the attention score matrix QK^T.

    Args:
        normalize: Whether to normalize scores before computing loss
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        student_scores: list[torch.Tensor],
        teacher_scores: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute attention relation distillation loss.

        Args:
            student_scores: List of attention scores (before softmax) per layer
            teacher_scores: List of attention scores per layer
            attention_mask: Optional attention mask

        Returns:
            MSE loss between normalized attention scores
        """
        total_loss = 0.0
        num_layers = len(student_scores)

        for layer_idx, (student_score, teacher_score) in enumerate(zip(student_scores, teacher_scores)):
            # Fail loudly if attentions are None
            if student_score is None:
                raise ValueError(
                    f"Student attention score at layer {layer_idx} is None. "
                    "Model may be using Flash/SDPA attention. Set use_flash_attention=False."
                )
            if teacher_score is None:
                raise ValueError(
                    f"Teacher attention score at layer {layer_idx} is None. "
                    "Load teacher with attn_implementation='eager'."
                )

            if self.normalize:
                # L2 normalize along key dimension
                student_norm = F.normalize(student_score, p=2, dim=-1)
                teacher_norm = F.normalize(teacher_score, p=2, dim=-1)
            else:
                student_norm = student_score
                teacher_norm = teacher_score

            # MSE loss
            layer_loss = F.mse_loss(student_norm, teacher_norm, reduction="mean")
            total_loss = total_loss + layer_loss

        return total_loss / num_layers if num_layers > 0 else total_loss


class BitDistillAttentionRelationLoss(nn.Module):
    """
    BitDistill attention relation distillation (Equation 11 from arxiv.org/abs/2510.13998).

    Distills relation matrices computed from attention weights:
        R = Softmax(A · Aᵀ / √d_r)

    where A is the attention weight matrix (after softmax).

    Key insight from paper: Distilling at a SINGLE layer provides more
    optimization flexibility than distilling across all layers.

    Args:
        alpha: Scaling coefficient for the loss
        distill_layer: Which layer to distill (-1 = last layer, recommended)
        temperature: Temperature for softmax in relation computation
    """

    def __init__(
        self,
        alpha: float = 1.0,
        distill_layer: int = -1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.distill_layer = distill_layer
        self.temperature = temperature

    def forward(
        self,
        student_attentions: list[torch.Tensor],
        teacher_attentions: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute BitDistill attention relation distillation loss.

        Args:
            student_attentions: List of attention weights per layer, each (B, H, S, S)
            teacher_attentions: List of attention weights per layer, each (B, H, S, S)
            attention_mask: Optional mask (B, S), 1=valid, 0=padding

        Returns:
            KL divergence loss between attention relation matrices
        """
        if len(student_attentions) == 0 or len(teacher_attentions) == 0:
            device = student_attentions[0].device if student_attentions else "cpu"
            return torch.tensor(0.0, device=device)

        # Select single layer (BitDistill: single-layer distillation)
        layer_idx = self.distill_layer
        student_attn = student_attentions[layer_idx]  # (B, H, S, S)
        teacher_attn = teacher_attentions[layer_idx]

        # Fail loudly if attentions are None - this indicates a configuration issue
        if student_attn is None:
            raise ValueError(
                f"Student attention at layer {layer_idx} is None. "
                "This usually means the student model is using Flash/SDPA attention "
                "which doesn't return attention weights. Check model config: use_flash_attention=False "
                "or ensure output_attentions=True is properly passed."
            )
        if teacher_attn is None:
            raise ValueError(
                f"Teacher attention at layer {layer_idx} is None. "
                "This usually means the teacher model is using Flash/SDPA attention. "
                "Load the teacher with attn_implementation='eager' to get attention weights."
            )

        # Compute relation matrices: R = Softmax(A · Aᵀ / √d_r)
        # A @ A^T gives token-to-token relations through attention patterns
        d_r = student_attn.shape[-1]  # sequence length as scaling dimension
        scale = math.sqrt(d_r)

        # Student relations
        student_aat = torch.matmul(student_attn, student_attn.transpose(-2, -1))

        # Teacher relations
        teacher_aat = torch.matmul(teacher_attn, teacher_attn.transpose(-2, -1))

        # Compute Softmax (with optional temperature)
        if self.temperature != 1.0:
            student_R = F.softmax(student_aat / (scale * self.temperature), dim=-1)
            teacher_R = F.softmax(teacher_aat / (scale * self.temperature), dim=-1)
        else:
            student_R = F.softmax(student_aat / scale, dim=-1)
            teacher_R = F.softmax(teacher_aat / scale, dim=-1)

        # Handle head count mismatch: Average over heads to get (B, 1, S, S)
        # This distills the "average" attention structure when architectures differ
        if student_R.shape[1] != teacher_R.shape[1]:
            student_R = student_R.mean(dim=1, keepdim=True)
            teacher_R = teacher_R.mean(dim=1, keepdim=True)

        # Numerical stability
        eps = 1e-10
        student_R = student_R.clamp(min=eps)
        teacher_R = teacher_R.clamp(min=eps)

        # KL divergence: KL(teacher || student) = sum(teacher * log(teacher / student))
        kl = teacher_R * (teacher_R.log() - student_R.log())

        # Sum over key dimension, mean over batch/head/query
        kl = kl.sum(dim=-1)  # (B, H, S)

        if attention_mask is not None:
            # Mask padding positions
            mask = attention_mask.unsqueeze(1).float()  # (B, 1, S)
            kl = kl * mask
            valid_count = mask.sum() * kl.shape[1]  # num_heads
            loss = kl.sum() / valid_count.clamp(min=1)
        else:
            loss = kl.mean()

        return self.alpha * loss


class HiddenStateDistillationLoss(nn.Module):
    """
    Hidden state distillation loss.

    Optionally used to align intermediate representations between
    teacher and student, in addition to attention distillation.

    L = MSE(H_student @ W, H_teacher)

    Where W is an optional projection matrix when dimensions differ.

    Args:
        student_dim: Student hidden dimension
        teacher_dim: Teacher hidden dimension
        use_projection: Whether to project student to teacher dim
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        use_projection: bool = True,
    ):
        super().__init__()
        self.use_projection = use_projection and student_dim != teacher_dim

        if self.use_projection:
            self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.projection = None

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute hidden state distillation loss.

        Args:
            student_hidden: Student hidden states (batch, seq, student_dim)
            teacher_hidden: Teacher hidden states (batch, seq, teacher_dim)
            attention_mask: Optional mask (batch, seq)

        Returns:
            MSE loss between hidden states
        """
        if self.projection is not None:
            student_hidden = self.projection(student_hidden)

        mse = (student_hidden - teacher_hidden).pow(2).mean(dim=-1)

        if attention_mask is not None:
            mse = mse * attention_mask.float()
            return mse.sum() / attention_mask.sum().clamp(min=1)

        return mse.mean()
