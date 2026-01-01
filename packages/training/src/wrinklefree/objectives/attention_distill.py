"""Attention relation distillation objective (BitDistill Equation 11).

Based on BitDistill: arxiv.org/abs/2510.13998
Distills relation matrices: R = Softmax(A · A^T / sqrt(d_r))
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from wrinklefree.objectives.base import Objective, ObjectiveOutput


class AttentionRelationDistillationObjective(Objective):
    """
    BitDistill attention relation distillation (Equation 11).

    Distills relation matrices computed from attention weights:
        R = Softmax(A · A^T / sqrt(d_r))

    where A is the attention weight matrix (after softmax).

    Key insight from paper: Distilling at a SINGLE layer provides more
    optimization flexibility than distilling across all layers.

    Args:
        distill_layer: Which layer to distill (-1 = last layer, recommended)
        temperature: Temperature for softmax in relation computation
        ignore_index: Index to ignore in labels (for masking)
    """

    requires_teacher = True
    requires_hidden_states = False
    requires_attentions = True
    modifies_input = False

    def __init__(
        self,
        distill_layer: int = -1,
        temperature: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.distill_layer = distill_layer
        self.temperature = temperature
        self.ignore_index = ignore_index

    @property
    def name(self) -> str:
        return "attention_distill"

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """
        Compute BitDistill attention relation distillation loss.

        Args:
            model_outputs: Must contain 'attentions' (list of tensors per layer)
            batch: Must contain 'labels' or 'attention_mask' for masking
            teacher_outputs: Must contain 'attentions'

        Returns:
            ObjectiveOutput with attention relation loss
        """
        if teacher_outputs is None:
            raise ValueError("AttentionRelationDistillationObjective requires teacher_outputs")

        student_attentions = model_outputs.get("attentions")
        teacher_attentions = teacher_outputs.get("attentions")

        if student_attentions is None or len(student_attentions) == 0:
            raise ValueError(
                "Student model must return attention weights. "
                "Use output_attentions=True and eager attention implementation."
            )
        if teacher_attentions is None or len(teacher_attentions) == 0:
            raise ValueError(
                "Teacher model must return attention weights. "
                "Load teacher with attn_implementation='eager'."
            )

        # Select single layer (BitDistill: single-layer distillation)
        layer_idx = self.distill_layer
        student_attn = student_attentions[layer_idx]  # (B, H, S, S)
        teacher_attn = teacher_attentions[layer_idx]

        if student_attn is None:
            raise ValueError(
                f"Student attention at layer {layer_idx} is None. "
                "Ensure output_attentions=True and use eager attention."
            )
        if teacher_attn is None:
            raise ValueError(
                f"Teacher attention at layer {layer_idx} is None. "
                "Load teacher with attn_implementation='eager'."
            )

        # Get attention mask
        attention_mask = batch.get("attention_mask")

        # Compute relation matrices: R = Softmax(A · A^T / sqrt(d_r))
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

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "attention_kl": loss.detach(),
                "distill_layer": torch.tensor(layer_idx % len(student_attentions), device=loss.device),
            },
        )
