"""Block-wise attention distillation objective for AR->DLM transfer.

Based on Fast-dLLM v2 block-causal attention pattern:
- Tokens can see all tokens WITHIN their block (bidirectional)
- Tokens can see all tokens in PREVIOUS blocks (causal)

Within each block, both teacher and student CAN use bidirectional attention,
so we match attention patterns ONLY within blocks.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from wrinklefree.objectives.base import Objective, ObjectiveOutput


class BlockAttentionDistillationObjective(Objective):
    """
    Block-wise attention distillation for AR->DLM transfer.

    Only matches attention patterns WITHIN blocks where both
    AR teacher and DLM student use bidirectional attention.

    Uses BitDistill-style attention relation distillation:
        R = Softmax(A · A^T / sqrt(d_r))
    But restricted to block-diagonal regions.

    Args:
        block_size: Block size for attention matching (default: 32 from Fast-dLLM v2)
        distill_layer: Layer for distillation (-1 = last)
        ignore_index: Index to ignore in labels
    """

    requires_teacher = True
    requires_hidden_states = False
    requires_attentions = True
    modifies_input = False

    def __init__(
        self,
        block_size: int = 32,
        distill_layer: int = -1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.block_size = block_size
        self.distill_layer = distill_layer
        self.ignore_index = ignore_index

    @property
    def name(self) -> str:
        return "block_attention_distill"

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """
        Compute block-wise attention relation loss.

        Args:
            model_outputs: Must contain 'attentions' (list of tensors per layer)
            batch: Must contain 'labels' or 'dlm_labels' for masking
            teacher_outputs: Must contain 'attentions'

        Returns:
            ObjectiveOutput with block attention loss
        """
        if teacher_outputs is None:
            raise ValueError("BlockAttentionDistillationObjective requires teacher_outputs")

        student_attentions = model_outputs.get("attentions")
        teacher_attentions = teacher_outputs.get("attentions")

        if student_attentions is None or len(student_attentions) == 0:
            raise ValueError(
                "Student model must return attention weights. "
                "Use output_attentions=True and eager attention."
            )
        if teacher_attentions is None or len(teacher_attentions) == 0:
            raise ValueError(
                "Teacher model must return attention weights. "
                "Load teacher with attn_implementation='eager'."
            )

        # Select single layer
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

        # Get response mask
        labels = batch.get("dlm_labels", batch["labels"])
        response_mask = labels != self.ignore_index

        batch_size, num_heads, seq_len, _ = student_attn.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        total_loss = torch.tensor(0.0, device=student_attn.device, dtype=student_attn.dtype)
        num_valid_blocks = 0

        for b in range(num_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, seq_len)

            # Extract block-diagonal attention submatrix
            s_block = student_attn[:, :, start:end, start:end]
            t_block = teacher_attn[:, :, start:end, start:end]

            # Check if block has response tokens
            block_mask = response_mask[:, start:end]  # (batch, block_len)
            if block_mask.sum() == 0:
                continue

            # Compute attention relations: R = Softmax(A·A^T / sqrt(d))
            d_r = s_block.size(-1)
            scale = math.sqrt(d_r)

            # Student relations
            s_aat = torch.matmul(s_block, s_block.transpose(-2, -1))
            s_rel = F.softmax(s_aat / scale, dim=-1)

            # Teacher relations
            t_aat = torch.matmul(t_block, t_block.transpose(-2, -1))
            t_rel = F.softmax(t_aat / scale, dim=-1)

            # Handle head count mismatch: Average over heads
            if s_rel.shape[1] != t_rel.shape[1]:
                s_rel = s_rel.mean(dim=1, keepdim=True)
                t_rel = t_rel.mean(dim=1, keepdim=True)

            # MSE loss on relation matrices
            block_loss = F.mse_loss(s_rel, t_rel, reduction="none")

            # Mask to response positions within block
            mask_2d = block_mask.unsqueeze(-1) & block_mask.unsqueeze(-2)
            mask_2d = mask_2d.unsqueeze(1).float()  # (batch, 1, block_len, block_len)

            # Apply mask and normalize
            masked_loss = (block_loss * mask_2d).sum()
            num_masked = mask_2d.sum().clamp(min=1)
            block_loss = masked_loss / num_masked

            total_loss = total_loss + block_loss
            num_valid_blocks += 1

        if num_valid_blocks == 0:
            loss = torch.tensor(0.0, device=student_attn.device, dtype=student_attn.dtype)
        else:
            loss = total_loss / num_valid_blocks

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "block_attention_loss": loss.detach(),
                "num_valid_blocks": torch.tensor(float(num_valid_blocks), device=loss.device),
            },
        )
