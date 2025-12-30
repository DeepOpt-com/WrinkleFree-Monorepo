"""Target Concrete Score (TCS) distillation for DLM students.

Based on:
- Apple's TCSM (ICML 2025): https://machinelearning.apple.com/research/target-concrete
- DDLM: https://openreview.net/forum?id=xfw92pDy2u
- BitDistill (arXiv:2510.13998) for attention relation distillation

Key differences from BitDistillLoss:
1. NO logit shifting - DLM operates on masked positions, not next-token prediction
2. Top-K TCS estimation - sparse distribution matching for efficiency
3. Block-wise attention distillation - matches attention only within blocks where
   both AR teacher and DLM student can use bidirectional attention
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCSDistillLoss(nn.Module):
    """
    Target Concrete Score (TCS) distillation for DLM students.
    Includes block-wise attention distillation for AR→DLM alignment.

    L = L_CE + lambda_tcs * L_TCS + gamma_attn * L_BlockAttn

    Where:
    - L_CE: Cross-entropy on ground truth labels (no shifting!)
    - L_TCS: KL(softmax(teacher_topk/T) || softmax(student[topk_indices]/T)) * T^2
    - L_BlockAttn: Block-wise attention relation distillation (A·Aᵀ)

    Key differences from AR distillation:
    - NO logit shifting (DLM predicts masked tokens, not next tokens)
    - Top-K estimation for computational efficiency
    - Block-wise attention matching (only within bd_size blocks)

    Args:
        lambda_tcs: Weight for TCS logits distillation loss
        gamma_attention: Weight for block-wise attention distillation loss
        temperature: Temperature for KL divergence
        top_k: Number of top tokens for sparse TCS estimation
        block_size: Block size for attention matching (default: 32 from Fast-dLLM v2)
        ignore_index: Index to ignore in CE loss (typically -100)
        distill_layer: Layer for attention distillation (-1 = last)
    """

    def __init__(
        self,
        lambda_tcs: float = 10.0,
        gamma_attention: float = 1e-5,
        temperature: float = 5.0,
        top_k: int = 100,
        block_size: int = 32,
        ignore_index: int = -100,
        distill_layer: int = -1,
    ):
        super().__init__()
        self.lambda_tcs = lambda_tcs
        self.gamma_attention = gamma_attention
        self.temperature = temperature
        self.top_k = top_k
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.distill_layer = distill_layer
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_attentions: list[torch.Tensor] | None = None,
        teacher_attentions: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute TCS distillation loss.

        Args:
            student_logits: Student model logits (batch, seq, vocab)
            teacher_logits: Teacher model logits (batch, seq, vocab)
            labels: Ground truth labels for CE loss (batch, seq)
            student_attentions: List of student attention weights per layer
            teacher_attentions: List of teacher attention weights per layer
            attention_mask: Optional attention mask (batch, seq)

        Returns:
            Dictionary containing:
                - loss: Total combined loss
                - ce_loss: Cross-entropy component
                - tcs_loss: TCS logits distillation component
                - attention_loss: Block-wise attention distillation component
        """
        # Response mask: positions where we compute loss (not -100)
        response_mask = labels != self.ignore_index

        # Cross-entropy loss - NO SHIFT for DLM!
        # DLM predicts the masked token at each position, not the next token
        ce = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
        )

        # TCS with Top-K estimation
        tcs_loss = self._compute_tcs_loss(
            student_logits, teacher_logits, response_mask
        )

        # Block-wise attention distillation
        attn_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
        if (
            self.gamma_attention > 0
            and student_attentions is not None
            and teacher_attentions is not None
            and len(student_attentions) > 0
            and len(teacher_attentions) > 0
        ):
            attn_loss = self._compute_block_attention_loss(
                student_attentions, teacher_attentions, response_mask
            )

        # Combined loss
        total = ce + self.lambda_tcs * tcs_loss + self.gamma_attention * attn_loss

        return {
            "loss": total,
            "ce_loss": ce,
            "tcs_loss": tcs_loss,
            "attention_loss": attn_loss,
        }

    def _compute_tcs_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Target Concrete Score loss with Top-K estimation.

        For each position, we:
        1. Get teacher's top-K logits and indices
        2. Gather student's logits at those indices
        3. Compute KL divergence on the sparse distributions

        This is computationally efficient while capturing most of the distribution.
        """
        batch_size, seq_len, vocab_size = student_logits.shape

        # Get teacher's top-K predictions
        # Shape: (batch, seq, top_k)
        teacher_topk_logits, topk_indices = torch.topk(
            teacher_logits, k=min(self.top_k, vocab_size), dim=-1
        )

        # Gather student logits at the same indices
        # Shape: (batch, seq, top_k)
        student_topk_logits = torch.gather(
            student_logits, dim=-1, index=topk_indices
        )

        # Temperature-scaled probabilities
        teacher_probs = F.softmax(teacher_topk_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_topk_logits / self.temperature, dim=-1)

        # KL divergence: sum_i p_teacher * (log p_teacher - log p_student)
        # Using F.kl_div with log_target=False (default): expects log(q) and p
        kl = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)

        # Mask to response positions only
        kl = kl * response_mask.float()
        num_valid = response_mask.sum().clamp(min=1)

        # Temperature^2 scaling (standard for distillation)
        return (kl.sum() / num_valid) * (self.temperature ** 2)

    def _compute_block_attention_loss(
        self,
        student_attentions: list[torch.Tensor],
        teacher_attentions: list[torch.Tensor],
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute block-wise attention relation loss for AR→DLM distillation.

        Fast-dLLM v2 uses block-causal attention:
        - Tokens can see all tokens WITHIN their block (bidirectional)
        - Tokens can see all tokens in PREVIOUS blocks (causal)

        Key insight: Within each block, both teacher and student CAN use
        bidirectional attention. We match attention patterns ONLY within blocks.

        Uses BitDistill-style attention relation distillation:
            R = Softmax(A · Aᵀ / √d_r)
        But restricted to block-diagonal regions.

        Args:
            student_attentions: List of attention weights per layer, each (B, H, S, S)
            teacher_attentions: List of attention weights per layer, each (B, H, S, S)
            response_mask: Mask for response positions (B, S)

        Returns:
            Block-wise attention distillation loss
        """
        # Select single layer (BitDistill: single-layer distillation)
        layer_idx = self.distill_layer
        student_attn = student_attentions[layer_idx]  # (B, H, S, S)
        teacher_attn = teacher_attentions[layer_idx]

        # Fail loudly if attentions are None
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

        batch_size, num_heads, seq_len, _ = student_attn.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        total_loss = 0.0
        num_valid_blocks = 0

        for b in range(num_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, seq_len)

            # Extract block-diagonal attention submatrix
            # Shape: (batch, heads, block_len, block_len)
            s_block = student_attn[:, :, start:end, start:end]
            t_block = teacher_attn[:, :, start:end, start:end]

            # Check if block has response tokens
            block_mask = response_mask[:, start:end]  # (batch, block_len)
            if block_mask.sum() == 0:
                continue

            # Compute attention relations: R = Softmax(A·Aᵀ / √d)
            # This captures token-to-token relations through attention patterns
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

            # MSE loss on relation matrices (per BitDistill Eq. 11)
            # Using MSE rather than KL for numerical stability
            block_loss = F.mse_loss(s_rel, t_rel, reduction='none')

            # Mask to response positions within block
            # Create 2D mask: (batch, block_len, block_len)
            mask_2d = block_mask.unsqueeze(-1) & block_mask.unsqueeze(-2)
            mask_2d = mask_2d.unsqueeze(1).float()  # (batch, 1, block_len, block_len)

            # Apply mask and normalize
            masked_loss = (block_loss * mask_2d).sum()
            num_masked = mask_2d.sum().clamp(min=1)
            block_loss = masked_loss / num_masked

            total_loss = total_loss + block_loss
            num_valid_blocks += 1

        if num_valid_blocks == 0:
            return torch.tensor(0.0, device=student_attn.device, dtype=student_attn.dtype)

        return total_loss / num_valid_blocks


class LogitsOnlyTCSLoss(nn.Module):
    """
    TCS distillation with logits only (no attention distillation).

    Use this when:
    - Teacher doesn't return attention weights (e.g., VLLM backend)
    - You want faster training without attention overhead
    - Student and teacher have incompatible attention architectures

    L = L_CE + lambda_tcs * L_TCS
    """

    def __init__(
        self,
        lambda_tcs: float = 10.0,
        temperature: float = 5.0,
        top_k: int = 100,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.lambda_tcs = lambda_tcs
        self.temperature = temperature
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,  # Ignore attention arguments
    ) -> dict[str, torch.Tensor]:
        """Compute TCS loss without attention distillation."""
        response_mask = labels != self.ignore_index

        # Cross-entropy loss - NO SHIFT for DLM!
        ce = self.ce_loss(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
        )

        # TCS with Top-K estimation
        batch_size, seq_len, vocab_size = student_logits.shape
        teacher_topk_logits, topk_indices = torch.topk(
            teacher_logits, k=min(self.top_k, vocab_size), dim=-1
        )
        student_topk_logits = torch.gather(
            student_logits, dim=-1, index=topk_indices
        )

        teacher_probs = F.softmax(teacher_topk_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_topk_logits / self.temperature, dim=-1)

        kl = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
        kl = kl * response_mask.float()
        num_valid = response_mask.sum().clamp(min=1)
        tcs_loss = (kl.sum() / num_valid) * (self.temperature ** 2)

        total = ce + self.lambda_tcs * tcs_loss

        return {
            "loss": total,
            "ce_loss": ce,
            "tcs_loss": tcs_loss,
            "attention_loss": torch.tensor(0.0, device=student_logits.device),
        }
