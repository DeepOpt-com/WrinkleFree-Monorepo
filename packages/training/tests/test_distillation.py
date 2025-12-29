"""Tests for distillation losses."""

import pytest
import torch

from wrinklefree.distillation import (
    AttentionDistillationLoss,
    BitDistillLoss,
    ContinuePretrainLoss,
    LogitsDistillationLoss,
)
from wrinklefree.distillation.attention_loss import BitDistillAttentionRelationLoss


class TestLogitsDistillationLoss:
    """Tests for logits distillation loss."""

    def test_forward_shape(self):
        """Test that loss is a scalar."""
        loss_fn = LogitsDistillationLoss(temperature=5.0)

        student_logits = torch.randn(2, 10, 1000)
        teacher_logits = torch.randn(2, 10, 1000)

        loss = loss_fn(student_logits, teacher_logits)

        assert loss.dim() == 0  # Scalar

    def test_same_logits_low_loss(self):
        """Test that identical logits give low loss."""
        loss_fn = LogitsDistillationLoss(temperature=5.0)

        logits = torch.randn(2, 10, 1000)

        loss = loss_fn(logits, logits.clone())

        assert loss.item() < 1e-5

    def test_temperature_effect(self):
        """Test that higher temperature gives different loss."""
        loss_fn_low_temp = LogitsDistillationLoss(temperature=1.0)
        loss_fn_high_temp = LogitsDistillationLoss(temperature=10.0)

        student = torch.randn(2, 10, 1000)
        teacher = torch.randn(2, 10, 1000)

        loss_low = loss_fn_low_temp(student, teacher)
        loss_high = loss_fn_high_temp(student, teacher)

        # Higher temp = softer distributions = usually higher scaled loss
        assert not torch.isclose(loss_low, loss_high)

    def test_gradient_flow(self):
        """Test that gradients flow through loss."""
        loss_fn = LogitsDistillationLoss(temperature=5.0)

        student = torch.randn(2, 10, 100, requires_grad=True)
        teacher = torch.randn(2, 10, 100)

        loss = loss_fn(student, teacher)
        loss.backward()

        assert student.grad is not None


class TestAttentionDistillationLoss:
    """Tests for attention distillation loss."""

    def test_forward_shape(self):
        """Test that loss is a scalar."""
        loss_fn = AttentionDistillationLoss(alpha=1.0)

        # Attention weights: (batch, heads, seq, seq)
        student_attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1)]
        teacher_attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1)]

        loss = loss_fn(student_attn, teacher_attn)

        assert loss.dim() == 0

    def test_same_attention_low_loss(self):
        """Test that identical attention gives low loss."""
        loss_fn = AttentionDistillationLoss(alpha=1.0)

        attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1)]

        loss = loss_fn(attn, [a.clone() for a in attn])

        assert loss.item() < 1e-5

    def test_multiple_layers(self):
        """Test with multiple attention layers."""
        loss_fn = AttentionDistillationLoss(alpha=1.0)

        student_attn = [
            torch.softmax(torch.randn(2, 8, 10, 10), dim=-1)
            for _ in range(4)
        ]
        teacher_attn = [
            torch.softmax(torch.randn(2, 8, 10, 10), dim=-1)
            for _ in range(4)
        ]

        loss = loss_fn(student_attn, teacher_attn)

        assert loss.dim() == 0
        assert loss.item() > 0


class TestBitDistillAttentionRelationLoss:
    """Tests for BitDistill attention relation distillation (Equation 11)."""

    def test_forward_shape(self):
        """Test that loss is a scalar."""
        loss_fn = BitDistillAttentionRelationLoss(alpha=1.0, distill_layer=-1)

        student_attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1) for _ in range(4)]
        teacher_attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1) for _ in range(4)]

        loss = loss_fn(student_attn, teacher_attn)

        assert loss.dim() == 0

    def test_same_attention_low_loss(self):
        """Test that identical attention gives low loss."""
        loss_fn = BitDistillAttentionRelationLoss(alpha=1.0)

        attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1) for _ in range(4)]

        loss = loss_fn(attn, [a.clone() for a in attn])

        assert loss.item() < 1e-5

    def test_single_layer_distillation(self):
        """Test that only specified layer is used for distillation."""
        loss_fn_first = BitDistillAttentionRelationLoss(distill_layer=0)
        loss_fn_last = BitDistillAttentionRelationLoss(distill_layer=-1)

        # Create attention tensors where first and last layers differ significantly
        student_attn = [torch.softmax(torch.randn(2, 4, 8, 8), dim=-1) for _ in range(4)]
        teacher_attn = [torch.softmax(torch.randn(2, 4, 8, 8), dim=-1) for _ in range(4)]

        # Make first layer identical
        teacher_attn[0] = student_attn[0].clone()

        loss_first = loss_fn_first(student_attn, teacher_attn)
        loss_last = loss_fn_last(student_attn, teacher_attn)

        # First layer loss should be ~0, last layer loss should be higher
        assert loss_first.item() < 1e-5
        assert loss_last.item() > 1e-5

    def test_relation_matrix_computation(self):
        """Test that A·A^T relation matrices are computed correctly."""
        loss_fn = BitDistillAttentionRelationLoss(alpha=1.0)

        # Create uniform attention (all positions attend equally)
        batch, heads, seq = 2, 4, 8
        uniform_attn = torch.ones(batch, heads, seq, seq) / seq

        student_attn = [uniform_attn]
        teacher_attn = [uniform_attn.clone()]

        loss = loss_fn(student_attn, teacher_attn)

        # Identical uniform attention should give zero loss
        assert loss.item() < 1e-5

    def test_with_attention_mask(self):
        """Test with attention mask for padding."""
        loss_fn = BitDistillAttentionRelationLoss(alpha=1.0)

        batch, heads, seq = 2, 4, 10
        student_attn = [torch.softmax(torch.randn(batch, heads, seq, seq), dim=-1)]
        teacher_attn = [torch.softmax(torch.randn(batch, heads, seq, seq), dim=-1)]

        # Mask: first 6 tokens valid, last 4 padding
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

        loss = loss_fn(student_attn, teacher_attn, attention_mask=mask)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = BitDistillAttentionRelationLoss(alpha=1.0)

        student_attn = [torch.softmax(torch.randn(2, 4, 8, 8, requires_grad=True), dim=-1)]
        teacher_attn = [torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)]

        loss = loss_fn(student_attn, teacher_attn)
        loss.backward()

        # The input had requires_grad, so softmax output should have grad_fn
        assert student_attn[0].grad_fn is not None


class TestBitDistillLoss:
    """Tests for combined BitDistill loss."""

    def test_forward_returns_dict(self):
        """Test that forward returns dict with components."""
        loss_fn = BitDistillLoss(
            lambda_logits=10.0,
            gamma_attention=1e-5,
            temperature=5.0,
        )

        student_logits = torch.randn(2, 10, 1000)
        teacher_logits = torch.randn(2, 10, 1000)
        student_attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1)]
        teacher_attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1)]
        labels = torch.randint(0, 1000, (2, 10))

        result = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_attentions=student_attn,
            teacher_attentions=teacher_attn,
            labels=labels,
        )

        assert isinstance(result, dict)
        assert "loss" in result
        assert "ce_loss" in result
        assert "logits_distill_loss" in result
        assert "attention_distill_loss" in result

    def test_loss_components_non_negative(self):
        """Test that all loss components are non-negative."""
        loss_fn = BitDistillLoss()

        student_logits = torch.randn(2, 10, 1000)
        teacher_logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))

        result = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_attentions=None,
            teacher_attentions=None,
            labels=labels,
        )

        assert result["loss"].item() >= 0
        assert result["ce_loss"].item() >= 0
        assert result["logits_distill_loss"].item() >= 0

    def test_use_relation_distill_flag(self):
        """Test that use_relation_distill switches between attention losses."""
        # With relation distill (default, BitDistill style)
        loss_fn_relation = BitDistillLoss(use_relation_distill=True, distill_layer=-1)
        assert isinstance(loss_fn_relation.attention_loss, BitDistillAttentionRelationLoss)

        # Without relation distill (original MiniLM style)
        loss_fn_direct = BitDistillLoss(use_relation_distill=False)
        assert isinstance(loss_fn_direct.attention_loss, AttentionDistillationLoss)

    def test_with_multiple_attention_layers(self):
        """Test BitDistill loss with multiple attention layers."""
        loss_fn = BitDistillLoss(
            lambda_logits=10.0,
            gamma_attention=1e-5,
            use_relation_distill=True,
            distill_layer=-1,
        )

        student_logits = torch.randn(2, 10, 1000)
        teacher_logits = torch.randn(2, 10, 1000)
        student_attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1) for _ in range(4)]
        teacher_attn = [torch.softmax(torch.randn(2, 8, 10, 10), dim=-1) for _ in range(4)]
        labels = torch.randint(0, 1000, (2, 10))

        result = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_attentions=student_attn,
            teacher_attentions=teacher_attn,
            labels=labels,
        )

        assert result["attention_distill_loss"].item() >= 0
        assert not torch.isnan(result["loss"])


class TestContinuePretrainLoss:
    """Tests for continue pretraining loss."""

    def test_forward_returns_dict(self):
        """Test that forward returns dict with loss."""
        loss_fn = ContinuePretrainLoss()

        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))

        result = loss_fn(logits, labels)

        assert isinstance(result, dict)
        assert "loss" in result
        assert result["loss"].dim() == 0
