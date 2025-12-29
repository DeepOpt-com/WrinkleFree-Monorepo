"""Tests for distillation loss functions."""

import pytest
import torch

from distillation.losses import (
    BitDistillLoss,
    ClassificationDistillLoss,
    LogitsDistillationLoss,
    SoftTargetCrossEntropy,
)


class TestLogitsDistillationLoss:
    """Tests for LogitsDistillationLoss."""

    def test_basic_loss_computation(self):
        """Test basic KL divergence computation."""
        loss_fn = LogitsDistillationLoss(temperature=5.0)

        batch_size, seq_len, vocab_size = 2, 10, 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        loss = loss_fn(student_logits, teacher_logits)

        assert loss.ndim == 0  # scalar
        assert loss >= 0  # KL divergence is non-negative

    def test_identical_logits_zero_loss(self):
        """Test that identical logits produce zero loss."""
        loss_fn = LogitsDistillationLoss(temperature=1.0)

        logits = torch.randn(2, 5, 100)
        loss = loss_fn(logits, logits.clone())

        assert loss.item() < 1e-5  # Should be ~0

    def test_temperature_scaling(self):
        """Test that higher temperature produces different loss."""
        student = torch.randn(2, 5, 100)
        teacher = torch.randn(2, 5, 100)

        loss_t1 = LogitsDistillationLoss(temperature=1.0)(student, teacher)
        loss_t5 = LogitsDistillationLoss(temperature=5.0)(student, teacher)

        # Different temperatures should give different losses
        assert not torch.allclose(loss_t1, loss_t5)


class TestSoftTargetCrossEntropy:
    """Tests for SoftTargetCrossEntropy."""

    def test_basic_loss_computation(self):
        """Test basic soft target cross-entropy."""
        loss_fn = SoftTargetCrossEntropy(temperature=5.0)

        student = torch.randn(2, 10, 100)
        teacher = torch.randn(2, 10, 100)

        loss = loss_fn(student, teacher)

        assert loss.ndim == 0
        assert loss >= 0


class TestBitDistillLoss:
    """Tests for BitDistillLoss."""

    def test_basic_loss_computation(self):
        """Test combined BitDistill loss."""
        loss_fn = BitDistillLoss(
            lambda_logits=10.0,
            gamma_attention=1e-5,
            temperature=5.0,
        )

        batch_size, seq_len, vocab_size = 2, 10, 1000
        num_heads, num_layers = 8, 12

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        student_attentions = [
            torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
            for _ in range(num_layers)
        ]
        teacher_attentions = [
            torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
            for _ in range(num_layers)
        ]
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_attentions=student_attentions,
            teacher_attentions=teacher_attentions,
            labels=labels,
        )

        assert "loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "logits_distill_loss" in loss_dict
        assert "attention_distill_loss" in loss_dict
        assert loss_dict["loss"] >= 0

    def test_logits_only_mode(self):
        """Test with gamma_attention=0 (logits only)."""
        loss_fn = BitDistillLoss(
            lambda_logits=10.0,
            gamma_attention=0.0,  # No attention distillation
            temperature=5.0,
        )

        batch_size, seq_len, vocab_size = 2, 10, 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_attentions=None,  # No attentions needed
            teacher_attentions=None,
            labels=labels,
        )

        assert loss_dict["attention_distill_loss"].item() == 0.0


class TestClassificationDistillLoss:
    """Tests for ClassificationDistillLoss."""

    def test_basic_loss_computation(self):
        """Test classification distillation loss."""
        loss_fn = ClassificationDistillLoss(
            lambda_logits=10.0,
            temperature=5.0,
            num_labels=2,
        )

        batch_size, num_labels = 4, 2
        student_logits = torch.randn(batch_size, num_labels)
        teacher_logits = torch.randn(batch_size, num_labels)
        labels = torch.randint(0, num_labels, (batch_size,))

        loss_dict = loss_fn(student_logits, teacher_logits, labels)

        assert "loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "logits_distill_loss" in loss_dict
        assert loss_dict["loss"] >= 0
