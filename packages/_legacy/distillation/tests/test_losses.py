"""Tests for distillation loss functions."""

import pytest
import torch

from distillation.losses import (
    BitDistillLoss,
    ClassificationDistillLoss,
    LogitsDistillationLoss,
    LogitsOnlyTCSLoss,
    SoftTargetCrossEntropy,
    TCSDistillLoss,
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


class TestTCSDistillLoss:
    """Tests for TCSDistillLoss (DLM distillation)."""

    def test_basic_loss_computation(self):
        """Test combined TCS loss for DLM students."""
        loss_fn = TCSDistillLoss(
            lambda_tcs=10.0,
            gamma_attention=1e-5,
            temperature=5.0,
            top_k=50,
            block_size=32,
        )

        batch_size, seq_len, vocab_size = 2, 64, 1000
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
        # Labels with some masked positions (like DLM training)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, :10] = -100  # Mask first 10 positions (prompt)

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_attentions=student_attentions,
            teacher_attentions=teacher_attentions,
        )

        assert "loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "tcs_loss" in loss_dict
        assert "attention_loss" in loss_dict
        assert loss_dict["loss"] >= 0
        assert loss_dict["tcs_loss"] >= 0
        assert loss_dict["attention_loss"] >= 0

    def test_no_logit_shifting(self):
        """Test that TCS loss does NOT shift logits (key difference from BitDistill).

        DLM models predict masked tokens at each position, not next tokens.
        So labels should align directly with logits without shifting.
        """
        loss_fn = TCSDistillLoss(
            lambda_tcs=10.0,
            gamma_attention=0.0,  # Disable attention for this test
            temperature=5.0,
        )

        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create logits where position i strongly predicts token i
        student_logits = torch.zeros(batch_size, seq_len, vocab_size)
        for i in range(seq_len):
            student_logits[:, i, i % vocab_size] = 10.0  # High logit for target

        teacher_logits = student_logits.clone()
        labels = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1) % vocab_size

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_attentions=None,
            teacher_attentions=None,
        )

        # CE loss should be very low since predictions match labels at same positions
        assert loss_dict["ce_loss"].item() < 1.0

    def test_top_k_estimation(self):
        """Test that top-k estimation works correctly."""
        loss_fn = TCSDistillLoss(
            lambda_tcs=10.0,
            gamma_attention=0.0,
            temperature=5.0,
            top_k=10,  # Small k for testing
        )

        batch_size, seq_len, vocab_size = 2, 5, 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_attentions=None,
            teacher_attentions=None,
        )

        assert loss_dict["tcs_loss"] >= 0
        assert loss_dict["tcs_loss"].isfinite()

    def test_block_wise_attention(self):
        """Test block-wise attention distillation."""
        block_size = 16
        loss_fn = TCSDistillLoss(
            lambda_tcs=0.0,  # Disable TCS for this test
            gamma_attention=1.0,
            temperature=5.0,
            block_size=block_size,
        )

        batch_size, seq_len, vocab_size = 2, 64, 100
        num_heads, num_layers = 4, 12

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create attention patterns that differ
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
            labels=labels,
            student_attentions=student_attentions,
            teacher_attentions=teacher_attentions,
        )

        assert loss_dict["attention_loss"] > 0  # Should have some loss

    def test_identical_attentions_low_loss(self):
        """Test that identical attentions produce low attention loss."""
        loss_fn = TCSDistillLoss(
            lambda_tcs=0.0,
            gamma_attention=1.0,
            temperature=5.0,
            block_size=8,
        )

        batch_size, seq_len, vocab_size = 2, 16, 100
        num_heads, num_layers = 4, 1

        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        # Use identical attention patterns
        shared_attentions = [
            torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
            for _ in range(num_layers)
        ]

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_attentions=shared_attentions,
            teacher_attentions=[a.clone() for a in shared_attentions],
        )

        # With identical attentions, attention loss should be ~0
        assert loss_dict["attention_loss"].item() < 1e-5

    def test_response_mask_handling(self):
        """Test that masked positions (label=-100) are handled correctly."""
        loss_fn = TCSDistillLoss(
            lambda_tcs=10.0,
            gamma_attention=0.0,
            temperature=5.0,
        )

        batch_size, seq_len, vocab_size = 2, 20, 100
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        # All positions masked - should have zero TCS loss
        labels = torch.full((batch_size, seq_len), -100)

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_attentions=None,
            teacher_attentions=None,
        )

        # TCS loss computed only on non-masked positions
        # With all masked, the denominator is clamped to 1, so loss should still be finite
        assert loss_dict["tcs_loss"].isfinite()


class TestLogitsOnlyTCSLoss:
    """Tests for LogitsOnlyTCSLoss."""

    def test_basic_loss_computation(self):
        """Test TCS loss without attention distillation."""
        loss_fn = LogitsOnlyTCSLoss(
            lambda_tcs=10.0,
            temperature=5.0,
            top_k=50,
        )

        batch_size, seq_len, vocab_size = 2, 10, 1000
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
        )

        assert "loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "tcs_loss" in loss_dict
        assert "attention_loss" in loss_dict
        assert loss_dict["attention_loss"].item() == 0.0  # No attention loss

    def test_ignores_attention_args(self):
        """Test that attention arguments are safely ignored."""
        loss_fn = LogitsOnlyTCSLoss(lambda_tcs=10.0, temperature=5.0)

        batch_size, seq_len, vocab_size = 2, 5, 100
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Pass attention args that should be ignored
        loss_dict = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_attentions="should be ignored",
            teacher_attentions="should be ignored",
        )

        assert loss_dict["attention_loss"].item() == 0.0
