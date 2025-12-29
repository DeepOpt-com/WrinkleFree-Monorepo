"""Tests for distillation losses that remain in training package.

NOTE: Stage 3 distillation (BitDistillLoss, LogitsDistillationLoss, AttentionDistillationLoss)
has been moved to the separate `distillation` package. See packages/distillation/tests/ for those tests.

This file tests:
- ContinuePretrainLoss (Stage 2)
- LayerwiseDistillationLoss (Stage 1.9)
"""

import pytest
import torch

from wrinklefree.distillation import (
    ContinuePretrainLoss,
    LayerwiseDistillationLoss,
    LayerwiseLossType,
)


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

    def test_loss_is_non_negative(self):
        """Test that loss is non-negative."""
        loss_fn = ContinuePretrainLoss()

        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))

        result = loss_fn(logits, labels)

        assert result["loss"].item() >= 0

    def test_ignore_index(self):
        """Test that ignore_index is respected."""
        loss_fn = ContinuePretrainLoss(ignore_index=-100)

        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))
        labels[:, -3:] = -100  # Last 3 tokens are padding

        result = loss_fn(logits, labels)

        assert not torch.isnan(result["loss"])


class TestLayerwiseDistillationLoss:
    """Tests for layer-wise distillation loss (Stage 1.9)."""

    def test_forward_shape(self):
        """Test that loss is a scalar."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE)

        batch_size, seq_len, hidden_dim = 2, 10, 256
        student_hidden = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(4)]
        teacher_hidden = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(4)]

        result = loss_fn(student_hidden, teacher_hidden)

        assert "loss" in result
        assert result["loss"].dim() == 0

    def test_same_hidden_low_loss(self):
        """Test that identical hidden states give low loss."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE)

        hidden = [torch.randn(2, 10, 256) for _ in range(4)]

        result = loss_fn(hidden, [h.clone() for h in hidden])

        assert result["loss"].item() < 1e-5

    def test_mse_normalized_loss(self):
        """Test MSE normalized loss type."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        student = [torch.randn(2, 10, 256) for _ in range(4)]
        teacher = [torch.randn(2, 10, 256) for _ in range(4)]

        result = loss_fn(student, teacher)

        assert "loss" in result
        assert result["loss"].dim() == 0
        assert not torch.isnan(result["loss"])
