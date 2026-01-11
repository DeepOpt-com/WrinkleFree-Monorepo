"""Tests for Stage 1.9 layer-wise distillation.

Tests the LayerwiseDistillationLoss class with various loss types
and layer weighting strategies.
"""

import pytest
import torch

from wf_train.distillation import (
    LayerwiseDistillationLoss,
    LayerwiseLossType,
)


class TestLayerwiseLossType:
    """Tests for LayerwiseLossType enum."""

    def test_all_types_valid(self):
        """Test that all loss types can be instantiated."""
        for loss_type in LayerwiseLossType:
            assert loss_type.value in [
                "mse",
                "mse_normalized",
                "kl",
                "inner_product",
            ]

    def test_string_conversion(self):
        """Test string to enum conversion."""
        assert LayerwiseLossType("mse") == LayerwiseLossType.MSE
        assert LayerwiseLossType("mse_normalized") == LayerwiseLossType.MSE_NORMALIZED


class TestLayerwiseDistillationLoss:
    """Tests for LayerwiseDistillationLoss."""

    @pytest.fixture
    def hidden_states(self):
        """Create sample hidden states for testing."""
        batch_size, seq_len, hidden_size = 2, 10, 128
        num_layers = 4
        return [
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers)
        ]

    def test_mse_normalized_loss_forward(self, hidden_states):
        """Test MSE normalized loss forward pass."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        result = loss_fn(hidden_states, hidden_states)

        assert "loss" in result
        assert "layer_losses" in result
        assert "mean_layer_loss" in result
        assert result["loss"].dim() == 0  # Scalar
        assert result["loss"].item() < 1e-5  # Same inputs = ~0 loss

    def test_mse_normalized_loss_string_type(self, hidden_states):
        """Test MSE normalized loss with string type."""
        loss_fn = LayerwiseDistillationLoss(loss_type="mse_normalized")

        result = loss_fn(hidden_states, hidden_states)

        assert result["loss"].item() < 1e-5

    def test_mse_loss(self, hidden_states):
        """Test MSE loss (without normalization)."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE,
            normalize=False,
        )

        teacher = [h.clone() for h in hidden_states]
        student = [h + 0.1 * torch.randn_like(h) for h in hidden_states]

        result = loss_fn(student, teacher)

        assert result["loss"].item() > 0
        assert len(result["layer_losses"]) == len(hidden_states)

    def test_mse_normalized_loss(self, hidden_states):
        """Test MSE normalized loss (OneBit style)."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
            normalize=True,
        )

        teacher = [h.clone() for h in hidden_states]
        student = [h + 0.1 * torch.randn_like(h) for h in hidden_states]

        result = loss_fn(student, teacher)

        assert result["loss"].item() > 0
        assert len(result["layer_losses"]) == len(hidden_states)

    def test_mse_normalized_same_inputs(self, hidden_states):
        """Test MSE normalized with identical inputs gives ~0 loss."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        result = loss_fn(hidden_states, hidden_states)

        assert result["loss"].item() < 1e-5

    def test_kl_loss(self):
        """Test KL divergence loss."""
        hidden_size, vocab_size = 128, 1000
        num_layers = 2

        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.KL,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            temperature=2.0,
        )

        student = [torch.randn(2, 10, hidden_size) for _ in range(num_layers)]
        teacher = [torch.randn(2, 10, hidden_size) for _ in range(num_layers)]

        result = loss_fn(student, teacher)

        assert result["loss"].item() >= 0  # KL is non-negative

    def test_kl_requires_dimensions(self):
        """Test that KL loss requires hidden_size and vocab_size."""
        with pytest.raises(ValueError, match="hidden_size and vocab_size"):
            LayerwiseDistillationLoss(loss_type=LayerwiseLossType.KL)

    def test_inner_product_loss(self, hidden_states):
        """Test inner product loss."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.INNER_PRODUCT,
        )

        # Same inputs normalized should give inner product of 1, so loss = -1
        result = loss_fn(hidden_states, hidden_states)

        # For same normalized vectors, inner product = 1, loss = -1
        assert result["loss"].item() < 0

    def test_inner_product_different_inputs(self, hidden_states):
        """Test inner product loss with different inputs."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.INNER_PRODUCT,
        )

        student = [torch.randn_like(h) for h in hidden_states]

        result = loss_fn(student, hidden_states)

        # Should be different from -1 (not perfectly aligned)
        assert result["loss"].item() > -1.0

    # Layer weights tests

    def test_uniform_layer_weights(self, hidden_states):
        """Test uniform layer weighting (default)."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
            layer_weights=None,
        )

        weights = loss_fn._get_layer_weights(len(hidden_states))

        # Should be uniform
        expected = 1.0 / len(hidden_states)
        for w in weights:
            assert abs(w - expected) < 1e-6

        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_progressive_layer_weights(self, hidden_states):
        """Test progressive layer weighting (later layers higher)."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
            layer_weights="progressive",
        )

        weights = loss_fn._get_layer_weights(len(hidden_states))

        # Later layers should have higher weights
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]

        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_exponential_layer_weights(self, hidden_states):
        """Test exponential layer weighting."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
            layer_weights="exponential",
        )

        weights = loss_fn._get_layer_weights(len(hidden_states))

        # Later layers should have exponentially higher weights
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]

        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_custom_layer_weights(self):
        """Test custom layer weights."""
        custom_weights = [0.1, 0.2, 0.3, 0.4]

        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
            layer_weights=custom_weights,
        )

        weights = loss_fn._get_layer_weights(4)

        # Should be normalized
        assert abs(sum(weights) - 1.0) < 1e-6

        # Ratios should be preserved
        assert abs(weights[1] / weights[0] - 2.0) < 1e-6
        assert abs(weights[3] / weights[0] - 4.0) < 1e-6

    def test_custom_weights_wrong_length(self):
        """Test that wrong length custom weights raises error."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
            layer_weights=[0.1, 0.2, 0.3],
        )

        hidden_states = [torch.randn(2, 10, 64) for _ in range(5)]

        with pytest.raises(ValueError, match="layer_weights length"):
            loss_fn(hidden_states, hidden_states)

    # Attention mask tests

    def test_attention_mask(self, hidden_states):
        """Test with attention mask."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE)

        mask = torch.ones(2, 10)
        mask[:, 5:] = 0  # Mask second half

        result = loss_fn(hidden_states, hidden_states, attention_mask=mask)

        assert result["loss"].item() < 1e-5

    def test_attention_mask_affects_loss(self, hidden_states):
        """Test that attention mask actually affects the loss."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE)

        student = [h + 0.5 * torch.randn_like(h) for h in hidden_states]

        # Full mask
        full_mask = torch.ones(2, 10)
        result_full = loss_fn(student, hidden_states, attention_mask=full_mask)

        # Partial mask
        partial_mask = torch.ones(2, 10)
        partial_mask[:, 5:] = 0
        result_partial = loss_fn(student, hidden_states, attention_mask=partial_mask)

        # Results should be different
        assert result_full["loss"].item() != result_partial["loss"].item()

    # Gradient flow tests

    def test_gradient_flow(self):
        """Test that gradients flow through loss."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        student = [torch.randn(2, 10, 64, requires_grad=True)]
        teacher = [torch.randn(2, 10, 64)]

        result = loss_fn(student, teacher)
        result["loss"].backward()

        assert student[0].grad is not None

    def test_gradient_flow_mse_normalized(self):
        """Test gradient flow for MSE normalized loss."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        student = [torch.randn(2, 10, 64, requires_grad=True)]
        teacher = [torch.randn(2, 10, 64)]

        result = loss_fn(student, teacher)
        result["loss"].backward()

        assert student[0].grad is not None

    def test_gradient_flow_kl(self):
        """Test gradient flow for KL loss."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.KL,
            hidden_size=64,
            vocab_size=100,
        )

        student = [torch.randn(2, 10, 64, requires_grad=True)]
        teacher = [torch.randn(2, 10, 64)]

        result = loss_fn(student, teacher)
        result["loss"].backward()

        assert student[0].grad is not None

    # Edge cases

    def test_empty_hidden_states(self):
        """Test with empty hidden states list."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        result = loss_fn([], [])

        assert result["loss"].item() == 0.0
        assert len(result["layer_losses"]) == 0

    def test_layer_count_mismatch(self, hidden_states):
        """Test that mismatched layer counts raise error."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        with pytest.raises(ValueError, match="Layer count mismatch"):
            loss_fn(hidden_states, hidden_states[:-1])

    def test_single_layer(self):
        """Test with single layer."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        student = [torch.randn(2, 10, 64)]
        teacher = [torch.randn(2, 10, 64)]

        result = loss_fn(student, teacher)

        assert result["loss"].item() > 0
        assert len(result["layer_losses"]) == 1

    # Output format tests

    def test_output_keys(self, hidden_states):
        """Test that output contains expected keys."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE_NORMALIZED)

        result = loss_fn(hidden_states, hidden_states)

        assert "loss" in result
        assert "layer_losses" in result
        assert "mean_layer_loss" in result

    def test_layer_losses_detached(self, hidden_states):
        """Test that layer_losses are detached (for logging)."""
        loss_fn = LayerwiseDistillationLoss(loss_type=LayerwiseLossType.MSE)

        student = [h.requires_grad_() for h in hidden_states]
        teacher = [torch.randn_like(h) for h in hidden_states]

        result = loss_fn(student, teacher)

        # Layer losses should be detached
        for layer_loss in result["layer_losses"]:
            assert not layer_loss.requires_grad

        # But main loss should allow gradients
        assert result["loss"].requires_grad


class TestLayerwiseDistillationLossIntegration:
    """Integration tests for LayerwiseDistillationLoss."""

    def test_all_loss_types_work(self):
        """Test that all loss types produce valid output."""
        hidden_states = [torch.randn(2, 10, 64) for _ in range(3)]

        for loss_type in LayerwiseLossType:
            if loss_type == LayerwiseLossType.KL:
                loss_fn = LayerwiseDistillationLoss(
                    loss_type=loss_type,
                    hidden_size=64,
                    vocab_size=100,
                )
            else:
                loss_fn = LayerwiseDistillationLoss(loss_type=loss_type)

            result = loss_fn(hidden_states, hidden_states)

            assert torch.isfinite(result["loss"])
            assert len(result["layer_losses"]) == 3

    def test_all_weight_strategies_work(self):
        """Test that all weight strategies produce valid output."""
        hidden_states = [torch.randn(2, 10, 64) for _ in range(4)]

        for weight_strategy in [None, "progressive", "exponential", [1, 2, 3, 4]]:
            loss_fn = LayerwiseDistillationLoss(
                loss_type=LayerwiseLossType.MSE_NORMALIZED,
                layer_weights=weight_strategy,
            )

            result = loss_fn(hidden_states, hidden_states)

            assert torch.isfinite(result["loss"])
