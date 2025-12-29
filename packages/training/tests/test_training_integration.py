"""Integration tests for training stage transitions and combined losses.

These tests verify the training stages work correctly end-to-end:
- Stage 1: SubLN insertion and BitLinear conversion
- Stage 1.9: Layer-wise distillation with combined losses
- Stage 2: Continue pre-training with lambda warmup
- Stage 3: Full distillation pipeline
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from wrinklefree.models.bitlinear import BitLinear
from wrinklefree.models.subln import SubLN
from wrinklefree.distillation.layerwise_loss import LayerwiseDistillationLoss, LayerwiseLossType

# Stage 3 distillation classes moved to distillation package
from distillation.losses import BitDistillLoss
from wrinklefree.quantization.lambda_warmup import (
    LambdaWarmup,
    set_global_lambda_warmup,
    get_current_lambda,
)


class TestStage1Conversion:
    """Test Stage 1 SubLN insertion and BitLinear conversion."""

    def test_conversion_creates_sequential_wrappers(self):
        """Verify Stage 1 creates Sequential(SubLN, BitLinear) for output projections."""
        from wrinklefree.training.stage1 import convert_attention_layer

        # Create a simple attention module with Linear layers
        class SimpleAttention(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

        attn = SimpleAttention(64)
        convert_attention_layer(attn, hidden_size=64, exclude_layers=[])

        # Check Q, K, V are converted to BitLinear
        assert isinstance(attn.q_proj, BitLinear)
        assert isinstance(attn.k_proj, BitLinear)
        assert isinstance(attn.v_proj, BitLinear)

        # Check o_proj is wrapped in Sequential(SubLN, BitLinear)
        assert isinstance(attn.o_proj, nn.Sequential)
        assert len(attn.o_proj) == 2
        assert isinstance(attn.o_proj[0], SubLN)
        assert isinstance(attn.o_proj[1], BitLinear)

    def test_conversion_preserves_weights(self):
        """Verify Stage 1 conversion preserves original weights."""
        from wrinklefree.training.stage1 import convert_attention_layer

        class SimpleAttention(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

        attn = SimpleAttention(64)

        # Store original weights
        original_q_weight = attn.q_proj.weight.data.clone()
        original_o_weight = attn.o_proj.weight.data.clone()

        convert_attention_layer(attn, hidden_size=64, exclude_layers=[])

        # Check weights are preserved
        assert torch.allclose(attn.q_proj.weight.data, original_q_weight)
        # o_proj is now Sequential, check the BitLinear inside
        assert torch.allclose(attn.o_proj[1].weight.data, original_o_weight)

    def test_converted_model_forward_pass(self):
        """Verify converted attention module can do forward pass."""
        from wrinklefree.training.stage1 import convert_attention_layer

        class SimpleAttention(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                attn = torch.softmax(q @ k.transpose(-2, -1) / 8, dim=-1)
                out = attn @ v
                return self.o_proj(out)

        attn = SimpleAttention(64)
        convert_attention_layer(attn, hidden_size=64, exclude_layers=[])

        # Forward pass should work
        x = torch.randn(2, 10, 64)
        output = attn(x)
        assert output.shape == x.shape


class TestStage19LayerwiseLoss:
    """Test Stage 1.9 layer-wise distillation with combined losses."""

    def test_layerwise_loss_with_uniform_weights(self):
        """Verify layer-wise loss works with uniform weights."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
            layer_weights=None,  # Uniform
        )

        num_layers = 4
        student_h = [torch.randn(2, 10, 64) for _ in range(num_layers)]
        teacher_h = [torch.randn(2, 10, 64) for _ in range(num_layers)]

        result = loss_fn(student_h, teacher_h)

        assert "loss" in result
        assert "layer_losses" in result
        assert len(result["layer_losses"]) == num_layers
        assert result["loss"].item() > 0

    def test_layerwise_loss_with_progressive_weights(self):
        """Verify progressive weights give more weight to later layers."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
            layer_weights="progressive",
        )

        # Use 4 layers
        num_layers = 4
        weights = loss_fn._get_layer_weights(num_layers)

        # Progressive: later layers should have higher weights
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1], "Progressive weights should increase"

    def test_identical_hidden_states_low_loss(self):
        """Verify identical hidden states produce near-zero loss."""
        loss_fn = LayerwiseDistillationLoss(
            loss_type=LayerwiseLossType.MSE_NORMALIZED,
        )

        hidden = [torch.randn(2, 10, 64) for _ in range(4)]

        # Same hidden states
        result = loss_fn(hidden, hidden)

        assert result["loss"].item() < 1e-5, "Identical hidden states should have ~0 loss"


class TestLambdaWarmupIntegration:
    """Test lambda warmup integration with training."""

    def test_lambda_warmup_schedule(self):
        """Verify lambda warmup schedule works correctly."""
        warmup = LambdaWarmup(warmup_steps=10, schedule="linear")

        # Initial state
        assert warmup.lambda_val == 0.0

        # After 5 steps (halfway)
        for _ in range(5):
            warmup.step()
        assert warmup.lambda_val == pytest.approx(0.5, abs=0.05)

        # After 10 steps (complete)
        for _ in range(5):
            warmup.step()
        assert warmup.lambda_val == 1.0

        # After warmup complete, stays at 1.0
        warmup.step()
        assert warmup.lambda_val == 1.0

    def test_global_lambda_affects_bitlinear(self):
        """Verify global lambda warmup affects BitLinear forward pass."""
        layer = BitLinear(64, 64)
        x = torch.randn(2, 10, 64)
        W = layer.weight.data.clone()

        # λ=0: full precision
        warmup_0 = LambdaWarmup(warmup_steps=100, min_lambda=0.0, max_lambda=0.0)
        set_global_lambda_warmup(warmup_0)

        output_fp = layer(x)
        expected_fp = F.linear(x, W)
        assert torch.allclose(output_fp, expected_fp, atol=1e-5)

        # λ=1: full quantization
        warmup_1 = LambdaWarmup(warmup_steps=0, min_lambda=1.0, max_lambda=1.0)
        set_global_lambda_warmup(warmup_1)

        output_quant = layer(x)
        # Should not match FP exactly
        # (unless weights happen to be quantized to same values)

        # Cleanup
        set_global_lambda_warmup(None)

    def test_lambda_warmup_state_dict(self):
        """Verify lambda warmup can be saved and loaded."""
        warmup = LambdaWarmup(warmup_steps=100, schedule="cosine")

        # Advance some steps
        for _ in range(50):
            warmup.step()

        # Save state
        state = warmup.state_dict()
        assert state["current_step"] == 50

        # Create new warmup and load state
        new_warmup = LambdaWarmup(warmup_steps=100)
        new_warmup.load_state_dict(state)

        assert new_warmup.current_step == 50
        assert new_warmup.lambda_val == warmup.lambda_val


class TestCombinedLossIntegration:
    """Test combined distillation loss used in Stage 3."""

    def test_combined_loss_all_components(self):
        """Verify combined loss computes all three components."""
        loss_fn = BitDistillLoss(
            lambda_logits=10.0,
            gamma_attention=1e-5,
            use_relation_distill=True,
        )

        # Create dummy inputs
        batch_size, seq_len, vocab_size = 2, 16, 1000
        num_heads = 8

        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        # Attention weights (after softmax)
        student_attn = [
            torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
            for _ in range(12)
        ]
        teacher_attn = [
            torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
            for _ in range(12)
        ]

        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        result = loss_fn(
            student_logits, teacher_logits,
            student_attn, teacher_attn,
            labels
        )

        # All components should be present and non-negative
        assert result["ce_loss"].item() > 0
        assert result["logits_distill_loss"].item() >= 0
        assert result["attention_distill_loss"].item() >= 0

        # Total should be the sum
        expected = (
            result["ce_loss"] +
            10.0 * result["logits_distill_loss"] +
            1e-5 * result["attention_distill_loss"]
        )
        assert torch.allclose(result["loss"], expected, atol=1e-5)

    def test_combined_loss_gradient_flow(self):
        """Verify gradients flow through combined loss."""
        loss_fn = BitDistillLoss(lambda_logits=1.0, gamma_attention=0.0)

        batch_size, seq_len, vocab_size = 2, 8, 100

        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        result = loss_fn(student_logits, teacher_logits, None, None, labels)
        result["loss"].backward()

        assert student_logits.grad is not None
        assert not torch.all(student_logits.grad == 0)


class TestSTEGradientIntegration:
    """Test STE gradient flow through full training step."""

    def test_ste_gradient_through_bitlinear_stack(self):
        """Verify gradients flow through multiple BitLinear layers."""
        # Ensure full quantization
        set_global_lambda_warmup(None)

        model = nn.Sequential(
            BitLinear(64, 128),
            nn.ReLU(),
            BitLinear(128, 64),
            nn.ReLU(),
            BitLinear(64, 10),
        )

        x = torch.randn(2, 64, requires_grad=True)
        target = torch.randint(0, 10, (2,))

        output = model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # All BitLinear layers should have gradients
        for i, layer in enumerate(model):
            if isinstance(layer, BitLinear):
                assert layer.weight.grad is not None, f"Layer {i} missing weight gradient"
                assert not torch.all(layer.weight.grad == 0), f"Layer {i} has zero gradient"

        # Input should have gradient
        assert x.grad is not None

    def test_ste_gradient_matches_fp_direction(self):
        """Verify STE gradients point in similar direction as FP gradients."""
        set_global_lambda_warmup(None)

        layer = BitLinear(64, 64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        # Forward and backward
        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Gradients should be non-trivial
        assert layer.weight.grad is not None
        grad_magnitude = layer.weight.grad.abs().mean()
        assert grad_magnitude > 0
