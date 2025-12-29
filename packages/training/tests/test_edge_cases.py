"""Edge case and integration tests."""

import pytest
import torch

from wrinklefree.models import BitLinear, SubLN, RMSNorm
from wrinklefree.models.attention import BitNetAttention
from wrinklefree.models.ffn import BitNetFFN
from wrinklefree.distillation import (
    AttentionDistillationLoss,
    LogitsDistillationLoss,
    BitDistillLoss,
    HiddenStateDistillationLoss,
    AttentionRelationDistillationLoss,
)


class TestNumericalStability:
    """Tests for numerical stability across modules."""

    def test_bitlinear_with_extreme_values(self):
        """Test BitLinear with extreme input values."""
        layer = BitLinear(64, 128)

        # Very large values
        x_large = torch.randn(2, 10, 64) * 1000
        output_large = layer(x_large)
        assert torch.isfinite(output_large).all()

        # Very small values
        x_small = torch.randn(2, 10, 64) * 1e-6
        output_small = layer(x_small)
        assert torch.isfinite(output_small).all()

    def test_subln_with_constant_input(self):
        """Test SubLN with constant input (edge case for normalization)."""
        norm = SubLN(64)

        # All same values - normalization should handle this
        x = torch.ones(2, 10, 64) * 5.0
        output = norm(x)

        assert torch.isfinite(output).all()

    def test_attention_with_long_sequences(self):
        """Test attention with longer sequences."""
        attn = BitNetAttention(
            hidden_size=128,
            num_attention_heads=4,
            max_position_embeddings=2048,
        )

        x = torch.randn(1, 512, 128)  # Longer sequence
        output, _ = attn(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_ffn_with_zero_input(self):
        """Test FFN with all-zero input."""
        ffn = BitNetFFN(hidden_size=64, intermediate_size=256)

        x = torch.zeros(2, 8, 64)
        output = ffn(x)

        assert torch.isfinite(output).all()


class TestAttentionMaskEdgeCases:
    """Tests for attention mask edge cases."""

    def test_attention_with_partial_mask(self):
        """Test attention with partial masking (some tokens masked)."""
        attn = BitNetAttention(hidden_size=128, num_attention_heads=4)

        batch, seq = 2, 8
        x = torch.randn(batch, seq, 128)

        # Mask out last 2 positions for each query
        mask = torch.zeros(1, 1, seq, seq)
        mask[:, :, :, -2:] = float("-inf")

        output, weights = attn(x, attention_mask=mask, output_attentions=True)

        # Masked positions should have ~0 attention
        assert weights[:, :, :, -2:].abs().max() < 1e-5

    def test_attention_distillation_with_mask(self):
        """Test attention distillation loss with attention mask."""
        loss_fn = AttentionDistillationLoss(alpha=1.0)

        batch, heads, seq = 2, 4, 8
        student_attn = [torch.softmax(torch.randn(batch, heads, seq, seq), dim=-1)]
        teacher_attn = [torch.softmax(torch.randn(batch, heads, seq, seq), dim=-1)]

        # Mask: (batch, seq) - 1 for valid, 0 for padding
        mask = torch.ones(batch, seq)
        mask[:, -2:] = 0  # Last 2 tokens are padding

        loss = loss_fn(student_attn, teacher_attn, attention_mask=mask)

        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_attention_distillation_all_masked(self):
        """Test attention distillation when all tokens are masked."""
        loss_fn = AttentionDistillationLoss(alpha=1.0)

        batch, heads, seq = 2, 4, 8
        student_attn = [torch.softmax(torch.randn(batch, heads, seq, seq), dim=-1)]
        teacher_attn = [torch.softmax(torch.randn(batch, heads, seq, seq), dim=-1)]

        # All masked
        mask = torch.zeros(batch, seq)

        loss = loss_fn(student_attn, teacher_attn, attention_mask=mask)

        # Should handle gracefully (return 0 or small value)
        assert torch.isfinite(loss)


class TestDistillationLossEdgeCases:
    """Edge case tests for distillation losses."""

    def test_logits_loss_with_very_different_distributions(self):
        """Test logits loss with very different student/teacher distributions."""
        loss_fn = LogitsDistillationLoss(temperature=5.0)

        # Student: uniform-ish
        student = torch.zeros(2, 10, 100)
        # Teacher: very peaked
        teacher = torch.zeros(2, 10, 100)
        teacher[:, :, 0] = 100.0

        loss = loss_fn(student, teacher)

        assert torch.isfinite(loss)
        assert loss.item() > 0  # Should be high divergence

    def test_logits_loss_with_single_class(self):
        """Test logits loss with single class vocabulary."""
        loss_fn = LogitsDistillationLoss(temperature=5.0)

        student = torch.randn(2, 10, 1)
        teacher = torch.randn(2, 10, 1)

        loss = loss_fn(student, teacher)

        # With single class, softmax is always 1, so KL should be ~0
        assert torch.isfinite(loss)

    def test_bitdistill_loss_without_attention(self):
        """Test BitDistill loss when attention distillation is disabled."""
        loss_fn = BitDistillLoss(
            lambda_logits=10.0,
            gamma_attention=0.0,  # Disable attention loss
            temperature=5.0,
        )

        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))

        result = loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_attentions=None,
            teacher_attentions=None,
            labels=labels,
        )

        assert torch.isfinite(result["loss"])
        assert result["attention_distill_loss"].item() == 0.0

    def test_hidden_state_distillation_same_dim(self):
        """Test hidden state distillation with same dimensions."""
        loss_fn = HiddenStateDistillationLoss(
            student_dim=64,
            teacher_dim=64,
            use_projection=True,
        )

        # Should not create projection when dims match
        assert loss_fn.projection is None

        student_hidden = torch.randn(2, 10, 64)
        teacher_hidden = torch.randn(2, 10, 64)

        loss = loss_fn(student_hidden, teacher_hidden)

        assert torch.isfinite(loss)

    def test_hidden_state_distillation_different_dim(self):
        """Test hidden state distillation with different dimensions."""
        loss_fn = HiddenStateDistillationLoss(
            student_dim=64,
            teacher_dim=128,
            use_projection=True,
        )

        # Should create projection
        assert loss_fn.projection is not None

        student_hidden = torch.randn(2, 10, 64)
        teacher_hidden = torch.randn(2, 10, 128)

        loss = loss_fn(student_hidden, teacher_hidden)

        assert torch.isfinite(loss)

    def test_attention_relation_distillation(self):
        """Test attention relation distillation loss."""
        loss_fn = AttentionRelationDistillationLoss(normalize=True)

        # Attention scores (before softmax)
        student_scores = [torch.randn(2, 4, 8, 8)]
        teacher_scores = [torch.randn(2, 4, 8, 8)]

        loss = loss_fn(student_scores, teacher_scores)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_attention_relation_without_normalization(self):
        """Test attention relation loss without normalization."""
        loss_fn = AttentionRelationDistillationLoss(normalize=False)

        student_scores = [torch.randn(2, 4, 8, 8)]
        teacher_scores = [torch.randn(2, 4, 8, 8)]

        loss = loss_fn(student_scores, teacher_scores)

        assert torch.isfinite(loss)


class TestBatchSizeEdgeCases:
    """Tests for various batch size edge cases."""

    def test_batch_size_1(self):
        """Test all modules with batch size 1."""
        # BitLinear
        layer = BitLinear(64, 128)
        x = torch.randn(1, 10, 64)
        assert layer(x).shape == (1, 10, 128)

        # SubLN
        norm = SubLN(64)
        x = torch.randn(1, 10, 64)
        assert norm(x).shape == (1, 10, 64)

        # Attention
        attn = BitNetAttention(hidden_size=64, num_attention_heads=4)
        x = torch.randn(1, 8, 64)
        output, _ = attn(x)
        assert output.shape == (1, 8, 64)

        # FFN
        ffn = BitNetFFN(hidden_size=64, intermediate_size=256)
        x = torch.randn(1, 8, 64)
        assert ffn(x).shape == (1, 8, 64)

    def test_sequence_length_1(self):
        """Test all modules with sequence length 1."""
        # Attention
        attn = BitNetAttention(hidden_size=64, num_attention_heads=4)
        x = torch.randn(2, 1, 64)
        output, _ = attn(x)
        assert output.shape == (2, 1, 64)

        # FFN
        ffn = BitNetFFN(hidden_size=64, intermediate_size=256)
        x = torch.randn(2, 1, 64)
        assert ffn(x).shape == (2, 1, 64)

    def test_large_batch_gradient_accumulation_simulation(self):
        """Test that gradients accumulate correctly with micro-batches."""
        layer = BitLinear(64, 128)

        # Single large batch
        x_large = torch.randn(16, 10, 64, requires_grad=True)
        output_large = layer(x_large)
        loss_large = output_large.sum()
        loss_large.backward()
        grad_large = layer.weight.grad.clone()

        # Reset gradients
        layer.zero_grad()

        # Two smaller batches accumulated
        x1 = x_large[:8].detach().requires_grad_(True)
        x2 = x_large[8:].detach().requires_grad_(True)

        output1 = layer(x1)
        loss1 = output1.sum()
        loss1.backward()

        output2 = layer(x2)
        loss2 = output2.sum()
        loss2.backward()

        grad_accumulated = layer.weight.grad.clone()

        # Gradients should be the same
        assert torch.allclose(grad_large, grad_accumulated, atol=1e-5)


class TestLayerWeightEdgeCases:
    """Tests for layer weight handling edge cases."""

    def test_attention_layer_weights_custom(self):
        """Test attention distillation with custom layer weights."""
        # Give more weight to later layers
        layer_weights = [0.1, 0.2, 0.3, 0.4]
        loss_fn = AttentionDistillationLoss(alpha=1.0, layer_weights=layer_weights)

        student_attn = [
            torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
            for _ in range(4)
        ]
        teacher_attn = [
            torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
            for _ in range(4)
        ]

        loss = loss_fn(student_attn, teacher_attn)

        assert torch.isfinite(loss)

    def test_mismatched_layer_count_raises(self):
        """Test that mismatched layer counts raise error."""
        loss_fn = AttentionDistillationLoss(alpha=1.0)

        student_attn = [torch.softmax(torch.randn(2, 4, 8, 8), dim=-1) for _ in range(4)]
        teacher_attn = [torch.softmax(torch.randn(2, 4, 8, 8), dim=-1) for _ in range(3)]

        with pytest.raises(ValueError, match="Number of layers mismatch"):
            loss_fn(student_attn, teacher_attn)

    def test_empty_layer_list(self):
        """Test attention loss with empty layer list."""
        loss_fn = AttentionDistillationLoss(alpha=1.0)

        loss = loss_fn([], [])

        # Should return 0 for empty list
        assert loss.item() == 0.0


class TestDtypeConsistency:
    """Tests for dtype consistency across operations."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_bitlinear_dtype_preservation(self, dtype):
        """Test BitLinear preserves dtype."""
        layer = BitLinear(64, 128).to(dtype)
        x = torch.randn(2, 10, 64, dtype=dtype)

        output = layer(x)

        assert output.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_subln_dtype_preservation(self, dtype):
        """Test SubLN preserves dtype."""
        norm = SubLN(64).to(dtype)
        x = torch.randn(2, 10, 64, dtype=dtype)

        output = norm(x)

        assert output.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_attention_dtype_preservation(self, dtype):
        """Test attention preserves dtype (skip fp16 due to precision issues)."""
        attn = BitNetAttention(hidden_size=64, num_attention_heads=4).to(dtype)
        x = torch.randn(2, 8, 64, dtype=dtype)

        output, _ = attn(x)

        assert output.dtype == dtype
