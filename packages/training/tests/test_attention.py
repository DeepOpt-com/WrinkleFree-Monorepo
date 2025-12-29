"""Tests for BitNet Attention module."""

import math

import pytest
import torch

from wrinklefree.models.attention import (
    BitNetAttention,
    BitNetFlashAttention,
    precompute_freqs_cis,
    apply_rotary_emb,
    repeat_kv,
)


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_precompute_freqs_shape(self):
        """Test that precomputed frequencies have correct shape."""
        dim = 64
        seq_len = 128

        freqs = precompute_freqs_cis(dim, seq_len)

        assert freqs.shape == (seq_len, dim // 2)
        assert freqs.dtype == torch.complex64

    def test_apply_rotary_preserves_shape(self):
        """Test that rotary embeddings preserve tensor shapes."""
        batch, seq, heads, head_dim = 2, 16, 8, 64
        xq = torch.randn(batch, seq, heads, head_dim)
        xk = torch.randn(batch, seq, heads, head_dim)
        freqs = precompute_freqs_cis(head_dim, seq)

        xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs)

        assert xq_rot.shape == xq.shape
        assert xk_rot.shape == xk.shape

    def test_apply_rotary_dtype_preserved(self):
        """Test that dtype is preserved after rotary embedding."""
        batch, seq, heads, head_dim = 2, 16, 8, 64

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            xq = torch.randn(batch, seq, heads, head_dim, dtype=dtype)
            xk = torch.randn(batch, seq, heads, head_dim, dtype=dtype)
            freqs = precompute_freqs_cis(head_dim, seq)

            xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs)

            assert xq_rot.dtype == dtype
            assert xk_rot.dtype == dtype

    def test_rotary_position_dependent(self):
        """Test that rotary embeddings produce different results at different positions."""
        batch, seq, heads, head_dim = 1, 4, 1, 64
        x = torch.randn(batch, seq, heads, head_dim)
        freqs = precompute_freqs_cis(head_dim, seq)

        x_rot, _ = apply_rotary_emb(x, x, freqs)

        # Different positions should have different rotations
        # (unless the original values happen to be the same)
        pos0 = x_rot[0, 0, 0, :]
        pos1 = x_rot[0, 1, 0, :]

        # The rotations should be different
        assert not torch.allclose(pos0, pos1, atol=1e-3)


class TestRepeatKV:
    """Tests for KV head repetition (GQA)."""

    def test_no_repetition_when_n_rep_1(self):
        """Test that n_rep=1 returns unchanged tensor."""
        x = torch.randn(2, 16, 4, 64)
        x_rep = repeat_kv(x, n_rep=1)

        assert x_rep is x  # Should be same tensor

    def test_correct_shape_after_repetition(self):
        """Test that repetition produces correct shape."""
        batch, seq, kv_heads, head_dim = 2, 16, 4, 64
        n_rep = 8  # Repeat each KV head 8 times

        x = torch.randn(batch, seq, kv_heads, head_dim)
        x_rep = repeat_kv(x, n_rep)

        assert x_rep.shape == (batch, seq, kv_heads * n_rep, head_dim)

    def test_repetition_preserves_values(self):
        """Test that repeated heads contain the original values."""
        batch, seq, kv_heads, head_dim = 2, 16, 2, 64
        n_rep = 4

        x = torch.randn(batch, seq, kv_heads, head_dim)
        x_rep = repeat_kv(x, n_rep)

        # Check that each KV head is repeated n_rep times
        for kv_idx in range(kv_heads):
            original = x[:, :, kv_idx, :]
            for rep_idx in range(n_rep):
                repeated = x_rep[:, :, kv_idx * n_rep + rep_idx, :]
                assert torch.allclose(original, repeated)


class TestBitNetAttention:
    """Tests for BitNet Attention layer."""

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        hidden_size = 256
        num_heads = 8
        batch, seq = 2, 16

        attn = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        x = torch.randn(batch, seq, hidden_size)
        output, _ = attn(x)

        assert output.shape == (batch, seq, hidden_size)

    def test_forward_with_gqa(self):
        """Test forward with Grouped-Query Attention."""
        hidden_size = 256
        num_heads = 8
        num_kv_heads = 2  # GQA: 4 query heads share each KV head
        batch, seq = 2, 16

        attn = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        x = torch.randn(batch, seq, hidden_size)
        output, _ = attn(x)

        assert output.shape == (batch, seq, hidden_size)

    def test_output_attentions(self):
        """Test that attention weights can be returned."""
        hidden_size = 256
        num_heads = 8
        batch, seq = 2, 16

        attn = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        x = torch.randn(batch, seq, hidden_size)
        output, attn_weights = attn(x, output_attentions=True)

        assert output.shape == (batch, seq, hidden_size)
        assert attn_weights is not None
        assert attn_weights.shape == (batch, num_heads, seq, seq)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along key dimension."""
        hidden_size = 256
        num_heads = 8
        batch, seq = 2, 16

        attn = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        x = torch.randn(batch, seq, hidden_size)
        _, attn_weights = attn(x, output_attentions=True)

        # Softmax along last dimension should sum to 1
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_attention_mask(self):
        """Test that attention mask is applied correctly."""
        hidden_size = 256
        num_heads = 8
        batch, seq = 2, 8

        attn = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        x = torch.randn(batch, seq, hidden_size)

        # Create causal mask (lower triangular)
        mask = torch.triu(torch.ones(seq, seq) * float("-inf"), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        _, attn_weights = attn(x, attention_mask=mask, output_attentions=True)

        # Upper triangular should be ~0 (masked out)
        for i in range(seq):
            for j in range(i + 1, seq):
                assert attn_weights[0, 0, i, j].item() < 1e-5

    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        hidden_size = 256
        num_heads = 8
        batch, seq = 2, 16

        attn = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        x = torch.randn(batch, seq, hidden_size, requires_grad=True)
        output, _ = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert attn.q_proj.weight.grad is not None
        assert attn.o_proj.weight.grad is not None

    def test_subln_is_applied(self):
        """Test that SubLN is present and applied."""
        hidden_size = 256
        num_heads = 8

        attn = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        # SubLN should be a submodule
        assert hasattr(attn, "subln")
        assert attn.subln is not None

    def test_different_head_dims(self):
        """Test attention with custom head dimensions."""
        hidden_size = 256
        num_heads = 8
        head_dim = 64  # Custom head dim (not hidden_size // num_heads)
        batch, seq = 2, 16

        attn = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
        )

        x = torch.randn(batch, seq, hidden_size)
        output, _ = attn(x)

        assert output.shape == (batch, seq, hidden_size)


class TestBitNetFlashAttention:
    """Tests for BitNet Flash Attention."""

    def test_forward_shape(self):
        """Test that flash attention produces correct output shape."""
        hidden_size = 256
        num_heads = 8
        batch, seq = 2, 16

        attn = BitNetFlashAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        x = torch.randn(batch, seq, hidden_size)
        output, _ = attn(x)

        assert output.shape == (batch, seq, hidden_size)

    def test_fallback_when_requesting_weights(self):
        """Test fallback to standard attention when requesting weights."""
        hidden_size = 256
        num_heads = 8
        batch, seq = 2, 16

        attn = BitNetFlashAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        x = torch.randn(batch, seq, hidden_size)
        output, attn_weights = attn(x, output_attentions=True)

        # Should fall back and return weights
        assert attn_weights is not None
        assert attn_weights.shape == (batch, num_heads, seq, seq)

    def test_flash_and_standard_similar_output(self):
        """Test that flash and standard attention produce similar results."""
        hidden_size = 256
        num_heads = 8
        batch, seq = 2, 16

        # Create both types with same weights
        attn_flash = BitNetFlashAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )
        attn_standard = BitNetAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
        )

        # Copy weights
        attn_standard.load_state_dict(attn_flash.state_dict())

        x = torch.randn(batch, seq, hidden_size)

        # Without mask, flash uses is_causal=True, so we need causal mask for standard
        with torch.no_grad():
            output_flash, _ = attn_flash(x)
            # For standard attention, we need to provide causal mask to match
            causal_mask = torch.triu(
                torch.ones(seq, seq) * float("-inf"), diagonal=1
            ).unsqueeze(0).unsqueeze(0)
            output_standard, _ = attn_standard(x, attention_mask=causal_mask)

        # Results should be similar (not exact due to implementation differences)
        assert torch.allclose(output_flash, output_standard, atol=1e-4)
