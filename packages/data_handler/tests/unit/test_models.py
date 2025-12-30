"""Unit tests for model components.

Tests attention, transformer blocks, and full model.
"""

import pytest
import torch

from data_handler._legacy.models.config import (
    MobileLLMConfig,
    MobileLLM140MConfig,
    MobileLLM950MConfig,
)
from data_handler._legacy.models.attention import (
    MultiHeadAttention,
    FeedForward,
    RMSNorm,
    precompute_rope_frequencies,
    apply_rope,
)
from data_handler._legacy.models.transformer import TransformerBlock, TransformerDecoder
from data_handler._legacy.models.mobilellm import MobileLLM


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self):
        """Test output shape matches input."""
        norm = RMSNorm(dim=512)
        x = torch.randn(2, 128, 512)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_rms(self):
        """Test that output has approximately unit RMS."""
        norm = RMSNorm(dim=512)
        x = torch.randn(2, 128, 512)
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        # Should be close to 1 after normalization (with learned weight)
        assert rms.mean().item() < 2.0


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_frequency_shape(self):
        """Test precomputed frequency shapes."""
        cos, sin = precompute_rope_frequencies(dim=64, max_seq_len=1024)
        assert cos.shape == (1024, 64)
        assert sin.shape == (1024, 64)

    def test_rope_preserves_shape(self):
        """Test RoPE preserves input shape."""
        cos, sin = precompute_rope_frequencies(dim=64, max_seq_len=128)
        x = torch.randn(2, 64, 8, 64)  # (batch, seq, heads, head_dim)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention."""

    def test_output_shape(self):
        """Test attention output shape."""
        attn = MultiHeadAttention(
            embed_dim=512,
            num_heads=8,
            num_kv_heads=2,
            use_qk_norm=True,
        )
        x = torch.randn(2, 128, 512)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_qk_norm(self):
        """Test QK-norm is applied when enabled."""
        attn = MultiHeadAttention(
            embed_dim=512,
            num_heads=8,
            num_kv_heads=2,
            use_qk_norm=True,
        )
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')

    def test_gqa_shapes(self):
        """Test GQA projection shapes."""
        attn = MultiHeadAttention(
            embed_dim=512,
            num_heads=8,
            num_kv_heads=2,  # 4:1 ratio
        )
        # Q should project to num_heads * head_dim
        assert attn.q_proj.out_features == 512
        # K, V should project to num_kv_heads * head_dim
        assert attn.k_proj.out_features == 128
        assert attn.v_proj.out_features == 128

    def test_causal_masking(self):
        """Test causal attention masking."""
        attn = MultiHeadAttention(
            embed_dim=256,
            num_heads=4,
            num_kv_heads=2,
        )
        x = torch.randn(1, 8, 256)
        out, _ = attn(x)
        # Output should not have NaN (causal mask applied correctly)
        assert not torch.isnan(out).any()


class TestFeedForward:
    """Tests for Feed-Forward Network."""

    def test_output_shape(self):
        """Test FFN output shape."""
        ffn = FeedForward(embed_dim=512, hidden_dim=2048)
        x = torch.randn(2, 128, 512)
        out = ffn(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    """Tests for Transformer Block."""

    def test_output_shape(self):
        """Test block output shape."""
        config = MobileLLM140MConfig()
        block = TransformerBlock(config, layer_idx=0)
        x = torch.randn(2, 64, config.embed_dim)
        out, _ = block(x)
        assert out.shape == x.shape


class TestMobileLLM:
    """Tests for full MobileLLM model."""

    def test_forward_pass(self):
        """Test basic forward pass."""
        config = MobileLLM140MConfig()
        config.max_seq_len = 256  # Smaller for testing
        model = MobileLLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 64))
        outputs = model(input_ids, return_dict=True)

        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 64, config.vocab_size)

    def test_loss_computation(self):
        """Test loss computation."""
        config = MobileLLM140MConfig()
        config.max_seq_len = 256
        model = MobileLLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 64))
        labels = input_ids.clone()

        loss, metrics = model.compute_loss(input_ids, labels)

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert "perplexity" in metrics

    def test_weight_sharing(self):
        """Test input/output embedding weight sharing."""
        config = MobileLLM140MConfig()
        config.use_weight_sharing = True
        model = MobileLLM(config)

        # Should share weights
        assert model.embed_tokens.weight is model.lm_head.weight

    def test_parameter_count(self):
        """Test parameter counting."""
        config = MobileLLM140MConfig()
        model = MobileLLM(config)

        num_params = model.num_parameters()
        # 140M model should have ~140M parameters
        assert 100_000_000 < num_params < 200_000_000

    def test_from_config_string(self):
        """Test creating model from config name."""
        model = MobileLLM.from_config("140m")
        assert isinstance(model, MobileLLM)
        assert model.config.num_layers == 15


class TestConfig:
    """Tests for model configuration."""

    def test_config_validation(self):
        """Test config validation."""
        # Valid config
        config = MobileLLMConfig(
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,
            embed_dim=768,
            hidden_dim=3072,
        )
        assert config.head_dim == 64
        assert config.num_kv_groups == 3

    def test_invalid_head_config(self):
        """Test invalid head configuration raises error."""
        with pytest.raises(AssertionError):
            MobileLLMConfig(
                num_layers=12,
                num_heads=12,
                num_kv_heads=5,  # Not divisible
                embed_dim=768,
                hidden_dim=3072,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
