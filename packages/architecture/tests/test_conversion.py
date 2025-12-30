"""Tests for model conversion utilities."""

import pytest
import torch
import torch.nn as nn

from bitnet_arch.layers.bitlinear import BitLinear
from bitnet_arch.layers.subln import SubLN
from bitnet_arch.conversion import (
    is_bitnet_model,
    auto_convert_if_needed,
    convert_model_to_bitnet,
    convert_attention_layer,
    convert_mlp_layer,
)


class SimpleLlamaAttention(nn.Module):
    """Mock LLaMA-style attention for testing."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class SimpleLlamaMLP(nn.Module):
    """Mock LLaMA-style MLP for testing."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 512):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class SimpleLlamaLayer(nn.Module):
    """Mock LLaMA transformer layer."""

    def __init__(self, hidden_size: int = 256, intermediate_size: int = 512):
        super().__init__()
        self.self_attn = SimpleLlamaAttention(hidden_size)
        self.mlp = SimpleLlamaMLP(hidden_size, intermediate_size)


class SimpleLlamaModel(nn.Module):
    """Mock LLaMA model for testing conversion."""

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        num_layers: int = 2,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleLlamaLayer(hidden_size, intermediate_size)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


class TestIsBitnetModel:
    """Test is_bitnet_model detection."""

    def test_regular_model_is_not_bitnet(self):
        """Test that regular model is not detected as BitNet."""
        model = SimpleLlamaModel()
        assert not is_bitnet_model(model)

    def test_model_with_bitlinear_is_bitnet(self):
        """Test that model with BitLinear is detected as BitNet."""
        model = nn.Sequential(
            nn.Linear(64, 32),
            BitLinear(32, 16),
        )
        assert is_bitnet_model(model)

    def test_nested_bitlinear_detected(self):
        """Test that nested BitLinear is detected."""
        model = nn.ModuleDict({
            "encoder": nn.Sequential(
                nn.Linear(64, 32),
                nn.ModuleDict({
                    "proj": BitLinear(32, 16)
                })
            )
        })
        assert is_bitnet_model(model)


class TestAutoConvertIfNeeded:
    """Test auto_convert_if_needed function."""

    def test_converts_non_bitnet_model(self):
        """Test that non-BitNet model gets converted."""
        model = SimpleLlamaModel(hidden_size=128, intermediate_size=256)

        converted = auto_convert_if_needed(
            model,
            hidden_size=128,
            intermediate_size=256,
        )

        assert is_bitnet_model(converted)

    def test_skips_already_bitnet_model(self):
        """Test that BitNet model is not re-converted."""
        # Create a model with BitLinear
        model = nn.Sequential(BitLinear(64, 32))

        # Should return same model
        result = auto_convert_if_needed(model, hidden_size=64, intermediate_size=128)

        assert result is model


class TestConvertAttentionLayer:
    """Test attention layer conversion."""

    def test_converts_projections_to_bitlinear(self):
        """Test that Q, K, V, O projections become BitLinear."""
        attn = SimpleLlamaAttention(hidden_size=128)

        convert_attention_layer(attn, hidden_size=128, exclude_layers=[])

        assert isinstance(attn.q_proj, BitLinear)
        assert isinstance(attn.k_proj, BitLinear)
        assert isinstance(attn.v_proj, BitLinear)

    def test_inserts_subln_before_o_proj(self):
        """Test that SubLN is inserted before o_proj."""
        attn = SimpleLlamaAttention(hidden_size=128)

        convert_attention_layer(attn, hidden_size=128, exclude_layers=[])

        # o_proj should now be Sequential(SubLN, BitLinear)
        assert isinstance(attn.o_proj, nn.Sequential)
        assert isinstance(attn.o_proj[0], SubLN)
        assert isinstance(attn.o_proj[1], BitLinear)

    def test_respects_exclude_layers(self):
        """Test that excluded layers are not converted."""
        attn = SimpleLlamaAttention(hidden_size=128)
        original_q_proj = attn.q_proj

        convert_attention_layer(attn, hidden_size=128, exclude_layers=["q_proj"])

        # q_proj should remain unchanged
        assert attn.q_proj is original_q_proj
        assert isinstance(attn.k_proj, BitLinear)

    def test_preserves_weights(self):
        """Test that weight values are preserved during conversion."""
        attn = SimpleLlamaAttention(hidden_size=64)
        original_weight = attn.q_proj.weight.data.clone()

        convert_attention_layer(attn, hidden_size=64, exclude_layers=[])

        assert torch.allclose(attn.q_proj.weight.data, original_weight)


class TestConvertMLPLayer:
    """Test MLP layer conversion."""

    def test_converts_projections_to_bitlinear(self):
        """Test that gate and up projections become BitLinear."""
        mlp = SimpleLlamaMLP(hidden_size=128, intermediate_size=256)

        convert_mlp_layer(mlp, hidden_size=256, exclude_layers=[])

        assert isinstance(mlp.gate_proj, BitLinear)
        assert isinstance(mlp.up_proj, BitLinear)

    def test_inserts_subln_before_down_proj(self):
        """Test that SubLN is inserted before down_proj."""
        mlp = SimpleLlamaMLP(hidden_size=128, intermediate_size=256)

        convert_mlp_layer(mlp, hidden_size=256, exclude_layers=[])

        # down_proj should now be Sequential(SubLN, BitLinear)
        assert isinstance(mlp.down_proj, nn.Sequential)
        assert isinstance(mlp.down_proj[0], SubLN)
        assert isinstance(mlp.down_proj[1], BitLinear)


class TestConvertModelToBitnet:
    """Test full model conversion."""

    def test_converts_full_model(self):
        """Test conversion of full LLaMA-style model."""
        model = SimpleLlamaModel(
            hidden_size=128,
            intermediate_size=256,
            num_layers=2,
        )

        converted = convert_model_to_bitnet(
            model,
            hidden_size=128,
            intermediate_size=256,
        )

        # Check that model is now BitNet
        assert is_bitnet_model(converted)

        # Check attention layers
        for layer in converted.layers:
            assert isinstance(layer.self_attn.q_proj, BitLinear)
            assert isinstance(layer.self_attn.o_proj, nn.Sequential)

            # Check MLP layers
            assert isinstance(layer.mlp.gate_proj, BitLinear)
            assert isinstance(layer.mlp.down_proj, nn.Sequential)

    def test_preserves_embeddings(self):
        """Test that embeddings are not converted."""
        model = SimpleLlamaModel(hidden_size=128, intermediate_size=256)
        original_embed = model.embed_tokens
        original_lm_head = model.lm_head

        convert_model_to_bitnet(model, hidden_size=128, intermediate_size=256)

        # Embeddings should remain unchanged
        assert model.embed_tokens is original_embed
        assert model.lm_head is original_lm_head

    def test_subln_placement_in_attention(self):
        """Test SubLN is correctly placed before o_proj."""
        model = SimpleLlamaModel(hidden_size=128, intermediate_size=256)

        convert_model_to_bitnet(model, hidden_size=128, intermediate_size=256)

        # Verify SubLN comes before the projection in o_proj Sequential
        o_proj = model.layers[0].self_attn.o_proj
        assert isinstance(o_proj, nn.Sequential)
        assert len(o_proj) == 2
        assert isinstance(o_proj[0], SubLN)
        assert isinstance(o_proj[1], BitLinear)

    def test_subln_placement_in_mlp(self):
        """Test SubLN is correctly placed before down_proj."""
        model = SimpleLlamaModel(hidden_size=128, intermediate_size=256)

        convert_model_to_bitnet(model, hidden_size=128, intermediate_size=256)

        # Verify SubLN comes before the projection in down_proj Sequential
        down_proj = model.layers[0].mlp.down_proj
        assert isinstance(down_proj, nn.Sequential)
        assert len(down_proj) == 2
        assert isinstance(down_proj[0], SubLN)
        assert isinstance(down_proj[1], BitLinear)
