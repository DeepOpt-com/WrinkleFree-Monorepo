"""Tests for model conversion utilities."""

import pytest
import torch
import torch.nn as nn

from wf_arch.layers.bitlinear import BitLinear
from wf_arch.layers.subln import SubLN
from wf_arch.conversion import (
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

    def test_inserts_subln_before_all_projections(self):
        """Test that SubLN is inserted before ALL projections (q/k/v/o).

        Per BitNet 1.58b paper: "we add LN immediately before the quantization function"
        This means SubLN before EVERY BitLinear, not just output projections.
        """
        attn = SimpleLlamaAttention(hidden_size=128)

        convert_attention_layer(attn, hidden_size=128, exclude_layers=[])

        # ALL projections should now be Sequential(SubLN, BitLinear)
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(attn, proj_name)
            assert isinstance(proj, nn.Sequential), f"{proj_name} should be Sequential"
            assert isinstance(proj[0], SubLN), f"{proj_name} should have SubLN first"
            assert isinstance(proj[1], BitLinear), f"{proj_name} should have BitLinear second"

    def test_subln_has_correct_dimensions(self):
        """Test that SubLN has correct input dimensions for each projection."""
        attn = SimpleLlamaAttention(hidden_size=128)

        convert_attention_layer(attn, hidden_size=128, exclude_layers=[])

        # q/k/v/o all take hidden_size as input
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(attn, proj_name)
            subln = proj[0]
            assert subln.hidden_size == 128, f"{proj_name} SubLN should have hidden_size=128"

    def test_respects_exclude_layers(self):
        """Test that excluded layers are not converted."""
        attn = SimpleLlamaAttention(hidden_size=128)
        original_q_proj = attn.q_proj

        convert_attention_layer(attn, hidden_size=128, exclude_layers=["q_proj"])

        # q_proj should remain unchanged
        assert attn.q_proj is original_q_proj
        # k_proj should still be converted with SubLN
        assert isinstance(attn.k_proj, nn.Sequential)
        assert isinstance(attn.k_proj[0], SubLN)

    def test_preserves_weights(self):
        """Test that weight values are preserved during conversion."""
        attn = SimpleLlamaAttention(hidden_size=64)
        original_weight = attn.q_proj.weight.data.clone()

        convert_attention_layer(attn, hidden_size=64, exclude_layers=[])

        # Weight is now in q_proj[1] (the BitLinear inside Sequential)
        assert torch.allclose(attn.q_proj[1].weight.data, original_weight)

    def test_no_subln_when_disabled(self):
        """Test that SubLN is not inserted when insert_subln=False."""
        attn = SimpleLlamaAttention(hidden_size=128)

        convert_attention_layer(attn, hidden_size=128, exclude_layers=[], insert_subln=False)

        # All projections should be plain BitLinear (no Sequential wrapper)
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            proj = getattr(attn, proj_name)
            assert isinstance(proj, BitLinear), f"{proj_name} should be BitLinear directly"
            assert not isinstance(proj, nn.Sequential), f"{proj_name} should not be Sequential"


class TestConvertMLPLayer:
    """Test MLP layer conversion."""

    def test_inserts_subln_before_all_projections(self):
        """Test that SubLN is inserted before ALL projections (gate/up/down).

        Per BitNet 1.58b paper: "we add LN immediately before the quantization function"
        This means SubLN before EVERY BitLinear, not just output projections.
        """
        mlp = SimpleLlamaMLP(hidden_size=128, intermediate_size=256)

        convert_mlp_layer(mlp, hidden_size=256, exclude_layers=[])

        # ALL projections should now be Sequential(SubLN, BitLinear)
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(mlp, proj_name)
            assert isinstance(proj, nn.Sequential), f"{proj_name} should be Sequential"
            assert isinstance(proj[0], SubLN), f"{proj_name} should have SubLN first"
            assert isinstance(proj[1], BitLinear), f"{proj_name} should have BitLinear second"

    def test_subln_has_correct_dimensions(self):
        """Test that SubLN has correct input dimensions for each projection."""
        mlp = SimpleLlamaMLP(hidden_size=128, intermediate_size=256)

        convert_mlp_layer(mlp, hidden_size=256, exclude_layers=[])

        # gate_proj and up_proj take hidden_size (128) as input
        assert mlp.gate_proj[0].hidden_size == 128
        assert mlp.up_proj[0].hidden_size == 128
        # down_proj takes intermediate_size (256) as input
        assert mlp.down_proj[0].hidden_size == 256

    def test_no_subln_when_disabled(self):
        """Test that SubLN is not inserted when insert_subln=False."""
        mlp = SimpleLlamaMLP(hidden_size=128, intermediate_size=256)

        convert_mlp_layer(mlp, hidden_size=256, exclude_layers=[], insert_subln=False)

        # All projections should be plain BitLinear (no Sequential wrapper)
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(mlp, proj_name)
            assert isinstance(proj, BitLinear), f"{proj_name} should be BitLinear directly"
            assert not isinstance(proj, nn.Sequential), f"{proj_name} should not be Sequential"


class TestConvertModelToBitnet:
    """Test full model conversion."""

    def test_converts_full_model_with_subln_everywhere(self):
        """Test conversion of full LLaMA-style model with SubLN before ALL BitLinear."""
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

        # Check ALL attention projections have SubLN
        for layer in converted.layers:
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(layer.self_attn, proj_name)
                assert isinstance(proj, nn.Sequential), f"{proj_name} should be Sequential"
                assert isinstance(proj[0], SubLN), f"{proj_name} should have SubLN"
                assert isinstance(proj[1], BitLinear), f"{proj_name} should have BitLinear"

            # Check ALL MLP projections have SubLN
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                proj = getattr(layer.mlp, proj_name)
                assert isinstance(proj, nn.Sequential), f"{proj_name} should be Sequential"
                assert isinstance(proj[0], SubLN), f"{proj_name} should have SubLN"
                assert isinstance(proj[1], BitLinear), f"{proj_name} should have BitLinear"

    def test_preserves_embeddings(self):
        """Test that embeddings are not converted."""
        model = SimpleLlamaModel(hidden_size=128, intermediate_size=256)
        original_embed = model.embed_tokens
        original_lm_head = model.lm_head

        convert_model_to_bitnet(model, hidden_size=128, intermediate_size=256)

        # Embeddings should remain unchanged
        assert model.embed_tokens is original_embed
        assert model.lm_head is original_lm_head

    def test_subln_count_matches_bitlinear_count(self):
        """Test that every BitLinear has a corresponding SubLN."""
        model = SimpleLlamaModel(hidden_size=128, intermediate_size=256, num_layers=2)

        convert_model_to_bitnet(model, hidden_size=128, intermediate_size=256)

        # Count SubLN and BitLinear modules
        subln_count = sum(1 for m in model.modules() if isinstance(m, SubLN))
        bitlinear_count = sum(1 for m in model.modules() if isinstance(m, BitLinear))

        # Each layer has 7 projections (q, k, v, o, gate, up, down) = 7 * 2 layers = 14
        assert subln_count == 14, f"Expected 14 SubLN, got {subln_count}"
        assert bitlinear_count == 14, f"Expected 14 BitLinear, got {bitlinear_count}"
        assert subln_count == bitlinear_count, "SubLN count should equal BitLinear count"

    def test_forward_pass_works_after_conversion(self):
        """Test that the converted model can do a forward pass."""
        model = SimpleLlamaModel(hidden_size=128, intermediate_size=256, num_layers=1)

        convert_model_to_bitnet(model, hidden_size=128, intermediate_size=256)

        # Create dummy input
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Get embeddings and pass through layers
        x = model.embed_tokens(input_ids)
        for layer in model.layers:
            # Simplified forward - just check it doesn't crash
            attn_out = layer.self_attn.o_proj(
                layer.self_attn.v_proj(x)  # Simplified
            )
            mlp_out = layer.mlp.down_proj(
                layer.mlp.gate_proj(x) * layer.mlp.up_proj(x)
            )
            x = x + attn_out + mlp_out

        logits = model.lm_head(x)
        assert logits.shape == (batch_size, seq_len, 1000)
