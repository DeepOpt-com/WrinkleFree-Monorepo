"""Test that our BitLinear produces equivalent outputs to BitNet inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

# Import from our src - no rewriting functionality
from wrinklefree.models.bitlinear import BitLinear
from wrinklefree.models.subln import SubLN


class TestBitLinearQuantization:
    """Test BitLinear quantization methods produce correct outputs."""

    def test_activation_quantization_8bit_range(self):
        """Verify activation quantization produces 8-bit integer values scaled correctly."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 512)
        layer = BitLinear(512, 256)

        x_quant = layer.activation_quant(x)

        # Compute scale the same way (per-token absmax)
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        # Quantized values should be integers when scaled
        x_scaled = x_quant * scale
        # Check they round-trip correctly
        assert torch.allclose(x_scaled.round(), x_scaled, atol=1e-4)

    def test_weight_quantization_produces_ternary(self):
        """Verify weight quantization produces only ternary values {-1, 0, 1}."""
        torch.manual_seed(42)
        layer = BitLinear(512, 256)

        w_quant = layer.weight_quant(layer.weight)

        # Scale to check ternary
        scale = 1.0 / layer.weight.abs().mean().clamp(min=1e-5)
        w_scaled = w_quant * scale
        unique_vals = torch.unique(w_scaled.round())

        # All values must be in {-1, 0, 1}
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals.tolist())

    def test_forward_uses_quantized_weights_and_activations(self):
        """Verify forward pass applies both weight and activation quantization."""
        torch.manual_seed(42)
        x = torch.randn(2, 16, 512)
        layer = BitLinear(512, 256, bias=False)

        # Get quantized versions manually
        x_quant = layer.activation_quant(x)
        w_quant = layer.weight_quant(layer.weight)

        # Manual forward with pre-quantized
        manual_out = F.linear(x_quant, w_quant)

        # Actual forward (STE means output should match for forward pass)
        actual_out = layer(x)

        assert torch.allclose(manual_out, actual_out, atol=1e-5)


class TestSubLNIntegration:
    """Test SubLN + BitLinear integration."""

    def test_subln_sequential_wrapper(self):
        """Verify Sequential(SubLN, BitLinear) works correctly."""
        subln = SubLN(512)
        proj = BitLinear(512, 256)
        wrapped = nn.Sequential(subln, proj)

        x = torch.randn(2, 16, 512)

        # Manual: SubLN first, then projection
        x_normed = subln(x)
        manual_output = proj(x_normed)

        # Sequential application
        seq_output = wrapped(x)

        assert torch.allclose(manual_output, seq_output, atol=1e-6)


class TestStage1Conversion:
    """Test Stage 1 model conversion creates correct architecture."""

    @pytest.mark.slow  # Mark as slow since it downloads a model
    def test_conversion_creates_sequential_wrappers(self):
        """Verify Stage 1 wraps o_proj and down_proj in Sequential(SubLN, BitLinear)."""
        from transformers import AutoModelForCausalLM
        from wrinklefree.training._legacy.stage1 import convert_model_to_bitnet

        # Load small model
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            torch_dtype=torch.float32,
        )

        # Convert
        model = convert_model_to_bitnet(model, hidden_size=576, intermediate_size=1536)

        # Check first layer's o_proj is Sequential(SubLN, BitLinear)
        layer0 = model.model.layers[0]
        assert isinstance(layer0.self_attn.o_proj, nn.Sequential)
        assert len(layer0.self_attn.o_proj) == 2
        assert "SubLN" in type(layer0.self_attn.o_proj[0]).__name__
        assert "BitLinear" in type(layer0.self_attn.o_proj[1]).__name__

        # Check down_proj is Sequential(SubLN, BitLinear)
        assert isinstance(layer0.mlp.down_proj, nn.Sequential)
        assert len(layer0.mlp.down_proj) == 2
        assert "SubLN" in type(layer0.mlp.down_proj[0]).__name__
        assert "BitLinear" in type(layer0.mlp.down_proj[1]).__name__
