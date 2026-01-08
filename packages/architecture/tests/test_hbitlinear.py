"""Tests for HBitLinear layer with online Hadamard quantization."""

import tempfile

import pytest
import torch
import torch.nn as nn

from wf_arch.layers.bitlinear import BitLinear
from wf_arch.layers.hbitlinear import HBitLinear
from wf_arch.layers.hadamard import hadamard_transform, hadamard_transform_weights, next_power_of_2
from wf_arch.quantization import set_global_lambda_warmup, LambdaWarmup


class TestHadamardTransform:
    """Test Hadamard transform correctness."""

    def test_orthogonality(self):
        """Test H @ H = n*I (Hadamard is orthogonal up to scaling)."""
        n = 64
        x = torch.eye(n)
        h_x = hadamard_transform(x, scale=1.0)
        h_h_x = hadamard_transform(h_x, scale=1.0)
        # H @ H = n*I, so H @ H @ x / n = x
        assert torch.allclose(h_h_x / n, x, atol=1e-5)

    def test_normalized_hadamard(self):
        """Test normalized Hadamard is involutory."""
        n = 128
        x = torch.randn(4, 32, n)
        scale = 1.0 / (n**0.5)
        h_x = hadamard_transform(x, scale=scale)
        h_h_x = hadamard_transform(h_x, scale=scale)
        assert torch.allclose(h_h_x, x, atol=1e-5)

    def test_gradient_flow(self):
        """Test gradients flow through Hadamard."""
        x = torch.randn(2, 16, 64, requires_grad=True)
        y = hadamard_transform(x, scale=1.0)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_sizes(self):
        """Test Hadamard with various power-of-2 sizes."""
        for n in [16, 32, 64, 128, 256, 512, 1024]:
            x = torch.randn(2, 8, n)
            y = hadamard_transform(x, scale=1.0)
            assert y.shape == x.shape

    def test_invalid_size_raises(self):
        """Test that non-power-of-2 raises assertion."""
        x = torch.randn(2, 8, 65)  # Not power of 2
        with pytest.raises(AssertionError):
            hadamard_transform(x, scale=1.0)


class TestNextPowerOf2:
    """Test power-of-2 padding utility."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (64, 64),
            (65, 128),
            (127, 128),
            (128, 128),
            (256, 256),
            (576, 1024),
            (768, 1024),
            (1, 1),
            (2, 2),
            (3, 4),
        ],
    )
    def test_next_power_of_2(self, n, expected):
        assert next_power_of_2(n) == expected


class TestHadamardTransformWeights:
    """Test weight Hadamard transformation."""

    def test_shape_preserved(self):
        """Test output shape matches input."""
        weight = torch.randn(256, 128)
        weight_h = hadamard_transform_weights(weight)
        assert weight_h.shape == weight.shape

    def test_non_power_of_2_input(self):
        """Test with non-power-of-2 in_features."""
        weight = torch.randn(256, 576)  # SmolLM hidden size
        weight_h = hadamard_transform_weights(weight)
        assert weight_h.shape == weight.shape

    def test_gradient_flow(self):
        """Test gradients flow through weight transform."""
        weight = torch.randn(64, 32, requires_grad=True)
        weight_h = hadamard_transform_weights(weight)
        weight_h.sum().backward()
        assert weight.grad is not None


class TestHBitLinear:
    """Test HBitLinear layer functionality."""

    def test_init(self):
        """Test HBitLinear initialization."""
        layer = HBitLinear(128, 256)
        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.padded_in == 128  # Already power of 2
        assert not layer.needs_padding

    def test_init_non_power_of_2(self):
        """Test HBitLinear with non-power-of-2 input."""
        layer = HBitLinear(576, 256)  # SmolLM hidden size
        assert layer.in_features == 576
        assert layer.padded_in == 1024
        assert layer.needs_padding

    def test_forward_shape(self):
        """Test output shape is correct."""
        layer = HBitLinear(128, 256)
        x = torch.randn(4, 32, 128)
        output = layer(x)
        assert output.shape == (4, 32, 256)

    def test_forward_non_power_of_2(self):
        """Test forward with padding."""
        layer = HBitLinear(576, 256)
        x = torch.randn(2, 16, 576)
        output = layer(x)
        assert output.shape == (2, 16, 256)

    def test_gradient_flow_ste(self):
        """Test gradients flow through Hadamard + STE."""
        layer = HBitLinear(64, 32)
        x = torch.randn(2, 16, 64, requires_grad=True)
        output = layer(x)
        output.sum().backward()
        assert x.grad is not None
        assert layer.weight.grad is not None

    def test_lambda_warmup(self):
        """Test lambda warmup integration."""
        layer = HBitLinear(64, 32)
        x = torch.randn(2, 8, 64)

        warmup = LambdaWarmup(warmup_steps=100, min_lambda=0.0)
        set_global_lambda_warmup(warmup)
        output_fp = layer(x).clone()

        for _ in range(100):
            warmup.step()
        output_quant = layer(x).clone()

        assert not torch.allclose(output_fp, output_quant, atol=1e-6)
        set_global_lambda_warmup(None)

    def test_inherits_from_bitlinear(self):
        """Test HBitLinear is a BitLinear subclass."""
        layer = HBitLinear(64, 32)
        assert isinstance(layer, BitLinear)


class TestHBitLinearAdditional:
    """Additional tests per Gemini review."""

    def test_padding_no_leak(self):
        """Test padding doesn't affect result incorrectly."""
        # Use non-power-of-2 dimension
        layer = HBitLinear(576, 256)

        # Ensure deterministic weights
        torch.manual_seed(42)
        layer.weight.data.normal_()

        x1 = torch.ones(2, 8, 576)
        x2 = torch.ones(2, 8, 576) * 2

        out1 = layer(x1)
        out2 = layer(x2)

        # Output should be different (input doubled)
        assert not torch.allclose(out1, out2, atol=1e-5)

    @pytest.mark.parametrize("bias", [True, False])
    def test_forward_with_bias(self, bias):
        """Test forward pass with and without bias."""
        layer = HBitLinear(128, 64, bias=bias)
        x = torch.randn(2, 8, 128)
        output = layer(x)
        assert output.shape == (2, 8, 64)
        if bias:
            assert layer.bias is not None
        else:
            assert layer.bias is None

    def test_serialization(self):
        """Test model with HBitLinear can be saved/loaded."""
        layer = HBitLinear(128, 64)
        x = torch.randn(2, 8, 128)
        out_before = layer(x)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(layer.state_dict(), f.name)
            layer2 = HBitLinear(128, 64)
            layer2.load_state_dict(torch.load(f.name, weights_only=True))
            out_after = layer2(x)

        assert torch.allclose(out_before, out_after)


class TestHBitLinearWithLoRA:
    """Test HBitLinear works correctly with LoRAAdapter."""

    def test_lora_wrapping(self):
        """Test LoRAAdapter can wrap HBitLinear."""
        from wf_arch.layers.lora_adapter import LoRAAdapter, LoRAConfig

        base = HBitLinear(128, 64)
        wrapped = LoRAAdapter(base, LoRAConfig(rank=8))
        x = torch.randn(2, 16, 128)
        output = wrapped(x)
        assert output.shape == (2, 16, 64)

    def test_lora_gradient_flow(self):
        """Test gradients flow through HBitLinear + LoRA."""
        from wf_arch.layers.lora_adapter import LoRAAdapter, LoRAConfig

        base = HBitLinear(128, 64)
        wrapped = LoRAAdapter(base, LoRAConfig(rank=8))
        x = torch.randn(2, 8, 128, requires_grad=True)
        output = wrapped(x)
        output.sum().backward()

        assert x.grad is not None
        assert wrapped.lora_A.weight.grad is not None
        assert wrapped.lora_B.weight.grad is not None


class TestHBitLinearConversion:
    """Test model conversion with use_hadamard flag."""

    def test_convert_linear_to_hbitlinear(self):
        """Test direct conversion of nn.Linear to HBitLinear."""
        from wf_arch.conversion.convert import convert_linear_to_hbitlinear

        linear = nn.Linear(64, 32, bias=False)
        original_weight = linear.weight.data.clone()

        hbit = convert_linear_to_hbitlinear(linear)

        # Verify it's an HBitLinear
        assert isinstance(hbit, HBitLinear)
        assert hbit.in_features == 64
        assert hbit.out_features == 32

        # Verify weights were transformed (not equal to original)
        assert not torch.allclose(hbit.weight.data, original_weight)

        # Verify weights were transformed correctly
        expected = hadamard_transform_weights(original_weight)
        assert torch.allclose(hbit.weight.data, expected, atol=1e-5)

    def test_convert_linear_to_hbitlinear_with_bias(self):
        """Test conversion preserves bias."""
        from wf_arch.conversion.convert import convert_linear_to_hbitlinear

        linear = nn.Linear(64, 32, bias=True)
        original_bias = linear.bias.data.clone()

        hbit = convert_linear_to_hbitlinear(linear)

        assert hbit.bias is not None
        assert torch.allclose(hbit.bias.data, original_bias)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available for model loading"
    )
    def test_convert_model_with_hadamard(self):
        """Test model conversion creates HBitLinear only for o_proj and down_proj."""
        from transformers import AutoModelForCausalLM

        from wf_arch import convert_model_to_bitnet

        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            torch_dtype=torch.bfloat16,
        )
        model = convert_model_to_bitnet(
            model,
            hidden_size=576,
            intermediate_size=1536,
            use_hadamard=True,
            insert_subln=True,
        )

        # Count HBitLinear and BitLinear layers
        hbit_count = 0
        bit_count = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                inner = module[-1]
                if isinstance(inner, HBitLinear):
                    hbit_count += 1
                    # o_proj and down_proj should be HBitLinear
                    assert (
                        "o_proj" in name or "down_proj" in name
                    ), f"Unexpected HBitLinear at {name}"
                elif isinstance(inner, BitLinear):
                    bit_count += 1
                    # q, k, v, gate, up should be BitLinear
                    assert not (
                        "o_proj" in name or "down_proj" in name
                    ), f"Expected HBitLinear at {name}"

        # SmolLM2-135M has 30 layers, each with o_proj and down_proj
        assert hbit_count > 0, "No HBitLinear layers found"
        assert bit_count > 0, "No BitLinear layers found"
