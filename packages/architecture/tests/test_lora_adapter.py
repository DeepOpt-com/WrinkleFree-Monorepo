"""Tests for LoRAAdapter composable wrapper.

Tests the new LoRA adapter pattern that enables orthogonal composition
of LoRA with any BitLinear variant (BitLinear, BitLinearSalient, etc.).
"""

import pytest
import torch
import torch.nn as nn

from wf_arch.layers import (
    BitLinear,
    BitLinearSalient,
    LoRAAdapter,
    LoRAConfig,
    add_lora_to_model,
    freeze_base_model,
    remove_lora_from_model,
    get_lora_stats,
)


class TestLoRAConfig:
    """Test LoRAConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LoRAConfig()
        assert config.rank is None
        assert config.rank_percentage == 0.1
        assert config.alpha == 1.0
        assert config.dropout == 0.0
        assert config.init_method == "kaiming"
        assert config.quantized is False

    def test_explicit_rank(self):
        """Test explicit rank configuration."""
        config = LoRAConfig(rank=32)
        assert config.rank == 32

    def test_rank_percentage(self):
        """Test rank percentage configuration."""
        config = LoRAConfig(rank_percentage=0.05)
        assert config.rank_percentage == 0.05

    def test_invalid_rank_both_specified(self):
        """Test that specifying both rank and rank_percentage raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            LoRAConfig(rank=32, rank_percentage=0.05)

    def test_invalid_init_method(self):
        """Test invalid init method raises error."""
        with pytest.raises(ValueError, match="init_method must be"):
            LoRAConfig(init_method="invalid")


class TestLoRAAdapter:
    """Test LoRAAdapter wrapper class."""

    @pytest.fixture
    def base_layer(self):
        """Create a base BitLinear layer for testing."""
        return BitLinear(in_features=64, out_features=32)

    @pytest.fixture
    def config(self):
        """Create default LoRA config."""
        return LoRAConfig(rank=8, init_method="kaiming")

    def test_initialization(self, base_layer, config):
        """Test LoRAAdapter initialization."""
        lora = LoRAAdapter(base_layer, config)

        assert lora.rank == 8
        assert lora.in_features == 64
        assert lora.out_features == 32
        assert lora.scaling == config.alpha / 8

    def test_rank_from_percentage(self, base_layer):
        """Test rank computed from percentage."""
        config = LoRAConfig(rank_percentage=0.25)
        lora = LoRAAdapter(base_layer, config)

        # min(64, 32) * 0.25 = 8
        assert lora.rank == 8

    def test_forward_shape(self, base_layer, config):
        """Test forward pass output shape."""
        lora = LoRAAdapter(base_layer, config)
        x = torch.randn(2, 16, 64)  # (batch, seq, in_features)

        output = lora(x)

        assert output.shape == (2, 16, 32)

    def test_forward_gradient_flow(self, base_layer, config):
        """Test gradients flow through LoRA matrices."""
        lora = LoRAAdapter(base_layer, config)
        x = torch.randn(2, 16, 64, requires_grad=True)

        output = lora(x)
        loss = output.sum()
        loss.backward()

        # LoRA matrices should have gradients
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None

    def test_kaiming_init_zero_output(self, base_layer):
        """Test Kaiming init gives zero initial LoRA output (B=0)."""
        config = LoRAConfig(rank=8, init_method="kaiming")
        lora = LoRAAdapter(base_layer, config)
        x = torch.randn(2, 16, 64)

        # With Kaiming init, B is zeros, so LoRA output should be zero
        lora_out = lora.lora_B(lora.lora_A(x)) * lora.scaling
        assert torch.allclose(lora_out, torch.zeros_like(lora_out))

        # But full output should equal base output
        base_out = base_layer(x)
        full_out = lora(x)
        assert torch.allclose(base_out, full_out)

    def test_zeros_init(self, base_layer):
        """Test zeros init gives zero LoRA contribution."""
        config = LoRAConfig(rank=8, init_method="zeros")
        lora = LoRAAdapter(base_layer, config)

        # Both A and B should be zeros
        assert torch.allclose(lora.lora_A.weight, torch.zeros_like(lora.lora_A.weight))
        assert torch.allclose(lora.lora_B.weight, torch.zeros_like(lora.lora_B.weight))

    def test_dropout(self, base_layer):
        """Test dropout is applied in LoRA path."""
        config = LoRAConfig(rank=8, dropout=0.5, init_method="zeros")
        lora = LoRAAdapter(base_layer, config)

        # Manually set non-zero weights so dropout has effect
        with torch.no_grad():
            lora.lora_A.weight.fill_(1.0)
            lora.lora_B.weight.fill_(0.1)

        # In training mode, dropout should be active
        lora.train()
        x = torch.randn(2, 16, 64)

        # Run multiple times - outputs should differ due to dropout
        outputs = [lora(x).detach().clone() for _ in range(10)]
        # At least some should be different (probabilistic)
        all_same = all(torch.allclose(outputs[0], o, atol=1e-5) for o in outputs[1:])
        # With 50% dropout, very unlikely all 10 runs are identical
        assert not all_same, "Dropout should cause output variation"

    def test_get_lora_weights(self, base_layer, config):
        """Test getting LoRA weights for export."""
        lora = LoRAAdapter(base_layer, config)
        A, B = lora.get_lora_weights()

        assert A.shape == (8, 64)  # (rank, in_features)
        assert B.shape == (32, 8)  # (out_features, rank)


class TestLoRAWithBitLinearSalient:
    """Test LoRA wrapping BitLinearSalient layers."""

    @pytest.fixture
    def salient_layer(self):
        """Create a BitLinearSalient layer."""
        layer = BitLinearSalient(in_features=64, out_features=32, salient_ratio=0.1)
        # Mark as calibrated with some indices
        num_salient = max(1, int(64 * 0.1))
        salient_indices = torch.arange(num_salient)
        layer.set_salient_columns(salient_indices)
        return layer

    def test_wrap_salient_layer(self, salient_layer):
        """Test LoRA can wrap BitLinearSalient."""
        config = LoRAConfig(rank=8)
        lora = LoRAAdapter(salient_layer, config)

        assert lora.base_layer is salient_layer
        assert lora.rank == 8

    def test_forward_with_salient(self, salient_layer):
        """Test forward pass with salient base layer."""
        config = LoRAConfig(rank=8)
        lora = LoRAAdapter(salient_layer, config)
        x = torch.randn(2, 16, 64)

        output = lora(x)

        assert output.shape == (2, 16, 32)

    def test_gradients_with_salient(self, salient_layer):
        """Test gradients flow correctly with salient base layer."""
        config = LoRAConfig(rank=8)
        lora = LoRAAdapter(salient_layer, config)
        x = torch.randn(2, 16, 64, requires_grad=True)

        output = lora(x)
        loss = output.sum()
        loss.backward()

        # LoRA matrices should have gradients
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None


class TestQuantizedLoRA:
    """Test quantized LoRA adapters (QA-LoRA style)."""

    @pytest.fixture
    def base_layer(self):
        return BitLinear(in_features=64, out_features=32)

    def test_quantized_init(self, base_layer):
        """Test quantized LoRA initialization."""
        config = LoRAConfig(rank=8, quantized=True, quant_bits=4, quant_group_size=16)
        lora = LoRAAdapter(base_layer, config)

        assert lora.config.quantized is True
        # Check that QuantizedLinearSTE is used
        from wf_arch.layers.lora_adapter import QuantizedLinearSTE
        assert isinstance(lora.lora_A, QuantizedLinearSTE)
        assert isinstance(lora.lora_B, QuantizedLinearSTE)

    def test_quantized_forward(self, base_layer):
        """Test quantized LoRA forward pass."""
        config = LoRAConfig(rank=8, quantized=True)
        lora = LoRAAdapter(base_layer, config)
        x = torch.randn(2, 16, 64)

        output = lora(x)

        assert output.shape == (2, 16, 32)

    def test_quantized_gradient_flow(self, base_layer):
        """Test gradients flow through quantized LoRA (STE)."""
        config = LoRAConfig(rank=8, quantized=True)
        lora = LoRAAdapter(base_layer, config)
        x = torch.randn(2, 16, 64, requires_grad=True)

        output = lora(x)
        loss = output.sum()
        loss.backward()

        # Gradients should flow via STE
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None


class TestModelLevelUtilities:
    """Test model-level LoRA utilities."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model with BitLinear layers."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = BitLinear(64, 32)
                self.layer2 = BitLinear(32, 16)
                self.norm = nn.LayerNorm(16)

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.norm(x)
                return x

        return SimpleModel()

    def test_add_lora_to_model(self, simple_model):
        """Test adding LoRA to all BitLinear layers."""
        config = LoRAConfig(rank=4)
        model = add_lora_to_model(simple_model, config)

        # Both BitLinear layers should be wrapped
        assert isinstance(model.layer1, LoRAAdapter)
        assert isinstance(model.layer2, LoRAAdapter)
        # LayerNorm should NOT be wrapped
        assert isinstance(model.norm, nn.LayerNorm)

    def test_freeze_base_model(self, simple_model):
        """Test freezing non-LoRA parameters."""
        config = LoRAConfig(rank=4)
        model = add_lora_to_model(simple_model, config)
        stats = freeze_base_model(model)

        assert stats["trainable"] > 0
        assert stats["frozen"] > 0

        # Check that only LoRA params are trainable
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_remove_lora_from_model(self, simple_model):
        """Test removing LoRA wrappers."""
        config = LoRAConfig(rank=4)
        model = add_lora_to_model(simple_model, config)

        # Verify LoRA was added
        assert isinstance(model.layer1, LoRAAdapter)

        # Remove LoRA
        model = remove_lora_from_model(model)

        # Should be back to BitLinear
        assert isinstance(model.layer1, BitLinear)
        assert isinstance(model.layer2, BitLinear)

    def test_get_lora_stats(self, simple_model):
        """Test getting LoRA statistics."""
        config = LoRAConfig(rank=4)
        model = add_lora_to_model(simple_model, config)
        stats = get_lora_stats(model)

        assert stats["num_lora_layers"] == 2
        assert stats["average_rank"] == 4.0
        assert stats["total_lora_params"] > 0
        assert len(stats["layers"]) == 2

    def test_target_modules_regex(self, simple_model):
        """Test target_modules regex filtering."""
        config = LoRAConfig(rank=4, target_modules=["layer1"])
        model = add_lora_to_model(simple_model, config)

        # Only layer1 should be wrapped
        assert isinstance(model.layer1, LoRAAdapter)
        assert isinstance(model.layer2, BitLinear)  # Not wrapped


class TestSVDInitialization:
    """Test SVD-based LoRA initialization."""

    @pytest.fixture
    def base_layer(self):
        return BitLinear(in_features=64, out_features=32)

    def test_svd_init_fallback(self, base_layer):
        """Test SVD init falls back to kaiming when no original weight provided."""
        config = LoRAConfig(rank=8, init_method="svd_residual")
        # No original_weight provided - should warn and use kaiming
        lora = LoRAAdapter(base_layer, config, original_weight=None)

        # Should still work (fell back to kaiming)
        x = torch.randn(2, 16, 64)
        output = lora(x)
        assert output.shape == (2, 16, 32)

    def test_svd_init_with_weight(self, base_layer):
        """Test SVD init with original weight."""
        original_weight = torch.randn(32, 64)  # (out, in)
        config = LoRAConfig(rank=8, init_method="svd_residual")
        lora = LoRAAdapter(base_layer, config, original_weight=original_weight)

        # LoRA weights should be non-zero after SVD init
        A, B = lora.get_lora_weights()

        # At least one should be non-zero (unless original == quantized exactly)
        has_nonzero = not torch.allclose(A, torch.zeros_like(A)) or \
                      not torch.allclose(B, torch.zeros_like(B))
        # This is probabilistic - SVD on random weights should give non-zero
        assert has_nonzero


class TestForwardPassEquivalence:
    """Test that LoRA wrapper maintains correct forward behavior."""

    def test_initial_output_equals_base(self):
        """Test that with kaiming init, initial output equals base."""
        base = BitLinear(64, 32)
        config = LoRAConfig(rank=8, init_method="kaiming")
        lora = LoRAAdapter(base, config)

        x = torch.randn(2, 16, 64)
        base.eval()
        lora.eval()

        base_out = base(x)
        lora_out = lora(x)

        # Should be equal since B is initialized to zero
        assert torch.allclose(base_out, lora_out, atol=1e-5)

    def test_training_changes_output(self):
        """Test that training LoRA changes the output."""
        base = BitLinear(64, 32)
        config = LoRAConfig(rank=8, init_method="kaiming")
        lora = LoRAAdapter(base, config)

        x = torch.randn(2, 16, 64)

        # Initial output
        initial_out = lora(x).detach().clone()

        # Simulate training by modifying B weights
        with torch.no_grad():
            lora.lora_B.weight.add_(torch.randn_like(lora.lora_B.weight) * 0.1)

        # Output should now be different
        new_out = lora(x)
        assert not torch.allclose(initial_out, new_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
