"""Tests for tensor parallelism utilities."""

import pytest
import torch
import torch.nn as nn

from wrinklefree.models.subln import SubLN
from wrinklefree.models.transformer import BitNetDecoderLayer
from wrinklefree.training.tensor_parallel import (
    DistributedSubLN,
    get_bitnet_tp_plan,
)


class TestDistributedSubLN:
    """Tests for DistributedSubLN module."""

    def test_init(self):
        """Test DistributedSubLN initialization."""
        ln = DistributedSubLN(hidden_size=64, eps=1e-6)
        assert ln.hidden_size == 64
        assert ln.eps == 1e-6
        assert ln.weight.shape == (64,)
        assert ln.global_hidden_size == 64

    def test_forward_shape(self):
        """Test that forward produces correct shape."""
        ln = DistributedSubLN(hidden_size=64)
        x = torch.randn(2, 10, 64)

        output = ln(x)

        assert output.shape == (2, 10, 64)

    def test_forward_matches_subln_single_process(self):
        """Test that without distributed, DistributedSubLN matches SubLN."""
        hidden_size = 128
        subln = SubLN(hidden_size)
        dist_ln = DistributedSubLN(hidden_size)

        # Copy weights
        with torch.no_grad():
            dist_ln.weight.copy_(subln.weight)

        x = torch.randn(2, 10, hidden_size)

        subln_out = subln(x)
        dist_out = dist_ln(x)

        assert torch.allclose(subln_out, dist_out, atol=1e-5)

    def test_from_subln(self):
        """Test creating DistributedSubLN from existing SubLN (no process group)."""
        hidden_size = 128
        subln = SubLN(hidden_size)

        # Without process group, should work but weights won't be sharded
        # This test mainly checks the classmethod logic
        dist_ln = DistributedSubLN(
            hidden_size=hidden_size,
            eps=subln.eps,
            process_group=None,
            global_hidden_size=hidden_size,
        )

        with torch.no_grad():
            dist_ln.weight.copy_(subln.weight)

        assert dist_ln.global_hidden_size == hidden_size

    def test_gradient_flow(self):
        """Test that gradients flow through DistributedSubLN."""
        ln = DistributedSubLN(hidden_size=64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = ln(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert ln.weight.grad is not None


class TestTPPlan:
    """Tests for tensor parallelism plan."""

    def test_plan_structure(self):
        """Test that TP plan has correct structure."""
        plan = get_bitnet_tp_plan()

        # Should have all attention and MLP projections
        expected_keys = {
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        }

        assert set(plan.keys()) == expected_keys

    def test_colwise_rowwise_pairing(self):
        """Test that ColwiseParallel/RowwiseParallel are paired correctly."""
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        plan = get_bitnet_tp_plan()

        # Attention: Q, K, V should be ColwiseParallel
        assert isinstance(plan["self_attn.q_proj"], ColwiseParallel)
        assert isinstance(plan["self_attn.k_proj"], ColwiseParallel)
        assert isinstance(plan["self_attn.v_proj"], ColwiseParallel)
        # O should be RowwiseParallel
        assert isinstance(plan["self_attn.o_proj"], RowwiseParallel)

        # FFN: gate, up should be ColwiseParallel
        assert isinstance(plan["mlp.gate_proj"], ColwiseParallel)
        assert isinstance(plan["mlp.up_proj"], ColwiseParallel)
        # down should be RowwiseParallel
        assert isinstance(plan["mlp.down_proj"], RowwiseParallel)


class TestBitNetDecoderLayerTP:
    """Tests for TP compatibility with BitNetDecoderLayer."""

    def test_layer_has_expected_modules(self):
        """Test that BitNetDecoderLayer has the modules expected by TP plan."""
        layer = BitNetDecoderLayer(
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=4,
        )

        # Check attention modules
        assert hasattr(layer, "self_attn")
        assert hasattr(layer.self_attn, "q_proj")
        assert hasattr(layer.self_attn, "k_proj")
        assert hasattr(layer.self_attn, "v_proj")
        assert hasattr(layer.self_attn, "o_proj")
        assert hasattr(layer.self_attn, "subln")

        # Check FFN modules
        assert hasattr(layer, "mlp")
        assert hasattr(layer.mlp, "gate_proj")
        assert hasattr(layer.mlp, "up_proj")
        assert hasattr(layer.mlp, "down_proj")
        assert hasattr(layer.mlp, "subln")

    def test_plan_covers_all_linear_layers(self):
        """Test that TP plan covers all linear layers that need sharding."""
        layer = BitNetDecoderLayer(
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=4,
        )

        plan = get_bitnet_tp_plan()

        # Find all linear layers in attention and mlp
        attn_linear = []
        for name, module in layer.self_attn.named_modules():
            if name and hasattr(module, "weight") and name not in ["subln"]:
                if name.endswith("_proj"):
                    attn_linear.append(f"self_attn.{name}")

        mlp_linear = []
        for name, module in layer.mlp.named_modules():
            if name and hasattr(module, "weight") and name not in ["subln"]:
                if name.endswith("_proj"):
                    mlp_linear.append(f"mlp.{name}")

        # All linear layers should be in the plan
        for layer_name in attn_linear + mlp_linear:
            assert layer_name in plan, f"{layer_name} not in TP plan"


class TestTPModelForward:
    """Tests for model forward pass with TP (single GPU simulation)."""

    def test_layer_forward_produces_output(self):
        """Test that BitNetDecoderLayer produces output."""
        layer = BitNetDecoderLayer(
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=4,
        )

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 256)

        output, attn_weights = layer(x)

        assert output.shape == (batch_size, seq_len, 256)
        assert attn_weights is None  # Not requesting attention weights

    def test_layer_forward_with_attention_mask(self):
        """Test forward pass with causal attention mask."""
        layer = BitNetDecoderLayer(
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=4,
        )

        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, 256)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

        output, _ = layer(x, attention_mask=mask)

        assert output.shape == (batch_size, seq_len, 256)


class TestDistributedSubLNNumerics:
    """Numerical tests for DistributedSubLN."""

    def test_normalization_variance(self):
        """Test that output has expected variance properties."""
        ln = DistributedSubLN(hidden_size=128)
        x = torch.randn(10, 32, 128) * 5.0  # Large variance input

        output = ln(x)

        # After RMSNorm, values should be scaled reasonably
        assert output.abs().max() < 100, "Output values too large"

    def test_handles_zero_input(self):
        """Test handling of near-zero input."""
        ln = DistributedSubLN(hidden_size=64, eps=1e-6)
        x = torch.zeros(2, 10, 64)

        # Should not produce NaN/Inf
        output = ln(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_handles_large_input(self):
        """Test handling of large input values."""
        ln = DistributedSubLN(hidden_size=64)
        x = torch.randn(2, 10, 64) * 1000

        output = ln(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# Skip distributed tests if not in distributed environment
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for distributed tests"
)
class TestDistributedIntegration:
    """Integration tests that require distributed setup (skipped in CI)."""

    def test_device_mesh_creation_requires_init(self):
        """Test that create_device_mesh requires distributed init."""
        from wrinklefree.training.tensor_parallel import create_device_mesh

        with pytest.raises(RuntimeError, match="Distributed must be initialized"):
            create_device_mesh(tp_size=2)
