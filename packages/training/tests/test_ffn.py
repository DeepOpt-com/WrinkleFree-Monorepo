"""Tests for BitNet Feed-Forward Network module."""

import pytest
import torch

from wrinklefree.models.ffn import BitNetFFN, BitNetMLP


class TestBitNetFFN:
    """Tests for BitNet FFN (gated variant)."""

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        hidden_size = 256
        intermediate_size = 1024
        batch, seq = 2, 16

        ffn = BitNetFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        x = torch.randn(batch, seq, hidden_size)
        output = ffn(x)

        assert output.shape == (batch, seq, hidden_size)

    def test_relu2_activation(self):
        """Test that ReLU^2 activation is applied correctly."""
        ffn = BitNetFFN(
            hidden_size=64,
            intermediate_size=256,
            hidden_act="relu2",
        )

        # Test the activation function directly
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 4.0])  # relu(x)^2

        result = ffn._relu_squared(x)
        assert torch.allclose(result, expected)

    def test_silu_activation(self):
        """Test that SiLU activation can be used."""
        hidden_size = 64
        intermediate_size = 256
        batch, seq = 2, 8

        ffn = BitNetFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
        )

        x = torch.randn(batch, seq, hidden_size)
        output = ffn(x)

        assert output.shape == (batch, seq, hidden_size)

    def test_invalid_activation_raises(self):
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            BitNetFFN(
                hidden_size=64,
                intermediate_size=256,
                hidden_act="invalid",
            )

    def test_gradient_flow(self):
        """Test that gradients flow through FFN."""
        hidden_size = 64
        intermediate_size = 256
        batch, seq = 2, 8

        ffn = BitNetFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        x = torch.randn(batch, seq, hidden_size, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert ffn.gate_proj.weight.grad is not None
        assert ffn.up_proj.weight.grad is not None
        assert ffn.down_proj.weight.grad is not None

    def test_subln_is_applied(self):
        """Test that SubLN is present and applied."""
        ffn = BitNetFFN(
            hidden_size=64,
            intermediate_size=256,
        )

        # SubLN should be a submodule
        assert hasattr(ffn, "subln")
        assert ffn.subln is not None

    def test_different_intermediate_sizes(self):
        """Test FFN with various intermediate sizes."""
        hidden_size = 64
        batch, seq = 2, 8

        for intermediate_size in [128, 256, 512, 1024]:
            ffn = BitNetFFN(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )

            x = torch.randn(batch, seq, hidden_size)
            output = ffn(x)

            assert output.shape == (batch, seq, hidden_size)

    def test_gating_effect(self):
        """Test that gating has an effect on output."""
        hidden_size = 64
        intermediate_size = 256

        ffn = BitNetFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        x = torch.randn(2, 8, hidden_size)

        with torch.no_grad():
            # Modify gate projection to all zeros (should zero out output mostly)
            original_gate_weight = ffn.gate_proj.weight.clone()
            ffn.gate_proj.weight.zero_()

            output_zeroed_gate = ffn(x)

            # Restore and get normal output
            ffn.gate_proj.weight.copy_(original_gate_weight)
            output_normal = ffn(x)

        # Output should be different
        assert not torch.allclose(output_zeroed_gate, output_normal, atol=1e-3)


class TestBitNetMLP:
    """Tests for BitNet MLP (non-gated variant)."""

    def test_forward_shape(self):
        """Test that forward produces correct output shape."""
        hidden_size = 64
        intermediate_size = 256
        batch, seq = 2, 8

        mlp = BitNetMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        x = torch.randn(batch, seq, hidden_size)
        output = mlp(x)

        assert output.shape == (batch, seq, hidden_size)

    def test_relu2_activation(self):
        """Test MLP with ReLU^2 activation."""
        mlp = BitNetMLP(
            hidden_size=64,
            intermediate_size=256,
            hidden_act="relu2",
        )

        x = torch.randn(2, 8, 64)
        output = mlp(x)

        assert output.shape == x.shape

    def test_relu_activation(self):
        """Test MLP with standard ReLU activation."""
        mlp = BitNetMLP(
            hidden_size=64,
            intermediate_size=256,
            hidden_act="relu",
        )

        x = torch.randn(2, 8, 64)
        output = mlp(x)

        assert output.shape == x.shape

    def test_gelu_activation(self):
        """Test MLP with GELU activation."""
        mlp = BitNetMLP(
            hidden_size=64,
            intermediate_size=256,
            hidden_act="gelu",
        )

        x = torch.randn(2, 8, 64)
        output = mlp(x)

        assert output.shape == x.shape

    def test_invalid_activation_raises(self):
        """Test that invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            BitNetMLP(
                hidden_size=64,
                intermediate_size=256,
                hidden_act="invalid",
            )

    def test_gradient_flow(self):
        """Test that gradients flow through MLP."""
        mlp = BitNetMLP(
            hidden_size=64,
            intermediate_size=256,
        )

        x = torch.randn(2, 8, 64, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None

    def test_subln_is_applied(self):
        """Test that SubLN is present."""
        mlp = BitNetMLP(
            hidden_size=64,
            intermediate_size=256,
        )

        assert hasattr(mlp, "subln")
        assert mlp.subln is not None


class TestFFNEdgeCases:
    """Edge case tests for FFN modules."""

    def test_single_token_input(self):
        """Test FFN with single token input."""
        ffn = BitNetFFN(hidden_size=64, intermediate_size=256)
        x = torch.randn(1, 1, 64)
        output = ffn(x)
        assert output.shape == (1, 1, 64)

    def test_large_batch_size(self):
        """Test FFN with large batch size."""
        ffn = BitNetFFN(hidden_size=64, intermediate_size=256)
        x = torch.randn(128, 32, 64)
        output = ffn(x)
        assert output.shape == (128, 32, 64)

    def test_3d_vs_2d_input(self):
        """Test that FFN handles both 2D and 3D inputs."""
        ffn = BitNetFFN(hidden_size=64, intermediate_size=256)

        # 3D input (batch, seq, hidden)
        x_3d = torch.randn(2, 8, 64)
        output_3d = ffn(x_3d)
        assert output_3d.shape == (2, 8, 64)

        # 2D input (batch*seq, hidden)
        x_2d = torch.randn(16, 64)
        output_2d = ffn(x_2d)
        assert output_2d.shape == (16, 64)

    def test_deterministic_output(self):
        """Test that FFN produces deterministic output in eval mode."""
        ffn = BitNetFFN(hidden_size=64, intermediate_size=256)
        ffn.eval()

        x = torch.randn(2, 8, 64)

        with torch.no_grad():
            output1 = ffn(x)
            output2 = ffn(x)

        assert torch.allclose(output1, output2)

    def test_mixed_precision(self):
        """Test FFN with different dtypes."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            ffn = BitNetFFN(hidden_size=64, intermediate_size=256)
            ffn = ffn.to(dtype)

            x = torch.randn(2, 8, 64, dtype=dtype)
            output = ffn(x)

            assert output.dtype == dtype
            assert torch.isfinite(output).all()
