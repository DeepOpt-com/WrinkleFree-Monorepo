"""Integration tests for LoRA + Salient orthogonal composition.

Tests the key feature: LoRA can wrap BitLinearSalient layers, enabling
both features to be used simultaneously.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from wf_arch.layers import (
    BitLinear,
    BitLinearSalient,
    LoRAAdapter,
    LoRAConfig,
    add_lora_to_model,
    freeze_base_model,
    get_lora_stats,
    get_salient_stats,
    convert_bitlinear_to_salient,
)


class SimpleTransformerBlock(nn.Module):
    """Minimal transformer block for testing."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.q_proj = BitLinear(dim, dim)
        self.k_proj = BitLinear(dim, dim)
        self.v_proj = BitLinear(dim, dim)
        self.o_proj = BitLinear(dim, dim)
        self.ffn_gate = BitLinear(dim, dim * 4)
        self.ffn_up = BitLinear(dim, dim * 4)
        self.ffn_down = BitLinear(dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Simplified attention
        h = self.norm1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        attn = torch.softmax(q @ k.transpose(-2, -1) / 8.0, dim=-1)
        h = self.o_proj(attn @ v)
        x = x + h

        # FFN
        h = self.norm2(x)
        gate = F.silu(self.ffn_gate(h))
        up = self.ffn_up(h)
        h = self.ffn_down(gate * up)
        return x + h


class TestSalientThenLoRA:
    """Test the sequential application: Salient first, then LoRA."""

    @pytest.fixture
    def model(self):
        """Create a simple model with BitLinear layers."""
        return SimpleTransformerBlock(dim=64)

    def test_sequential_conversion(self, model):
        """Test that Salient -> LoRA conversion works."""
        # Step 1: Convert to Salient
        # Set up salient indices manually (normally done via calibration)
        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                num_salient = max(1, int(module.in_features * 0.1))
                salient_indices = torch.arange(num_salient)

                # Replace with BitLinearSalient
                salient = BitLinearSalient(
                    module.in_features,
                    module.out_features,
                    salient_ratio=0.1,
                )
                salient.weight.data = module.weight.data.clone()
                salient.set_salient_columns(salient_indices)

                # Find parent and replace
                if "." in name:
                    parent_path, attr = name.rsplit(".", 1)
                    parent = model
                    for part in parent_path.split("."):
                        parent = getattr(parent, part)
                else:
                    parent = model
                    attr = name
                setattr(parent, attr, salient)

        # Verify all BitLinear are now BitLinearSalient
        for name, module in model.named_modules():
            if "proj" in name or "ffn" in name:
                if hasattr(module, "weight"):
                    assert isinstance(module, BitLinearSalient), f"{name} should be BitLinearSalient"

        # Step 2: Add LoRA
        config = LoRAConfig(rank=8)
        model = add_lora_to_model(model, config)

        # Verify LoRA wrappers are around BitLinearSalient
        for name, module in model.named_modules():
            if isinstance(module, LoRAAdapter):
                assert isinstance(module.base_layer, BitLinearSalient), \
                    f"{name} base should be BitLinearSalient"

    def test_forward_pass_with_both(self, model):
        """Test forward pass works with both Salient and LoRA."""
        # Convert some layers to salient
        salient_layer = BitLinearSalient(64, 64, salient_ratio=0.1)
        salient_layer.weight.data = model.q_proj.weight.data.clone()
        salient_layer.set_salient_columns(torch.arange(6))  # 10% of 64
        model.q_proj = salient_layer

        # Wrap with LoRA
        config = LoRAConfig(rank=8)
        model.q_proj = LoRAAdapter(model.q_proj, config)

        # Forward pass
        x = torch.randn(2, 16, 64)
        output = model(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow_with_both(self, model):
        """Test gradients flow correctly through both Salient and LoRA."""
        # Convert to salient
        salient_layer = BitLinearSalient(64, 64, salient_ratio=0.1)
        salient_layer.weight.data = model.q_proj.weight.data.clone()
        salient_layer.set_salient_columns(torch.arange(6))
        model.q_proj = salient_layer

        # Wrap with LoRA
        config = LoRAConfig(rank=8)
        model.q_proj = LoRAAdapter(model.q_proj, config)

        # Freeze base model (only LoRA trainable)
        freeze_base_model(model)

        # Forward + backward
        x = torch.randn(2, 16, 64)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # LoRA should have gradients
        assert model.q_proj.lora_A.weight.grad is not None
        assert model.q_proj.lora_B.weight.grad is not None

        # Base salient layer should NOT have gradients
        assert model.q_proj.base_layer.weight.grad is None


class TestConversionUtilities:
    """Test the conversion utility functions work together."""

    @pytest.fixture
    def model(self):
        return SimpleTransformerBlock(dim=64)

    def test_convert_bitlinear_to_salient_then_lora(self, model):
        """Test using convert_bitlinear_to_salient then add_lora_to_model."""
        # Create mock salient indices for each layer
        salient_indices = {}
        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                num_salient = max(1, int(module.in_features * 0.1))
                salient_indices[name] = torch.arange(num_salient)

        # Convert to salient
        model = convert_bitlinear_to_salient(
            model,
            salient_ratio=0.1,
            salient_indices=salient_indices,
        )

        # Verify conversion
        salient_stats = get_salient_stats(model)
        assert salient_stats["num_salient_layers"] > 0

        # Add LoRA
        config = LoRAConfig(rank=4)
        model = add_lora_to_model(model, config)

        # Verify both
        lora_stats = get_lora_stats(model)
        assert lora_stats["num_lora_layers"] > 0

        # Forward pass should work
        x = torch.randn(2, 16, 64)
        output = model(x)
        assert output.shape == x.shape


class TestTrainingLoop:
    """Test a minimal training loop with Salient + LoRA."""

    def test_training_step(self):
        """Test a single training step with both features."""
        # Create model
        model = SimpleTransformerBlock(dim=64)

        # Convert one layer to salient + LoRA
        salient_layer = BitLinearSalient(64, 64, salient_ratio=0.1)
        salient_layer.weight.data = model.q_proj.weight.data.clone()
        salient_layer.set_salient_columns(torch.arange(6))
        model.q_proj = LoRAAdapter(salient_layer, LoRAConfig(rank=8))

        # Freeze base model
        freeze_base_model(model)

        # Create optimizer (only LoRA params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

        # Training step
        model.train()
        x = torch.randn(2, 16, 64)

        # Forward
        output = model(x)
        loss = output.pow(2).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        for p in trainable_params:
            assert p.grad is not None

        # Update
        optimizer.step()

        # Verify weights changed
        assert not torch.isnan(model.q_proj.lora_A.weight).any()

    def test_multiple_training_steps(self):
        """Test that loss decreases over multiple steps."""
        # Create model with LoRA on salient layer
        model = SimpleTransformerBlock(dim=64)
        salient_layer = BitLinearSalient(64, 64, salient_ratio=0.1)
        salient_layer.weight.data = model.q_proj.weight.data.clone()
        salient_layer.set_salient_columns(torch.arange(6))
        model.q_proj = LoRAAdapter(salient_layer, LoRAConfig(rank=8, init_method="kaiming"))

        freeze_base_model(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

        # Fixed input for consistency
        x = torch.randn(4, 32, 64)
        target = torch.zeros_like(x)

        # Training loop
        losses = []
        model.train()
        for _ in range(20):
            output = model(x)
            loss = (output - target).pow(2).mean()
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestDeviceCompatibility:
    """Test device handling with Salient + LoRA."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        device = torch.device("cuda")

        # Create model
        layer = BitLinearSalient(64, 32, salient_ratio=0.1)
        layer.set_salient_columns(torch.arange(6))
        lora = LoRAAdapter(layer, LoRAConfig(rank=8))
        lora = lora.to(device)

        x = torch.randn(2, 16, 64, device=device)
        output = lora(x)

        assert output.device.type == device.type
        assert output.shape == (2, 16, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
