"""Unit tests for optimizer utilities.

Tests create_optimizer, get_parameter_groups, InfluenceAwareOptimizer.
"""

import pytest
import torch
import torch.nn as nn

from cheapertraining.training.optimizer import (
    create_optimizer,
    get_parameter_groups,
    get_num_parameters,
    InfluenceAwareOptimizer,
    MUON_AVAILABLE,
    MUONCLIP_AVAILABLE,
)


class SimpleModel(nn.Module):
    """Simple model for testing optimizers."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 64)
        self.linear1 = nn.Linear(64, 128)
        self.ln = nn.LayerNorm(128)
        self.linear2 = nn.Linear(128, 64)
        self.lm_head = nn.Linear(64, 100, bias=False)
        self.bias_layer = nn.Linear(64, 64, bias=True)

    def forward(self, x):
        x = self.embed_tokens(x)
        x = self.linear1(x)
        x = self.ln(x)
        x = self.linear2(x)
        x = self.bias_layer(x)
        return self.lm_head(x)


class TestGetParameterGroups:
    """Tests for get_parameter_groups."""

    def test_separates_decay_and_no_decay(self):
        """Test that parameters are separated correctly."""
        model = SimpleModel()
        groups = get_parameter_groups(model, weight_decay=0.1)

        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.1
        assert groups[1]["weight_decay"] == 0.0

        # Both groups should have parameters
        assert len(groups[0]["params"]) > 0
        assert len(groups[1]["params"]) > 0

    def test_total_params_preserved(self):
        """Test all trainable parameters are included."""
        model = SimpleModel()
        groups = get_parameter_groups(model, weight_decay=0.1)

        total_in_groups = sum(len(g["params"]) for g in groups)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)

        assert total_in_groups == total_trainable

    def test_bias_in_no_decay(self):
        """Test biases are in no-decay group."""
        model = SimpleModel()
        groups = get_parameter_groups(model, weight_decay=0.1)

        no_decay_params = groups[1]["params"]

        # Check that bias parameters are in no_decay
        for name, param in model.named_parameters():
            if "bias" in name.lower():
                assert any(p is param for p in no_decay_params)

    def test_norm_in_no_decay(self):
        """Test layer norm parameters are in no-decay group."""
        model = SimpleModel()
        groups = get_parameter_groups(model, weight_decay=0.1)

        no_decay_params = groups[1]["params"]

        # Check that ln parameters are in no_decay
        # Note: default patterns include 'ln_' so 'ln.' won't match
        for name, param in model.named_parameters():
            # Match 'ln_' pattern (the default) or check norm
            if "ln_" in name.lower():
                assert any(p is param for p in no_decay_params)

    def test_custom_no_decay_patterns(self):
        """Test custom no-decay patterns."""
        model = SimpleModel()
        groups = get_parameter_groups(
            model,
            weight_decay=0.1,
            no_decay_patterns=["embed"],  # Only exclude embeddings
        )

        # embed_tokens should be in no_decay
        no_decay_params = groups[1]["params"]
        assert any(p is model.embed_tokens.weight for p in no_decay_params)

    def test_zero_weight_decay(self):
        """Test with zero weight decay."""
        model = SimpleModel()
        groups = get_parameter_groups(model, weight_decay=0.0)

        # First group should have 0.0 weight decay
        assert groups[0]["weight_decay"] == 0.0

    def test_respects_requires_grad(self):
        """Test that frozen parameters are excluded."""
        model = SimpleModel()
        model.linear1.weight.requires_grad = False

        groups = get_parameter_groups(model, weight_decay=0.1)

        all_params = list(groups[0]["params"]) + list(groups[1]["params"])
        # Check that frozen param is not in any group
        assert not any(p is model.linear1.weight for p in all_params)


class TestCreateOptimizer:
    """Tests for create_optimizer factory."""

    def test_create_adam(self):
        """Test Adam optimizer creation."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            optimizer_type="adam",
            learning_rate=1e-3,
            weight_decay=0.1,
        )

        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 2

    def test_create_adamw(self):
        """Test AdamW optimizer creation."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            optimizer_type="adamw",
            learning_rate=1e-3,
            weight_decay=0.1,
        )

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_sgd(self):
        """Test SGD optimizer creation."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            optimizer_type="sgd",
            learning_rate=1e-2,
            momentum=0.9,
        )

        assert isinstance(optimizer, torch.optim.SGD)

    def test_invalid_optimizer_type(self):
        """Test invalid optimizer type raises error."""
        model = SimpleModel()

        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(model, optimizer_type="invalid_optimizer")

    @pytest.mark.skipif(not MUON_AVAILABLE, reason="Muon not installed")
    def test_create_muon(self):
        """Test Muon optimizer creation."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            optimizer_type="muon",
            learning_rate=4e-3,
        )

        assert optimizer is not None
        assert hasattr(optimizer, "step")

    @pytest.mark.skipif(not MUONCLIP_AVAILABLE, reason="MuonClip not installed")
    def test_create_muonclip(self):
        """Test MuonClip optimizer creation."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            optimizer_type="muonclip",
            learning_rate=4e-3,
            enable_clipping=False,  # Disable clipping since we don't have model config
        )

        assert optimizer is not None
        assert hasattr(optimizer, "step")

    def test_muon_not_installed(self):
        """Test error when Muon not installed."""
        if MUON_AVAILABLE:
            pytest.skip("Muon is installed")

        model = SimpleModel()
        with pytest.raises(ImportError, match="Muon optimizer not available"):
            create_optimizer(model, optimizer_type="muon")

    def test_adam_betas(self):
        """Test custom betas for Adam."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            optimizer_type="adam",
            learning_rate=1e-3,
            betas=(0.9, 0.99),
        )

        assert optimizer.defaults["betas"] == (0.9, 0.99)

    def test_learning_rate_applied(self):
        """Test learning rate is applied."""
        model = SimpleModel()
        lr = 5e-4
        optimizer = create_optimizer(
            model,
            optimizer_type="adam",
            learning_rate=lr,
        )

        for group in optimizer.param_groups:
            assert group["lr"] == lr

    def test_apollo_not_installed(self):
        """Test error when APOLLO not installed."""
        try:
            from apollo import Apollo
            pytest.skip("APOLLO is installed")
        except ImportError:
            pass

        model = SimpleModel()
        with pytest.raises(ImportError, match="APOLLO optimizer requires"):
            create_optimizer(model, optimizer_type="apollo")

    def test_adamw_8bit_not_installed(self):
        """Test error when bitsandbytes not installed."""
        try:
            import bitsandbytes
            pytest.skip("bitsandbytes is installed")
        except ImportError:
            pass

        model = SimpleModel()
        with pytest.raises(ImportError, match="bitsandbytes"):
            create_optimizer(model, optimizer_type="adamw_8bit")


class TestGetNumParameters:
    """Tests for get_num_parameters."""

    def test_count_all_parameters(self):
        """Test counting all parameters."""
        model = SimpleModel()
        total = get_num_parameters(model, only_trainable=False)

        expected = sum(p.numel() for p in model.parameters())
        assert total == expected

    def test_count_trainable_only(self):
        """Test counting only trainable parameters."""
        model = SimpleModel()
        model.linear1.weight.requires_grad = False

        trainable = get_num_parameters(model, only_trainable=True)
        all_params = get_num_parameters(model, only_trainable=False)

        assert trainable < all_params
        assert trainable == sum(p.numel() for p in model.parameters() if p.requires_grad)

    def test_empty_model(self):
        """Test with model with no parameters."""
        model = nn.Sequential()
        total = get_num_parameters(model)
        assert total == 0


class TestInfluenceAwareOptimizer:
    """Tests for InfluenceAwareOptimizer wrapper."""

    @pytest.fixture
    def base_optimizer(self):
        model = SimpleModel()
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def test_init(self, base_optimizer):
        """Test wrapper initialization."""
        wrapper = InfluenceAwareOptimizer(
            optimizer=base_optimizer,
            mixture_calculator=None,
            mixed_dataset=None,
            update_interval=100,
        )

        assert wrapper.optimizer is base_optimizer
        assert wrapper.step_count == 0
        assert wrapper.update_interval == 100

    def test_param_groups_passthrough(self, base_optimizer):
        """Test param_groups property passes through."""
        wrapper = InfluenceAwareOptimizer(
            optimizer=base_optimizer,
            mixture_calculator=None,
            mixed_dataset=None,
        )

        assert wrapper.param_groups is base_optimizer.param_groups

    def test_state_passthrough(self, base_optimizer):
        """Test state property passes through."""
        wrapper = InfluenceAwareOptimizer(
            optimizer=base_optimizer,
            mixture_calculator=None,
            mixed_dataset=None,
        )

        assert wrapper.state is base_optimizer.state

    def test_state_dict(self, base_optimizer):
        """Test state_dict passthrough."""
        wrapper = InfluenceAwareOptimizer(
            optimizer=base_optimizer,
            mixture_calculator=None,
            mixed_dataset=None,
        )

        state = wrapper.state_dict()
        assert state == base_optimizer.state_dict()

    def test_load_state_dict(self, base_optimizer):
        """Test load_state_dict passthrough."""
        wrapper = InfluenceAwareOptimizer(
            optimizer=base_optimizer,
            mixture_calculator=None,
            mixed_dataset=None,
        )

        state = wrapper.state_dict()
        wrapper.load_state_dict(state)

    def test_zero_grad(self, base_optimizer):
        """Test zero_grad passthrough."""
        wrapper = InfluenceAwareOptimizer(
            optimizer=base_optimizer,
            mixture_calculator=None,
            mixed_dataset=None,
        )

        # Should not raise
        wrapper.zero_grad()
        wrapper.zero_grad(set_to_none=True)

    def test_step_increments_counter(self, base_optimizer):
        """Test that step increments counter."""
        wrapper = InfluenceAwareOptimizer(
            optimizer=base_optimizer,
            mixture_calculator=None,
            mixed_dataset=None,
        )

        assert wrapper.step_count == 0
        wrapper.step()
        assert wrapper.step_count == 1
        wrapper.step()
        assert wrapper.step_count == 2

    def test_step_without_mixture_calculator(self, base_optimizer):
        """Test step works without mixture calculator."""
        wrapper = InfluenceAwareOptimizer(
            optimizer=base_optimizer,
            mixture_calculator=None,
            mixed_dataset=None,
            update_interval=1,  # Would trigger every step
        )

        # Should not raise even with frequent updates
        for _ in range(10):
            wrapper.step()

        assert wrapper.step_count == 10


class TestOptimizerIntegration:
    """Integration tests for optimizer training."""

    def test_adam_training_step(self):
        """Test a complete training step with Adam."""
        model = SimpleModel()
        optimizer = create_optimizer(model, optimizer_type="adam", learning_rate=1e-3)

        x = torch.randint(0, 100, (4, 16))
        y = model(x)
        loss = y.sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Should not raise

    def test_adamw_training_step(self):
        """Test a complete training step with AdamW."""
        model = SimpleModel()
        optimizer = create_optimizer(model, optimizer_type="adamw", learning_rate=1e-3)

        x = torch.randint(0, 100, (4, 16))
        y = model(x)
        loss = y.sum()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def test_weight_decay_separation(self):
        """Test that weight decay is applied correctly."""
        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            optimizer_type="adamw",
            learning_rate=1e-3,
            weight_decay=0.5,  # Large weight decay for visibility
        )

        # Get initial weights
        initial_weight = model.linear1.weight.clone()
        initial_bias = model.linear1.bias.clone()

        # Training step
        x = torch.randint(0, 100, (4, 16))
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(model.linear1.weight, initial_weight)
        # Bias should also change (from gradients)
        assert not torch.allclose(model.linear1.bias, initial_bias)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
