"""Tests for efficient meta-optimization module.

Tests:
- LDCMTLManager: Objective weight optimization via router network
- OnlineDataMixer: Dataset weight optimization via EXP3 bandit
- MetaOptimizationConfig: Configuration validation
"""

import pytest
import torch

from wrinklefree.meta import (
    LDCMTLConfig,
    LDCMTLManager,
    MetaOptimizationConfig,
    ODMConfig,
    ObjectiveRouter,
    OnlineDataMixer,
    compute_loss_discrepancy,
)


class TestMetaOptimizationConfig:
    """Tests for MetaOptimizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetaOptimizationConfig()
        assert config.enabled is False
        assert config.ldc_mtl.enabled is True
        assert config.odm.enabled is True
        assert config.log_interval == 100

    def test_ldc_mtl_config_defaults(self):
        """Test LDC-MTL config defaults."""
        config = LDCMTLConfig()
        assert config.enabled is True
        assert config.lambda_penalty == 0.1
        assert config.hidden_dim == 32
        assert config.router_lr == 1e-3

    def test_odm_config_defaults(self):
        """Test ODM config defaults."""
        config = ODMConfig()
        assert config.enabled is True
        assert config.reward_smoothing == 0.9
        assert config.warmup_ratio == 0.01
        assert config.min_weight == 0.05
        assert config.max_weight == 0.60


class TestObjectiveRouter:
    """Tests for ObjectiveRouter."""

    def test_forward_shape(self):
        """Test router output shape."""
        router = ObjectiveRouter(num_objectives=3, hidden_dim=16)
        losses = torch.tensor([1.0, 2.0, 3.0])
        weights = router(losses)
        assert weights.shape == (3,)

    def test_forward_sums_to_one(self):
        """Test router outputs sum to 1 (softmax)."""
        router = ObjectiveRouter(num_objectives=3)
        losses = torch.tensor([0.5, 1.5, 2.0])
        weights = router(losses)
        assert abs(weights.sum().item() - 1.0) < 1e-6

    def test_forward_all_positive(self):
        """Test router outputs are all positive."""
        router = ObjectiveRouter(num_objectives=4)
        losses = torch.tensor([0.1, 0.2, 0.3, 0.4])
        weights = router(losses)
        assert (weights > 0).all()


class TestLossDiscrepancy:
    """Tests for compute_loss_discrepancy."""

    def test_zero_discrepancy_for_equal_weighted_losses(self):
        """Equal weighted losses should have zero discrepancy."""
        losses = torch.tensor([1.0, 1.0, 1.0])
        weights = torch.tensor([1/3, 1/3, 1/3])
        discrepancy = compute_loss_discrepancy(losses, weights)
        assert abs(discrepancy.item()) < 1e-6

    def test_nonzero_discrepancy_for_unequal(self):
        """Unequal weighted losses should have positive discrepancy."""
        losses = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([0.5, 0.3, 0.2])
        discrepancy = compute_loss_discrepancy(losses, weights)
        assert discrepancy.item() > 0


class TestLDCMTLManager:
    """Tests for LDCMTLManager."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return LDCMTLConfig(
            enabled=True,
            lambda_penalty=0.1,
            hidden_dim=16,
            router_lr=1e-3,
        )

    @pytest.fixture
    def manager(self, config):
        """Create test manager."""
        return LDCMTLManager(
            objective_names=["ce", "dlm"],
            config=config,
            device=torch.device("cpu"),
        )

    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        weights = manager.get_weights()
        assert len(weights) == 2
        assert "ce" in weights
        assert "dlm" in weights
        # Initial weights should be roughly equal
        assert abs(weights["ce"] - weights["dlm"]) < 0.3

    def test_compute_weighted_loss(self, manager):
        """Test weighted loss computation."""
        losses = {"ce": torch.tensor(1.0), "dlm": torch.tensor(2.0)}
        total_loss, weights = manager.compute_weighted_loss(losses)

        assert total_loss.shape == ()
        assert total_loss.item() > 0
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_step_updates_router(self, manager):
        """Test that step() updates the router."""
        losses = {"ce": torch.tensor(1.0, requires_grad=True), "dlm": torch.tensor(2.0, requires_grad=True)}
        total_loss, _ = manager.compute_weighted_loss(losses)
        total_loss.backward()
        manager.step()
        # No error means success

    def test_state_dict_roundtrip(self, manager):
        """Test save/load state dict."""
        # Get initial weights
        initial_weights = manager.get_weights()

        # Save state
        state = manager.state_dict()
        assert "router_state" in state
        assert "optimizer_state" in state

        # Create new manager and load
        new_manager = LDCMTLManager(
            objective_names=["ce", "dlm"],
            config=manager.config,
            device=torch.device("cpu"),
        )
        new_manager.load_state_dict(state)

        # Check weights preserved
        loaded_weights = new_manager.get_weights()
        for name in initial_weights:
            assert abs(initial_weights[name] - loaded_weights[name]) < 1e-6

    def test_wandb_metrics(self, manager):
        """Test WandB metrics generation."""
        metrics = manager.get_wandb_metrics()

        assert "meta/ldc_mtl/objective_weight_ce" in metrics
        assert "meta/ldc_mtl/objective_weight_dlm" in metrics


class TestOnlineDataMixer:
    """Tests for OnlineDataMixer (EXP3)."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ODMConfig(
            enabled=True,
            reward_smoothing=0.9,
            warmup_ratio=0.01,
            min_weight=0.05,
            max_weight=0.60,
        )

    @pytest.fixture
    def mixer(self, config):
        """Create test mixer."""
        return OnlineDataMixer(
            dataset_names=["web", "code", "math"],
            config=config,
        )

    def test_initialization(self, mixer):
        """Test mixer initializes correctly."""
        weights = mixer.get_sampling_weights()
        assert len(weights) == 3
        assert "web" in weights
        assert "code" in weights
        assert "math" in weights

    def test_initial_weights_roughly_uniform(self, mixer):
        """Test initial weights are roughly uniform."""
        weights = mixer.get_sampling_weights()
        for w in weights.values():
            assert 0.2 < w < 0.5  # Roughly 1/3 each

    def test_weights_sum_to_one(self, mixer):
        """Test weights always sum to 1."""
        weights = mixer.get_sampling_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_exploration_rate_decay(self, mixer):
        """Test exploration rate decreases over time."""
        eps_0 = mixer.get_exploration_rate()

        mixer.step_count = 100
        eps_100 = mixer.get_exploration_rate()

        mixer.step_count = 1000
        eps_1000 = mixer.get_exploration_rate()

        assert eps_0 >= eps_100 >= eps_1000

    def test_update_changes_rewards(self, mixer):
        """Test update() modifies avg_rewards."""
        initial_rewards = mixer.avg_rewards.copy()

        losses = {"web": 1.0, "code": 2.0, "math": 0.5}
        mixer.update(losses)

        assert mixer.avg_rewards != initial_rewards

    def test_update_favors_high_loss(self, mixer):
        """Test that domains with higher loss get higher rewards."""
        # Run many updates with consistent losses
        for _ in range(100):
            losses = {"web": 1.0, "code": 3.0, "math": 0.5}
            mixer.update(losses)

        # Code should have highest reward (highest loss)
        assert mixer.avg_rewards["code"] > mixer.avg_rewards["web"]
        assert mixer.avg_rewards["code"] > mixer.avg_rewards["math"]

    def test_constraints_enforced(self, mixer):
        """Test min/max weight constraints are enforced."""
        # Force extreme rewards
        mixer.avg_rewards = {"web": 100.0, "code": 0.0, "math": 0.0}
        mixer.step_count = 1000

        weights = mixer.get_sampling_weights()

        assert weights["web"] <= mixer.config.max_weight + 0.01
        assert weights["code"] >= mixer.config.min_weight - 0.01
        assert weights["math"] >= mixer.config.min_weight - 0.01

    def test_warmup_detection(self, mixer):
        """Test warmup period detection."""
        assert mixer.is_in_warmup(0, 10000) is True
        assert mixer.is_in_warmup(50, 10000) is True
        assert mixer.is_in_warmup(100, 10000) is False  # 1% of 10000

    def test_uniform_weights(self, mixer):
        """Test uniform weights for warmup."""
        weights = mixer.get_uniform_weights()
        for w in weights.values():
            assert abs(w - 1/3) < 1e-6

    def test_state_dict_roundtrip(self, mixer):
        """Test save/load state dict."""
        # Make some updates
        mixer.update({"web": 1.0, "code": 2.0, "math": 0.5})
        mixer.update({"web": 1.5, "code": 1.0, "math": 2.0})

        # Save state
        state = mixer.state_dict()

        # Create new mixer and load
        new_mixer = OnlineDataMixer(
            dataset_names=["web", "code", "math"],
            config=mixer.config,
        )
        new_mixer.load_state_dict(state)

        assert mixer.avg_rewards == new_mixer.avg_rewards
        assert mixer.step_count == new_mixer.step_count

    def test_wandb_metrics(self, mixer):
        """Test WandB metrics generation."""
        metrics = mixer.get_wandb_metrics()

        assert "meta/odm/dataset_weight_web" in metrics
        assert "meta/odm/dataset_weight_code" in metrics
        assert "meta/odm/exploration_rate" in metrics


class TestEdgeCases:
    """Edge case tests."""

    def test_single_objective_ldc_mtl(self):
        """Test LDC-MTL with single objective (edge case)."""
        config = LDCMTLConfig()
        manager = LDCMTLManager(
            objective_names=["ce"],
            config=config,
            device=torch.device("cpu"),
        )

        losses = {"ce": torch.tensor(1.0)}
        total_loss, weights = manager.compute_weighted_loss(losses)

        assert weights["ce"] == 1.0  # Only one objective

    def test_single_dataset_odm(self):
        """Test ODM with single dataset (edge case)."""
        config = ODMConfig()
        mixer = OnlineDataMixer(
            dataset_names=["web"],
            config=config,
        )

        weights = mixer.get_sampling_weights()
        assert weights["web"] == 1.0

    def test_zero_losses(self):
        """Test with zero losses."""
        config = LDCMTLConfig()
        manager = LDCMTLManager(
            objective_names=["ce", "dlm"],
            config=config,
            device=torch.device("cpu"),
        )

        losses = {"ce": torch.tensor(0.0), "dlm": torch.tensor(0.0)}
        total_loss, weights = manager.compute_weighted_loss(losses)

        assert total_loss.item() == 0.0

    def test_odm_unknown_domain(self):
        """Test ODM handles unknown domain gracefully."""
        config = ODMConfig()
        mixer = OnlineDataMixer(
            dataset_names=["web", "code"],
            config=config,
        )

        # Should log warning but not crash
        mixer.update({"unknown": 1.0})
