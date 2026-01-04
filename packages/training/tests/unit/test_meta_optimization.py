"""Tests for meta-optimization module.

Tests:
- MetaParameterManager: weight management and updates
- ParetoGradientSolver: MGDA and EPO algorithms
- MetaOptimizationConfig: configuration validation
"""

import pytest
import torch

from wrinklefree.meta import (
    MetaOptimizationConfig,
    MetaParameterManager,
    ParetoConfig,
    ParetoGradientSolver,
)


class TestMetaOptimizationConfig:
    """Tests for MetaOptimizationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetaOptimizationConfig()
        assert config.enabled is False
        assert config.optimize_dataset_weights is True
        assert config.optimize_objective_weights is True
        assert config.optimize_learning_rates is False
        assert config.update_interval == 1000
        assert config.warmup_steps == 500

    def test_enabled_config_gets_default_validation_objectives(self):
        """Test that enabled config gets default validation objective."""
        config = MetaOptimizationConfig(enabled=True)
        assert len(config.validation_objectives) == 1
        assert config.validation_objectives[0].name == "validation_perplexity"

    def test_pareto_config_defaults(self):
        """Test Pareto config defaults."""
        config = ParetoConfig()
        assert config.method == "mgda"
        assert config.max_iter == 10
        assert config.normalize_gradients is True


class TestMetaParameterManager:
    """Tests for MetaParameterManager."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return MetaOptimizationConfig(
            enabled=True,
            optimize_dataset_weights=True,
            optimize_objective_weights=True,
            optimize_learning_rates=True,
            meta_lr=0.1,
            meta_momentum=0.9,
        )

    @pytest.fixture
    def manager(self, config):
        """Create a test manager."""
        return MetaParameterManager(
            config=config,
            dataset_names=["web", "code", "math"],
            objective_names=["ce", "dlm"],
            optimizer_param_groups=["muon", "adamw"],
        )

    def test_initialization(self, manager):
        """Test that manager initializes correctly."""
        # Dataset weights should be uniform (softmax of zeros)
        weights = manager.get_dataset_weights()
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert abs(weights["web"] - 1/3) < 1e-6

        # Objective weights should be 1.0
        obj_weights = manager.get_objective_weights()
        assert len(obj_weights) == 2
        assert obj_weights["ce"] == 1.0
        assert obj_weights["dlm"] == 1.0

        # LR scales should be 1.0
        lr_scales = manager.get_lr_scales()
        assert len(lr_scales) == 2
        assert lr_scales["muon"] == 1.0

    def test_update_dataset_weights(self, manager):
        """Test gradient update for dataset weights."""
        # Gradient suggesting to increase 'code' weight
        grads = {"web": 0.1, "code": -0.2, "math": 0.1}
        manager.update_from_gradients(dataset_grads=grads)

        weights = manager.get_dataset_weights()
        # 'code' should now have higher weight (negative gradient = increase)
        assert weights["code"] > weights["web"]
        assert weights["code"] > weights["math"]

    def test_update_objective_weights(self, manager):
        """Test gradient update for objective weights."""
        grads = {"ce": -0.1, "dlm": 0.1}
        manager.update_from_gradients(objective_grads=grads)

        obj_weights = manager.get_objective_weights()
        # 'ce' should increase (negative gradient)
        assert obj_weights["ce"] > obj_weights["dlm"]

    def test_objective_weight_constraints(self, manager):
        """Test that objective weights respect constraints."""
        # Apply large negative gradient (should try to push below min)
        grads = {"ce": 10.0, "dlm": 10.0}
        for _ in range(100):  # Many updates
            manager.update_from_gradients(objective_grads=grads)

        obj_weights = manager.get_objective_weights()
        min_w, max_w = manager.config.constraints.objective_weight_range
        assert obj_weights["ce"] >= min_w
        assert obj_weights["dlm"] >= min_w

    def test_lr_scale_constraints(self, manager):
        """Test that LR scales respect constraints."""
        grads = {"muon": 10.0, "adamw": 10.0}
        for _ in range(100):
            manager.update_from_gradients(lr_grads=grads)

        lr_scales = manager.get_lr_scales()
        min_s, max_s = manager.config.constraints.lr_scale_range
        assert lr_scales["muon"] >= min_s
        assert lr_scales["adamw"] >= min_s

    def test_state_dict_roundtrip(self, manager):
        """Test save/load state dict."""
        # Make some updates
        manager.update_from_gradients(
            dataset_grads={"web": -0.1, "code": 0.1, "math": 0.0},
            objective_grads={"ce": 0.1, "dlm": -0.1},
        )

        # Save state
        state = manager.state_dict()

        # Create new manager and load
        new_manager = MetaParameterManager(
            config=manager.config,
            dataset_names=["web", "code", "math"],
            objective_names=["ce", "dlm"],
            optimizer_param_groups=["muon", "adamw"],
        )
        new_manager.load_state_dict(state)

        # Check weights match
        assert manager.get_dataset_weights() == new_manager.get_dataset_weights()
        assert manager.get_objective_weights() == new_manager.get_objective_weights()

    def test_wandb_metrics(self, manager):
        """Test WandB metrics generation."""
        metrics = manager.get_wandb_metrics(prefix="meta")

        assert "meta/dataset_weight_web" in metrics
        assert "meta/dataset_weight_code" in metrics
        assert "meta/objective_weight_ce" in metrics
        assert "meta/lr_scale_muon" in metrics


class TestParetoGradientSolver:
    """Tests for ParetoGradientSolver."""

    @pytest.fixture
    def solver(self):
        """Create solver with default config."""
        return ParetoGradientSolver(ParetoConfig())

    def test_single_objective(self, solver):
        """Test with single objective (should return gradient as-is)."""
        g = torch.tensor([1.0, 2.0, 3.0])
        result = solver.solve([g])
        assert torch.allclose(result, g)

    def test_mgda_aligned_gradients(self, solver):
        """Test MGDA with aligned gradients."""
        # Two gradients pointing in same direction
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([2.0, 0.0])

        result = solver.solve_mgda([g1, g2])

        # Result should point in same direction
        assert result[0] > 0
        assert abs(result[1]) < 1e-6

    def test_mgda_orthogonal_gradients(self, solver):
        """Test MGDA with orthogonal gradients."""
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([0.0, 1.0])

        result = solver.solve_mgda([g1, g2])

        # Result should be in first quadrant (positive in both dims)
        # because that's the only direction that improves both
        assert result[0] > 0
        assert result[1] > 0

    def test_mgda_conflicting_gradients(self, solver):
        """Test MGDA with conflicting gradients."""
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([-1.0, 0.0])

        result = solver.solve_mgda([g1, g2])

        # When gradients conflict, MGDA should find compromise
        # Result should have smaller norm than inputs
        assert result.norm() < max(g1.norm(), g2.norm())

    def test_mgda_three_objectives(self, solver):
        """Test MGDA with three objectives."""
        g1 = torch.tensor([1.0, 0.0, 0.0])
        g2 = torch.tensor([0.0, 1.0, 0.0])
        g3 = torch.tensor([0.0, 0.0, 1.0])

        result = solver.solve_mgda([g1, g2, g3])

        # All components should be positive
        assert (result > 0).all()

    def test_pareto_weights_sum_to_one(self, solver):
        """Test that Pareto weights form valid simplex."""
        g1 = torch.tensor([1.0, 2.0])
        g2 = torch.tensor([2.0, 1.0])

        weights = solver.compute_pareto_weights([g1, g2])

        assert abs(weights.sum() - 1.0) < 1e-6
        assert (weights >= 0).all()

    def test_epo_with_preferences(self, solver):
        """Test EPO with preference weights."""
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([0.0, 1.0])

        # Prefer first objective more
        result = solver.solve_epo([g1, g2], preferences=[0.8, 0.2])

        # Result should be biased toward first gradient
        assert result[0] > result[1]

    def test_linear_scalarization(self, solver):
        """Test linear scalarization method."""
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([0.0, 1.0])

        result = solver.solve_linear([g1, g2], weights=[0.7, 0.3])

        # Should be weighted sum
        expected = 0.7 * g1 + 0.3 * g2
        assert torch.allclose(result, expected)

    def test_normalization(self):
        """Test gradient normalization option."""
        solver = ParetoGradientSolver(ParetoConfig(normalize_gradients=True))

        # Gradients with different magnitudes
        g1 = torch.tensor([10.0, 0.0])
        g2 = torch.tensor([0.0, 1.0])

        result = solver.solve_mgda([g1, g2])

        # With normalization, magnitudes shouldn't dominate
        # Both directions should be roughly equal
        assert abs(result[0] - result[1]) < 0.5


class TestMetaGradientIntegration:
    """Integration tests for meta-gradient computation.

    These tests verify the full pipeline works together.
    """

    def test_pareto_with_meta_manager(self):
        """Test Pareto solver integrates with meta manager."""
        config = MetaOptimizationConfig(
            enabled=True,
            meta_lr=0.1,
        )
        manager = MetaParameterManager(
            config=config,
            dataset_names=["a", "b"],
            objective_names=["ce"],
        )
        solver = ParetoGradientSolver()

        # Simulate two validation objectives with different preferences
        # Val 1: prefers dataset a
        meta_grads_1 = {"a": -0.5, "b": 0.5}
        # Val 2: prefers dataset b
        meta_grads_2 = {"a": 0.5, "b": -0.5}

        # Convert to tensors for Pareto solver
        grad_1 = torch.tensor([meta_grads_1["a"], meta_grads_1["b"]])
        grad_2 = torch.tensor([meta_grads_2["a"], meta_grads_2["b"]])

        # Solve Pareto
        combined = solver.solve([grad_1, grad_2])

        # Apply to manager
        combined_dict = {"a": combined[0].item(), "b": combined[1].item()}
        manager.update_from_gradients(dataset_grads=combined_dict)

        # Weights should be balanced (conflicting preferences average out)
        weights = manager.get_dataset_weights()
        assert abs(weights["a"] - weights["b"]) < 0.3  # Roughly equal


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_dataset_names(self):
        """Test manager with no datasets."""
        config = MetaOptimizationConfig(enabled=True)
        manager = MetaParameterManager(
            config=config,
            dataset_names=[],
            objective_names=["ce"],
        )
        assert manager.get_dataset_weights() == {}

    def test_zero_gradients(self):
        """Test update with zero gradients."""
        config = MetaOptimizationConfig(enabled=True, meta_lr=0.1)
        manager = MetaParameterManager(
            config=config,
            dataset_names=["a", "b"],
            objective_names=["ce"],
        )

        initial_weights = manager.get_dataset_weights().copy()
        manager.update_from_gradients(dataset_grads={"a": 0.0, "b": 0.0})
        final_weights = manager.get_dataset_weights()

        # Weights should not change with zero gradients
        assert initial_weights == final_weights

    def test_pareto_empty_list(self):
        """Test Pareto solver with empty gradient list."""
        solver = ParetoGradientSolver()
        with pytest.raises(ValueError):
            solver.solve([])
