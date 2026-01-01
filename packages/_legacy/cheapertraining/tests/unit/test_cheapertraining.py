"""Tests for cheapertraining source modules.

These tests import and validate the cheapertraining package to ensure coverage.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch


class TestDataMixing:
    """Tests for cheapertraining.data.mixing module."""

    def test_import_mixing(self):
        """Test that mixing module can be imported."""
        from cheapertraining.data.mixing import DatasetMixture, MixedDataset

        assert DatasetMixture is not None
        assert MixedDataset is not None

    def test_dataset_mixture_creation(self):
        """Test DatasetMixture dataclass creation."""
        from cheapertraining.data.mixing import DatasetMixture

        mixture = DatasetMixture(
            name="fineweb",
            weight=0.3,
            path="HuggingFaceFW/fineweb-edu",
        )

        assert mixture.name == "fineweb"
        assert mixture.weight == 0.3
        assert mixture.path == "HuggingFaceFW/fineweb-edu"
        assert mixture.split == "train"  # default
        assert mixture.text_column == "text"  # default

    def test_dataset_mixture_with_subset(self):
        """Test DatasetMixture with subset."""
        from cheapertraining.data.mixing import DatasetMixture

        mixture = DatasetMixture(
            name="fineweb-10bt",
            weight=0.5,
            path="HuggingFaceFW/fineweb-edu",
            subset="sample-10BT",
        )

        assert mixture.subset == "sample-10BT"

    def test_mixed_dataset_init(self):
        """Test MixedDataset initialization."""
        from cheapertraining.data.mixing import DatasetMixture, MixedDataset

        mixtures = [
            DatasetMixture(name="a", weight=0.3, path="path/a"),
            DatasetMixture(name="b", weight=0.7, path="path/b"),
        ]

        dataset = MixedDataset(mixtures, seed=42)

        assert len(dataset.mixtures) == 2
        assert dataset.seed == 42
        assert dataset.streaming is True  # default
        assert dataset.rank == 0
        assert dataset.world_size == 1

    def test_mixed_dataset_weight_normalization(self):
        """Test that MixedDataset normalizes weights."""
        from cheapertraining.data.mixing import DatasetMixture, MixedDataset

        mixtures = [
            DatasetMixture(name="a", weight=1.0, path="path/a"),
            DatasetMixture(name="b", weight=3.0, path="path/b"),
        ]

        dataset = MixedDataset(mixtures)

        # Weights should normalize to 1
        total = dataset.normalized_weights.sum().item()
        assert abs(total - 1.0) < 1e-6

        # Check individual weights
        assert abs(dataset.normalized_weights[0].item() - 0.25) < 1e-6  # 1/(1+3)
        assert abs(dataset.normalized_weights[1].item() - 0.75) < 1e-6  # 3/(1+3)

    def test_mixed_dataset_update_weights(self):
        """Test update_weights_from_influence method."""
        from cheapertraining.data.mixing import DatasetMixture, MixedDataset

        mixtures = [
            DatasetMixture(name="a", weight=0.5, path="path/a"),
            DatasetMixture(name="b", weight=0.5, path="path/b"),
        ]

        dataset = MixedDataset(mixtures)

        # Update weights
        new_weights = {"a": 0.2, "b": 0.8}
        dataset.update_weights_from_influence(new_weights)

        # Check updated weights
        assert abs(dataset.normalized_weights[0].item() - 0.2) < 1e-6
        assert abs(dataset.normalized_weights[1].item() - 0.8) < 1e-6

    def test_mixed_dataset_get_current_weights(self):
        """Test get_current_weights method."""
        from cheapertraining.data.mixing import DatasetMixture, MixedDataset

        mixtures = [
            DatasetMixture(name="code", weight=0.3, path="path/code"),
            DatasetMixture(name="math", weight=0.7, path="path/math"),
        ]

        dataset = MixedDataset(mixtures)
        weights = dataset.get_current_weights()

        assert "code" in weights
        assert "math" in weights
        assert abs(weights["code"] - 0.3) < 1e-6
        assert abs(weights["math"] - 0.7) < 1e-6


class TestInfluenceConfig:
    """Tests for cheapertraining.influence.config module."""

    def test_import_influence_config(self):
        """Test that influence config module can be imported."""
        from cheapertraining.influence.config import (
            InfluenceConfig,
            InfluenceTarget,
            ProbeSetConfig,
            MixtureOptimizationConfig,
            SelfBoostingConfig,
        )

        assert InfluenceConfig is not None
        assert InfluenceTarget is not None
        assert ProbeSetConfig is not None
        assert MixtureOptimizationConfig is not None
        assert SelfBoostingConfig is not None

    def test_influence_target_enum(self):
        """Test InfluenceTarget enum values."""
        from cheapertraining.influence.config import InfluenceTarget

        assert InfluenceTarget.EMBEDDING_ONLY.value == "embedding"
        assert InfluenceTarget.OUTPUT_ONLY.value == "output"
        assert InfluenceTarget.EMBEDDING_AND_OUTPUT.value == "both"

    def test_influence_config_defaults(self):
        """Test InfluenceConfig default values."""
        from cheapertraining.influence.config import (
            InfluenceConfig,
            InfluenceTarget,
        )

        config = InfluenceConfig()

        assert config.target_layers == InfluenceTarget.EMBEDDING_AND_OUTPUT
        assert config.lambda_reg == 1e-4
        assert config.batch_size == 32
        assert config.use_fp16 is True
        assert config.cache_gradients is True
        assert config.max_grad_norm == 1.0

    def test_probe_set_config_defaults(self):
        """Test ProbeSetConfig default values."""
        from cheapertraining.influence.config import ProbeSetConfig

        config = ProbeSetConfig()

        assert config.probe_set_size == 10000
        assert config.seed == 42
        assert config.fineweb_edu_min_score == 4.0
        assert config.ask_llm_top_fraction == 0.10
        assert config.dedup_method == "minhash"
        assert len(config.domains) == 3

    def test_mixture_optimization_config_defaults(self):
        """Test MixtureOptimizationConfig defaults."""
        from cheapertraining.influence.config import MixtureOptimizationConfig

        config = MixtureOptimizationConfig()

        assert config.samples_per_dataset == 1000
        assert config.normalize_weights is True
        assert config.min_weight == 0.01
        assert config.max_weight == 0.90
        assert config.weight_update_interval == 10000

    def test_self_boosting_config_defaults(self):
        """Test SelfBoostingConfig defaults."""
        from cheapertraining.influence.config import SelfBoostingConfig

        config = SelfBoostingConfig()

        assert config.influence_threshold == 0.0
        assert config.num_stages == 2
        assert config.compression_ratio_per_stage == 0.5
        assert config.recompute_interval == 1000
        assert config.min_batch_size == 1


class TestInfluenceGradient:
    """Tests for cheapertraining.influence.gradient module."""

    def test_import_gradient(self):
        """Test that gradient module can be imported."""
        from cheapertraining.influence import gradient

        assert gradient is not None


class TestInfluenceDataInf:
    """Tests for cheapertraining.influence.datainf module."""

    def test_import_datainf(self):
        """Test that datainf module can be imported."""
        from cheapertraining.influence import datainf

        assert datainf is not None


class TestInfluenceMixtureCalculator:
    """Tests for cheapertraining.influence.mixture_calculator module."""

    def test_import_mixture_calculator(self):
        """Test that mixture_calculator module can be imported."""
        from cheapertraining.influence import mixture_calculator

        assert mixture_calculator is not None


class TestInit:
    """Tests for cheapertraining package init."""

    def test_package_import(self):
        """Test that cheapertraining package can be imported."""
        import cheapertraining

        assert cheapertraining is not None


class TestPrecision:
    """Tests for cheapertraining.precision module."""

    def test_import_precision(self):
        """Test that precision module can be imported."""
        from cheapertraining import precision

        assert precision is not None


class TestTraining:
    """Tests for cheapertraining.training module."""

    def test_import_training(self):
        """Test that training module can be imported."""
        from cheapertraining import training

        assert training is not None


class TestUtils:
    """Tests for cheapertraining.utils module."""

    def test_import_utils(self):
        """Test that utils module can be imported."""
        from cheapertraining import utils

        assert utils is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
