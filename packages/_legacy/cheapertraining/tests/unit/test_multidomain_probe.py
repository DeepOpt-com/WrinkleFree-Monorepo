"""Unit tests for multi-domain probe functionality.

Tests MobileLLM-R1 style multi-domain influence calculation.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from cheapertraining.data.mixing import (
    DomainProbeConfig,
    DomainProbeDataset,
    create_domain_probe_loaders,
    get_domain_weights,
)
from cheapertraining.influence.config import MixtureOptimizationConfig
from cheapertraining.influence.mixture_calculator import (
    MixtureWeightCalculator,
    DEFAULT_DOMAIN_WEIGHTS,
)


class TestDomainProbeConfig:
    """Tests for DomainProbeConfig dataclass."""

    def test_default_values(self):
        """Test default values for domain probe config."""
        config = DomainProbeConfig(
            domain="code",
            path="codeparrot/codeparrot-clean",
        )
        assert config.domain == "code"
        assert config.path == "codeparrot/codeparrot-clean"
        assert config.subset is None
        assert config.split == "train"
        assert config.samples == 2000
        assert config.text_column == "text"

    def test_custom_values(self):
        """Test custom values for domain probe config."""
        config = DomainProbeConfig(
            domain="math",
            path="LLM360/MegaMath",
            subset="subset-v1",
            split="validation",
            samples=1000,
            text_column="content",
        )
        assert config.domain == "math"
        assert config.samples == 1000
        assert config.text_column == "content"


class TestDomainProbeDataset:
    """Tests for DomainProbeDataset class."""

    def test_initialization(self):
        """Test dataset initialization."""
        config = DomainProbeConfig(
            domain="code",
            path="test/path",
            samples=100,
        )
        dataset = DomainProbeDataset(config, streaming=True, seed=42)

        assert dataset.config == config
        assert dataset.streaming is True
        assert dataset.seed == 42
        assert dataset._dataset is None

    def test_sample_limit(self):
        """Test that dataset respects sample limit."""
        config = DomainProbeConfig(
            domain="test",
            path="test/path",
            samples=5,
        )
        dataset = DomainProbeDataset(config, streaming=True)

        # Mock the dataset loading
        mock_data = [{"text": f"sample {i}"} for i in range(100)]

        with patch.object(dataset, "_load_dataset"):
            dataset._dataset = iter(mock_data)

            samples = list(dataset)
            assert len(samples) == 5
            assert all(s["domain"] == "test" for s in samples)


class TestGetDomainWeights:
    """Tests for get_domain_weights function."""

    def test_explicit_weights(self):
        """Test with explicitly specified weights."""
        config = {
            "domains": {"code": {}, "math": {}, "knowledge": {}},
            "domain_weights": {
                "code": 0.4,
                "math": 0.3,
                "knowledge": 0.3,
            },
        }
        weights = get_domain_weights(config)

        assert weights["code"] == 0.4
        assert weights["math"] == 0.3
        assert weights["knowledge"] == 0.3

    def test_default_equal_weights(self):
        """Test default equal weights when not specified."""
        config = {
            "domains": {"code": {}, "math": {}, "knowledge": {}},
        }
        weights = get_domain_weights(config)

        # Should be equal for 3 domains
        expected = 1.0 / 3
        assert abs(weights["code"] - expected) < 0.01
        assert abs(weights["math"] - expected) < 0.01
        assert abs(weights["knowledge"] - expected) < 0.01

    def test_partial_weights(self):
        """Test with some weights specified, others defaulted."""
        config = {
            "domains": {"code": {}, "math": {}},
            "domain_weights": {
                "code": 0.6,
            },
        }
        weights = get_domain_weights(config)

        assert weights["code"] == 0.6
        assert "math" in weights

    def test_empty_config(self):
        """Test with empty configuration."""
        weights = get_domain_weights({})
        assert weights == {}


class TestDefaultDomainWeights:
    """Tests for default domain weights constant."""

    def test_default_weights_sum_to_one(self):
        """Test that default domain weights sum to 1.0."""
        total = sum(DEFAULT_DOMAIN_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_default_weights_has_all_domains(self):
        """Test that default weights include all expected domains."""
        assert "code" in DEFAULT_DOMAIN_WEIGHTS
        assert "math" in DEFAULT_DOMAIN_WEIGHTS
        assert "knowledge" in DEFAULT_DOMAIN_WEIGHTS


class SimpleModel(nn.Module):
    """Simple model for testing influence calculations."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.head = nn.Linear(64, 100)

    def forward(self, x):
        return self.head(self.linear(x))


class TestMixtureWeightCalculatorMultiDomain:
    """Tests for MixtureWeightCalculator multi-domain mode."""

    def test_multi_domain_mode_detection(self):
        """Test that multi-domain mode is detected correctly."""
        model = SimpleModel()

        # Mock the influence calculator creation
        with patch(
            "cheapertraining.influence.mixture_calculator.create_influence_calculator"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            # Without domain probes - single domain mode
            calc_single = MixtureWeightCalculator(model)
            assert calc_single.multi_domain_mode is False

            # With domain probes - multi domain mode
            mock_loaders = {
                "code": MagicMock(),
                "math": MagicMock(),
            }
            calc_multi = MixtureWeightCalculator(
                model,
                domain_probe_loaders=mock_loaders,
            )
            assert calc_multi.multi_domain_mode is True

    def test_domain_weights_override(self):
        """Test custom domain weights are used."""
        model = SimpleModel()
        custom_weights = {
            "code": 0.5,
            "math": 0.3,
            "knowledge": 0.2,
        }

        with patch(
            "cheapertraining.influence.mixture_calculator.create_influence_calculator"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            calc = MixtureWeightCalculator(
                model,
                domain_weights=custom_weights,
            )

            assert calc.domain_weights == custom_weights

    def test_default_domain_weights(self):
        """Test default domain weights are applied."""
        model = SimpleModel()

        with patch(
            "cheapertraining.influence.mixture_calculator.create_influence_calculator"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            calc = MixtureWeightCalculator(model)

            assert calc.domain_weights == DEFAULT_DOMAIN_WEIGHTS

    def test_domain_probe_caching_tracking(self):
        """Test that domain probe caching is tracked correctly."""
        model = SimpleModel()
        mock_loaders = {
            "code": MagicMock(),
            "math": MagicMock(),
        }

        with patch(
            "cheapertraining.influence.mixture_calculator.create_influence_calculator"
        ) as mock_create:
            mock_create.return_value = MagicMock()

            calc = MixtureWeightCalculator(
                model,
                domain_probe_loaders=mock_loaders,
            )

            # Initially, no domains are cached
            assert calc._domain_probe_cached.get("code", False) is False
            assert calc._domain_probe_cached.get("math", False) is False


class TestMixtureOptimizationConfig:
    """Tests for mixture optimization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MixtureOptimizationConfig()

        assert config.samples_per_dataset > 0
        assert config.normalize_weights is True
        assert 0.0 < config.min_weight < config.max_weight
        assert config.max_weight <= 1.0

    def test_weight_constraints(self):
        """Test weight constraints are reasonable."""
        config = MixtureOptimizationConfig()

        # Min weight should allow at least 5% per dataset
        assert config.min_weight >= 0.0
        assert config.min_weight <= 0.1

        # Max weight should prevent single source dominance
        assert config.max_weight >= 0.5
        assert config.max_weight <= 1.0


class TestMultiDomainProbeYamlConfig:
    """Tests for multi-domain probe YAML configuration."""

    def test_mixed_pretrain_has_probe_domains(self):
        """Test that mixed_pretrain.yaml has multi-domain probe config."""
        from pathlib import Path
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "WrinkleFree-1.58Quant"
            / "configs"
            / "data"
            / "mixed_pretrain.yaml"
        )

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check probe configuration exists
        assert "probe" in config, "Probe configuration missing"
        probe = config["probe"]

        # Check domains
        assert "domains" in probe, "Domain configuration missing"
        domains = probe["domains"]

        # Check required domains
        assert "code" in domains, "Code domain missing"
        assert "math" in domains, "Math domain missing"
        assert "knowledge" in domains, "Knowledge domain missing"

    def test_probe_samples_from_training_sources(self):
        """Test that probe samples from our training sources."""
        from pathlib import Path
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "WrinkleFree-1.58Quant"
            / "configs"
            / "data"
            / "mixed_pretrain.yaml"
        )

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        sources = config.get("sources", [])
        source_paths = {s["path"] for s in sources}

        probe = config.get("probe", {})
        domains = probe.get("domains", {})

        # Check each domain probe is from training sources
        for domain, domain_cfg in domains.items():
            if isinstance(domain_cfg, dict):
                probe_path = domain_cfg.get("path", "")
                assert probe_path in source_paths, (
                    f"Domain {domain} probe path '{probe_path}' "
                    f"not in training sources: {source_paths}"
                )

    def test_dclm_in_training_mix(self):
        """Test that DCLM is included in training mix."""
        from pathlib import Path
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "WrinkleFree-1.58Quant"
            / "configs"
            / "data"
            / "mixed_pretrain.yaml"
        )

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        sources = config.get("sources", [])
        source_names = [s["name"] for s in sources]

        assert "dclm" in source_names, "DCLM should be in training mix"

        # Check DCLM config
        dclm_config = next(s for s in sources if s["name"] == "dclm")
        assert dclm_config["path"] == "mlfoundations/dclm-baseline-1.0"
        assert dclm_config["weight"] > 0.2, "DCLM should have significant weight"

    def test_weights_sum_to_one(self):
        """Test that training mix weights sum to approximately 1.0."""
        from pathlib import Path
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "WrinkleFree-1.58Quant"
            / "configs"
            / "data"
            / "mixed_pretrain.yaml"
        )

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        sources = config.get("sources", [])
        total_weight = sum(s["weight"] for s in sources)

        assert abs(total_weight - 1.0) < 0.01, (
            f"Weights should sum to 1.0, got {total_weight}"
        )


class TestInfluenceMixtureYamlConfig:
    """Tests for influence mixture YAML configuration."""

    def test_multi_domain_enabled(self):
        """Test that multi-domain mode is enabled in config."""
        from pathlib import Path
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "influence"
            / "mixture.yaml"
        )

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        probe = config.get("probe", {})
        assert probe.get("multi_domain") is True, "Multi-domain mode should be enabled"

    def test_domain_weights_in_config(self):
        """Test that domain weights are configured."""
        from pathlib import Path
        import yaml

        config_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "influence"
            / "mixture.yaml"
        )

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        probe = config.get("probe", {})
        domain_weights = probe.get("domain_weights", {})

        assert "code" in domain_weights
        assert "math" in domain_weights
        assert "knowledge" in domain_weights

        # Weights should sum to 1.0
        total = sum(domain_weights.values())
        assert abs(total - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
