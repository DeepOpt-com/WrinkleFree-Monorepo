"""Tests for optimizer creation and MuonClip integration."""

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing optimizers."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.linear = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 100)

    def forward(self, x):
        x = self.embed(x)
        x = self.linear(x)
        x = self.norm(x)
        return self.head(x)


class TestCreateOptimizer:
    """Tests for create_optimizer function."""

    def test_create_muonclip_optimizer(self):
        """Test MuonClip optimizer creation (default)."""
        pytest.importorskip("muon", reason="muon-clip not installed")
        from wf_train.training._legacy.trainer import create_optimizer

        model = SimpleModel()
        # SimpleModel doesn't have attention heads, so clipping will be auto-disabled
        optimizer = create_optimizer(
            model,
            learning_rate=4e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            optimizer_type="muonclip",
        )

        assert optimizer is not None
        assert hasattr(optimizer, "step")
        assert hasattr(optimizer, "zero_grad")

    def test_muonclip_with_custom_clipping(self):
        """Test MuonClip with custom clipping parameters (disabled for simple model)."""
        pytest.importorskip("muon", reason="muon-clip not installed")
        from wf_train.training._legacy.trainer import create_optimizer

        model = SimpleModel()
        # Clipping requires model_config with attention heads, so it will be disabled
        optimizer = create_optimizer(
            model,
            learning_rate=4e-3,
            optimizer_type="muonclip",
            enable_clipping=False,  # Explicitly disable for simple model
            clipping_threshold=100.0,
            clipping_alpha=0.3,
        )

        assert optimizer is not None

    def test_muonclip_clipping_disabled(self):
        """Test MuonClip with QK-clipping disabled."""
        pytest.importorskip("muon", reason="muon-clip not installed")
        from wf_train.training._legacy.trainer import create_optimizer

        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            learning_rate=4e-3,
            optimizer_type="muonclip",
            enable_clipping=False,
        )

        assert optimizer is not None

    @pytest.mark.gpu
    def test_muonclip_training_step(self):
        """Test MuonClip can perform forward/backward/step."""
        if not torch.cuda.is_available():
            pytest.skip("MuonClip test requires GPU (falls back to 8-bit optimizer on CPU)")

        pytest.importorskip("muon", reason="muon-clip not installed")
        from wf_train.training._legacy.trainer import create_optimizer

        model = SimpleModel().cuda()
        optimizer = create_optimizer(
            model,
            learning_rate=4e-3,
            optimizer_type="muonclip",
            enable_clipping=False,  # Disable for simple model
        )

        # Forward pass
        x = torch.randint(0, 100, (2, 16)).cuda()
        logits = model(x)
        loss = logits.sum()

        # Backward pass
        loss.backward()

        # Optimizer step should not raise
        optimizer.step()
        optimizer.zero_grad()

    def test_adamw_8bit_fallback(self):
        """Test AdamW 8-bit optimizer creation."""
        from wf_train.training._legacy.trainer import create_optimizer

        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            weight_decay=0.1,
            optimizer_type="adamw_8bit",
        )

        # Should return some optimizer (8-bit or fallback)
        assert optimizer is not None
        assert hasattr(optimizer, "step")

    def test_adamw_optimizer(self):
        """Test standard AdamW optimizer creation."""
        from wf_train.training._legacy.trainer import create_optimizer

        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            weight_decay=0.1,
            optimizer_type="adamw",
        )

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_parameter_groups_weight_decay(self):
        """Test that bias/norm params have no weight decay."""
        from wf_train.training._legacy.trainer import create_optimizer

        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            learning_rate=1e-4,
            weight_decay=0.1,
            optimizer_type="adamw",
        )

        # Should have 2 param groups
        assert len(optimizer.param_groups) == 2
        # First group: decay params
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        # Second group: no decay params (bias, norm)
        assert optimizer.param_groups[1]["weight_decay"] == 0.0


class TestDatasetConfig:
    """Tests for dataset configuration."""

    def test_finemath_in_config(self):
        """Test that FineMath is configured in mixed_pretrain.yaml."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check sources contain finemath
        sources = config.get("sources", [])
        source_names = [s["name"] for s in sources]

        assert "finemath" in source_names, "FineMath should be in dataset sources"

        # Check finemath config
        finemath_config = next(s for s in sources if s["name"] == "finemath")
        assert finemath_config["path"] == "HuggingFaceTB/finemath"
        assert finemath_config["weight"] == 0.15

    def test_github_code_in_config(self):
        """Test that GitHub Code 2025 is configured in mixed_pretrain.yaml."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        sources = config.get("sources", [])
        source_names = [s["name"] for s in sources]

        assert "github_code" in source_names, "GitHub Code should be in dataset sources"


class TestMultiDomainProbe:
    """Tests for multi-domain probe configuration (MobileLLM-R1 style)."""

    def test_probe_has_all_domains(self):
        """Test that probe config has code, math, web_edu, dclm, and diverse domains."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        probe = config.get("probe", {})
        domains = probe.get("domains", {})

        assert "code" in domains, "Code domain missing from probe"
        assert "math" in domains, "Math domain missing from probe"
        assert "web_edu" in domains, "Web edu domain missing from probe"
        assert "dclm" in domains, "DCLM domain missing from probe"
        assert "diverse" in domains, "Diverse domain missing from probe"

    def test_probe_samples_from_training_sources(self):
        """Test that each domain probe samples from training sources."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Get training source paths
        sources = config.get("sources", [])
        source_paths = {s["path"] for s in sources}

        # Check probe domains
        probe = config.get("probe", {})
        domains = probe.get("domains", {})

        for domain_name, domain_cfg in domains.items():
            if isinstance(domain_cfg, dict):
                probe_path = domain_cfg.get("path", "")
                assert probe_path in source_paths, (
                    f"Domain '{domain_name}' probe path '{probe_path}' "
                    f"not found in training sources: {source_paths}"
                )

    def test_probe_total_samples(self):
        """Test that total probe samples equals sum of domain samples."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        probe = config.get("probe", {})
        domains = probe.get("domains", {})
        total_samples = probe.get("total_samples", 0)

        domain_samples_sum = sum(
            d.get("samples", 0) for d in domains.values() if isinstance(d, dict)
        )

        assert total_samples == domain_samples_sum, (
            f"total_samples ({total_samples}) != sum of domain samples ({domain_samples_sum})"
        )

    def test_code_domain_uses_github_code(self):
        """Test code domain uses GitHub Code 2025 dataset."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        probe = config.get("probe", {})
        code_domain = probe.get("domains", {}).get("code", {})

        assert code_domain.get("path") == "nick007x/github-code-2025"
        assert code_domain.get("text_column") == "content"

    def test_math_domain_uses_finemath(self):
        """Test math domain uses FineMath dataset."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        probe = config.get("probe", {})
        math_domain = probe.get("domains", {}).get("math", {})

        assert math_domain.get("path") == "HuggingFaceTB/finemath"


class TestDCLMDataset:
    """Tests for DCLM dataset in training mix."""

    def test_dclm_in_config(self):
        """Test that DCLM is configured in training mix."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        sources = config.get("sources", [])
        source_names = [s["name"] for s in sources]

        assert "dclm" in source_names, "DCLM should be in dataset sources"

    def test_dclm_config_correct(self):
        """Test DCLM configuration is correct."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        sources = config.get("sources", [])
        dclm_config = next((s for s in sources if s["name"] == "dclm"), None)

        assert dclm_config is not None
        assert dclm_config["path"] == "mlfoundations/dclm-baseline-1.0-parquet"
        assert dclm_config["weight"] == 0.25  # 25% weight
        assert dclm_config["streaming"] is True

    def test_training_mix_weights_sum_to_one(self):
        """Test that all training mix weights sum to 1.0."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        sources = config.get("sources", [])
        total_weight = sum(s["weight"] for s in sources)

        assert abs(total_weight - 1.0) < 0.01, (
            f"Training mix weights should sum to 1.0, got {total_weight}"
        )

    def test_expected_sources_present(self):
        """Test that all expected sources are present."""
        from pathlib import Path
        import yaml

        config_path = Path(__file__).parent.parent / "configs" / "data" / "mixed_pretrain.yaml"

        if not config_path.exists():
            pytest.skip("Config file not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        sources = config.get("sources", [])
        source_names = {s["name"] for s in sources}

        expected_sources = {"dclm", "fineweb_edu", "finemath", "github_code", "slimpajama"}

        assert source_names == expected_sources, (
            f"Expected sources {expected_sources}, got {source_names}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
