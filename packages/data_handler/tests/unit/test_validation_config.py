"""Unit tests for validation dataset configuration.

Tests for loading and using the C4 validation config.
"""

import pytest
from pathlib import Path

from wf_data.data.config_loader import (
    load_data_config,
    list_available_configs,
    get_config_path,
)


class TestC4ValidationConfig:
    """Tests for the c4_validation config."""

    def test_c4_validation_config_exists(self):
        """Test that c4_validation config is listed as available."""
        configs = list_available_configs()
        assert "c4_validation" in configs, f"c4_validation not in available configs: {configs}"

    def test_c4_validation_config_path(self):
        """Test that c4_validation config path exists."""
        config_path = get_config_path("c4_validation")
        assert config_path.exists(), f"Config path does not exist: {config_path}"
        assert config_path.suffix == ".yaml"

    def test_load_c4_validation_config(self):
        """Test loading the c4_validation config."""
        config = load_data_config("c4_validation")

        # Check required fields
        assert "name" in config
        assert config["name"] == "c4_validation"

        # Check dataset config
        assert "dataset" in config
        dataset_cfg = config["dataset"]
        assert dataset_cfg["path"] == "allenai/c4"
        assert dataset_cfg["split"] == "validation"
        assert dataset_cfg["streaming"] is True

        # Check preprocessing
        assert "preprocessing" in config
        preprocessing = config["preprocessing"]
        assert preprocessing["max_length"] == 2048
        assert preprocessing["text_column"] == "text"
        assert preprocessing["packed"] is True

        # Check dataloader config
        assert "dataloader" in config
        dataloader_cfg = config["dataloader"]
        assert dataloader_cfg["batch_size"] == 8
        assert dataloader_cfg["num_workers"] == 2

    def test_c4_validation_is_single_source(self):
        """Test that c4_validation uses single-source mode (no 'sources' key)."""
        config = load_data_config("c4_validation")

        # Should have 'dataset' key (single-source) not 'sources' (multi-source)
        assert "dataset" in config
        assert "sources" not in config


class TestValidationConfigIntegration:
    """Integration tests for validation config with dataloader creation."""

    def test_validation_config_compatible_with_factory(self):
        """Test that c4_validation config is compatible with create_dataloader."""
        from wf_data.data.config_loader import load_data_config

        config = load_data_config("c4_validation")

        # Verify config has fields expected by factory
        # Single-source mode requires 'dataset' with 'path' and 'split'
        assert "dataset" in config
        assert "path" in config["dataset"]
        assert "split" in config["dataset"]

        # Verify streaming is enabled (required for C4 validation)
        assert config["dataset"].get("streaming", False) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
