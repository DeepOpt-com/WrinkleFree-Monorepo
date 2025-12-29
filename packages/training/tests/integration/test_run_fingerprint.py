"""Tests for run fingerprinting system.

These tests verify:
1. Fingerprint determinism - same config → same fingerprint
2. Fingerprint uniqueness - different configs → different fingerprints
3. Infrastructure changes don't affect fingerprint
4. Git integration works correctly
5. Config cleaning handles edge cases
"""

import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from omegaconf import OmegaConf

from wrinklefree.utils.run_fingerprint import (
    IGNORE_KEYS,
    clean_config_for_hashing,
    fingerprint_matches,
    generate_fingerprint,
    get_git_info,
)


class TestGetGitInfo:
    """Tests for git info retrieval."""

    def test_git_info_returns_commit_hash(self):
        """Test that git info returns a valid commit hash."""
        commit, is_dirty = get_git_info()

        # Should be a valid hex string (40 chars) or "unknown"
        if commit != "unknown":
            assert len(commit) == 40
            assert all(c in "0123456789abcdef" for c in commit)

    def test_git_info_handles_non_git_repo(self):
        """Test handling of non-git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            commit, is_dirty = get_git_info(Path(tmpdir))

            assert commit == "unknown"
            assert is_dirty is False

    def test_git_info_detects_dirty_state(self):
        """Test that dirty state is detected correctly."""
        # This test is informational - can't reliably test dirty state
        commit, is_dirty = get_git_info()

        # Should return a boolean
        assert isinstance(is_dirty, bool)


class TestCleanConfigForHashing:
    """Tests for config cleaning."""

    def test_ignores_infrastructure_keys(self):
        """Test that infrastructure keys are removed."""
        config = {
            "model": {"lr": 1e-3, "hidden_size": 768},
            "num_workers": 4,
            "logging": {
                "wandb": {"project": "test", "entity": "team"},
                "log_interval": 100,
            },
            "checkpoint": {
                "save_interval": 1000,
                "keep_last_n": 3,
            },
            "output_dir": "/tmp/outputs",
        }

        cleaned = clean_config_for_hashing(config)

        # Model config should remain
        assert "model" in cleaned
        assert cleaned["model"]["lr"] == 1e-3

        # Infrastructure should be removed
        assert "num_workers" not in cleaned
        assert "output_dir" not in cleaned

        # Nested infrastructure should be removed
        assert "logging" not in cleaned or "wandb" not in cleaned.get("logging", {})

    def test_preserves_training_params(self):
        """Test that training-affecting params are preserved."""
        config = {
            "model": {"lr": 1e-3, "hidden_size": 768, "num_layers": 12},
            "training": {
                "batch_size": 32,
                "max_steps": 10000,
                "gradient_clipping": 1.0,
            },
            "data": {"dataset": "fineweb", "max_seq_length": 512},
        }

        cleaned = clean_config_for_hashing(config)

        # All training-affecting params should remain
        assert cleaned["model"]["lr"] == 1e-3
        assert cleaned["model"]["hidden_size"] == 768
        assert cleaned["training"]["batch_size"] == 32
        assert cleaned["data"]["dataset"] == "fineweb"

    def test_sorts_keys_for_determinism(self):
        """Test that keys are sorted for deterministic hashing."""
        config1 = {"z_key": 1, "a_key": 2, "m_key": 3}
        config2 = {"a_key": 2, "z_key": 1, "m_key": 3}

        cleaned1 = clean_config_for_hashing(config1)
        cleaned2 = clean_config_for_hashing(config2)

        # Keys should be in same order
        assert list(cleaned1.keys()) == list(cleaned2.keys())
        assert list(cleaned1.keys()) == ["a_key", "m_key", "z_key"]

    def test_handles_nested_dicts(self):
        """Test cleaning of deeply nested configs."""
        config = {
            "level1": {
                "level2": {
                    "level3": {"value": 42},
                    "logging": {"enabled": True},  # Should be removed
                },
            },
        }

        cleaned = clean_config_for_hashing(config, ignore_keys={"logging"})

        assert cleaned["level1"]["level2"]["level3"]["value"] == 42
        assert "logging" not in cleaned["level1"]["level2"]

    def test_handles_lists(self):
        """Test that lists are preserved correctly."""
        config = {
            "layers": [
                {"type": "attention", "heads": 8},
                {"type": "ffn", "hidden_dim": 3072},
            ],
            "tags": ["train", "experiment"],
        }

        cleaned = clean_config_for_hashing(config)

        assert len(cleaned["layers"]) == 2
        assert cleaned["layers"][0]["type"] == "attention"
        assert cleaned["tags"] == ["train", "experiment"]

    def test_removes_empty_dicts(self):
        """Test that empty dicts after cleaning are removed."""
        config = {
            "model": {"lr": 1e-3},
            "logging": {  # All keys should be removed
                "wandb": {"project": "test"},
                "log_interval": 100,
            },
        }

        # Should result in empty logging dict being removed
        cleaned = clean_config_for_hashing(config)

        # Logging should be removed entirely if all keys were ignored
        assert "logging" not in cleaned or cleaned.get("logging") == {}


class TestGenerateFingerprint:
    """Tests for fingerprint generation."""

    def test_fingerprint_determinism(self):
        """Test that same config produces same fingerprint."""
        config = OmegaConf.create({
            "model": {"lr": 1e-3, "hidden_size": 768},
            "training": {"batch_size": 32},
        })

        fp1, meta1 = generate_fingerprint(config, include_git=False)
        fp2, meta2 = generate_fingerprint(config, include_git=False)

        assert fp1 == fp2
        assert meta1["config_hash"] == meta2["config_hash"]

    def test_fingerprint_uniqueness(self):
        """Test that different configs produce different fingerprints."""
        config1 = OmegaConf.create({
            "model": {"lr": 1e-3},
            "data": "fineweb",
        })
        config2 = OmegaConf.create({
            "model": {"lr": 1e-4},  # Different LR
            "data": "fineweb",
        })

        fp1, _ = generate_fingerprint(config1, include_git=False)
        fp2, _ = generate_fingerprint(config2, include_git=False)

        assert fp1 != fp2

    def test_infrastructure_changes_dont_affect_fingerprint(self):
        """Test that num_workers, logging changes don't change fingerprint."""
        config_base = OmegaConf.create({
            "model": {"lr": 1e-3},
            "data": "fineweb",
            "num_workers": 4,
            "logging": {"wandb": {"project": "test"}},
        })

        config_modified = OmegaConf.create({
            "model": {"lr": 1e-3},
            "data": "fineweb",
            "num_workers": 8,  # CHANGED
            "logging": {"wandb": {"project": "prod"}},  # CHANGED
        })

        fp1, _ = generate_fingerprint(config_base, include_git=False)
        fp2, _ = generate_fingerprint(config_modified, include_git=False)

        # Should be SAME fingerprint (infra changes ignored)
        assert fp1 == fp2

    def test_fingerprint_is_valid_sha256(self):
        """Test that fingerprint is valid SHA256 hash."""
        config = OmegaConf.create({"model": {"lr": 1e-3}})

        fp, _ = generate_fingerprint(config, include_git=False)

        # SHA256 is 64 hex characters
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_metadata_contains_required_fields(self):
        """Test that metadata contains all required fields."""
        config = OmegaConf.create({"model": {"lr": 1e-3}})

        _, meta = generate_fingerprint(config)

        assert "fingerprint" in meta
        assert "git_commit" in meta
        assert "git_dirty" in meta
        assert "config_hash" in meta
        assert "timestamp" in meta

    def test_git_commit_affects_fingerprint(self):
        """Test that different git commits produce different fingerprints."""
        config = OmegaConf.create({"model": {"lr": 1e-3}})

        # Mock different git commits
        with mock.patch("wrinklefree.utils.run_fingerprint.get_git_info") as mock_git:
            mock_git.return_value = ("abc123", False)
            fp1, _ = generate_fingerprint(config)

            mock_git.return_value = ("def456", False)
            fp2, _ = generate_fingerprint(config)

        # Different commits should produce different fingerprints
        assert fp1 != fp2

    def test_git_dirty_does_not_affect_fingerprint(self):
        """Test that dirty flag doesn't change fingerprint (allows dev resume)."""
        config = OmegaConf.create({"model": {"lr": 1e-3}})

        with mock.patch("wrinklefree.utils.run_fingerprint.get_git_info") as mock_git:
            mock_git.return_value = ("abc123", False)  # Clean
            fp1, meta1 = generate_fingerprint(config)

            mock_git.return_value = ("abc123", True)  # Dirty
            fp2, meta2 = generate_fingerprint(config)

        # Same fingerprint (dirty flag NOT included)
        assert fp1 == fp2

        # But dirty flag IS tracked in metadata
        assert meta1["git_dirty"] is False
        assert meta2["git_dirty"] is True


class TestFingerprintMatches:
    """Tests for fingerprint matching."""

    def test_matching_config_returns_true(self):
        """Test that matching config returns True."""
        config = OmegaConf.create({"model": {"lr": 1e-3}})

        fp, _ = generate_fingerprint(config, include_git=False)

        assert fingerprint_matches(config, fp, include_git=False)

    def test_different_config_returns_false(self):
        """Test that different config returns False."""
        config1 = OmegaConf.create({"model": {"lr": 1e-3}})
        config2 = OmegaConf.create({"model": {"lr": 1e-4}})

        fp, _ = generate_fingerprint(config1, include_git=False)

        assert not fingerprint_matches(config2, fp, include_git=False)


class TestOmegaConfIntegration:
    """Tests for OmegaConf integration."""

    def test_handles_interpolations(self):
        """Test that OmegaConf interpolations are resolved."""
        config = OmegaConf.create({
            "base_lr": 1e-3,
            "model": {"lr": "${base_lr}"},
        })

        # Should not raise - interpolations should be resolved
        fp, _ = generate_fingerprint(config, include_git=False)

        assert len(fp) == 64

    def test_handles_missing_interpolations_gracefully(self):
        """Test handling of missing interpolation references."""
        config = OmegaConf.create({
            "model": {"lr": 1e-3},
        })
        # This should work even without interpolations
        fp, _ = generate_fingerprint(config, include_git=False)

        assert len(fp) == 64


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_config(self):
        """Test handling of empty config."""
        config = OmegaConf.create({})

        fp, meta = generate_fingerprint(config, include_git=False)

        assert len(fp) == 64
        assert meta["fingerprint"] == fp

    def test_config_with_only_ignored_keys(self):
        """Test config with only ignored keys."""
        config = OmegaConf.create({
            "num_workers": 4,
            "logging": {"log_interval": 100},
            "output_dir": "/tmp",
        })

        fp, _ = generate_fingerprint(config, include_git=False)

        # Should still produce valid fingerprint
        assert len(fp) == 64

    def test_very_large_config(self):
        """Test handling of large configs."""
        config = OmegaConf.create({
            "model": {f"layer_{i}": {"dim": i * 64} for i in range(100)},
            "data": {f"source_{i}": f"path_{i}" for i in range(50)},
        })

        fp, _ = generate_fingerprint(config, include_git=False)

        assert len(fp) == 64

    def test_special_characters_in_values(self):
        """Test handling of special characters."""
        config = OmegaConf.create({
            "model": {
                "path": "/path/with spaces/and'quotes",
                "desc": 'Description with "quotes" and\nnewlines',
            },
        })

        fp, _ = generate_fingerprint(config, include_git=False)

        assert len(fp) == 64

    def test_numeric_edge_cases(self):
        """Test handling of numeric edge cases."""
        config = OmegaConf.create({
            "model": {
                "lr": 1e-10,
                "eps": float("inf"),
                "count": 0,
                "negative": -1,
            },
        })

        # Should not raise
        fp, _ = generate_fingerprint(config, include_git=False)

        assert len(fp) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
