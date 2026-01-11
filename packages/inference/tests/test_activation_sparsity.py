"""Tests for activation sparsity module."""

import pytest
import torch

from wf_infer.sglang_backend.activation_sparsity import (
    ActivationSparsityConfig,
    SparsityMode,
    apply_sparsity,
    apply_threshold_sparsity,
    apply_top_k_sparsity,
    apply_adaptive_sparsity,
    measure_sparsity,
    get_default_config,
    get_qsparse_config,
    get_conservative_config,
    get_adaptive_config,
)


class TestMeasureSparsity:
    """Tests for measure_sparsity function."""

    def test_all_zeros(self):
        x = torch.zeros(10, 10)
        assert measure_sparsity(x) == 1.0

    def test_no_zeros(self):
        x = torch.ones(10, 10)
        assert measure_sparsity(x) == 0.0

    def test_half_zeros(self):
        x = torch.tensor([0.0, 0.0, 1.0, 1.0])
        assert abs(measure_sparsity(x) - 0.5) < 0.01

    def test_threshold(self):
        x = torch.tensor([0.0001, 0.001, 0.01, 0.1])
        # With threshold=0.005, first two are "zero"
        assert abs(measure_sparsity(x, threshold=0.005) - 0.5) < 0.01


class TestThresholdSparsity:
    """Tests for threshold-based sparsity."""

    def test_basic(self):
        x = torch.tensor([[0.001, 0.1, 0.5, 0.001]])
        sparse_x, sparsity = apply_threshold_sparsity(x, threshold=0.01)

        # Values below 0.01 should be zeroed
        assert sparse_x[0, 0].item() == 0.0
        assert sparse_x[0, 3].item() == 0.0
        assert abs(sparse_x[0, 1].item() - 0.1) < 1e-5
        assert abs(sparse_x[0, 2].item() - 0.5) < 1e-5
        assert abs(sparsity - 0.5) < 0.01

    def test_all_above_threshold(self):
        x = torch.ones(4, 4) * 0.5
        sparse_x, sparsity = apply_threshold_sparsity(x, threshold=0.01)
        assert torch.allclose(sparse_x, x)
        assert sparsity == 0.0

    def test_all_below_threshold(self):
        x = torch.ones(4, 4) * 0.001
        sparse_x, sparsity = apply_threshold_sparsity(x, threshold=0.01)
        assert torch.allclose(sparse_x, torch.zeros_like(x))
        assert sparsity == 1.0


class TestTopKSparsity:
    """Tests for top-k sparsity."""

    def test_keep_half(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        sparse_x, sparsity = apply_top_k_sparsity(x, ratio=0.5)

        # Should keep top 2 values (3.0 and 4.0)
        assert sparse_x[0, 2].item() == 3.0
        assert sparse_x[0, 3].item() == 4.0
        assert sparse_x[0, 0].item() == 0.0
        assert sparse_x[0, 1].item() == 0.0
        assert abs(sparsity - 0.5) < 0.01

    def test_keep_all(self):
        x = torch.randn(4, 4)
        sparse_x, sparsity = apply_top_k_sparsity(x, ratio=1.0)
        assert torch.allclose(sparse_x, x)
        assert sparsity == 0.0

    def test_keep_none(self):
        x = torch.randn(4, 4)
        sparse_x, sparsity = apply_top_k_sparsity(x, ratio=0.0)
        assert torch.allclose(sparse_x, torch.zeros_like(x))
        assert sparsity == 1.0

    def test_per_token(self):
        # Each row should have its own top-k selection
        x = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ])
        sparse_x, sparsity = apply_top_k_sparsity(x, ratio=0.5)

        # Row 0: keep indices 2, 3
        assert sparse_x[0, 2].item() == 3.0
        assert sparse_x[0, 3].item() == 4.0

        # Row 1: keep indices 0, 1
        assert sparse_x[1, 0].item() == 4.0
        assert sparse_x[1, 1].item() == 3.0

    def test_qsparse_ratio(self):
        """Test with Q-Sparse optimal ratio of 0.4 (60% sparsity)."""
        x = torch.randn(32, 4096)
        sparse_x, sparsity = apply_top_k_sparsity(x, ratio=0.4)
        assert abs(sparsity - 0.6) < 0.01

        # Check that output has correct number of non-zeros per row
        non_zeros_per_row = (sparse_x != 0).sum(dim=1)
        expected = int(4096 * 0.4)
        assert (non_zeros_per_row == expected).all()


class TestAdaptiveSparsity:
    """Tests for adaptive sparsity."""

    def test_basic(self):
        x = torch.randn(4, 100)
        sparse_x, sparsity = apply_adaptive_sparsity(x, min_ratio=0.3, max_ratio=0.7)

        # Sparsity should be between 30% and 70%
        assert 0.25 < sparsity < 0.75

    def test_high_variance_keeps_more(self):
        # High variance token should keep more activations
        x = torch.zeros(2, 100)
        x[0, :] = torch.randn(100) * 10  # High variance
        x[1, :] = torch.randn(100) * 0.1  # Low variance

        sparse_x, _ = apply_adaptive_sparsity(x, min_ratio=0.3, max_ratio=0.7)

        # High variance row should have more non-zeros
        high_var_nonzeros = (sparse_x[0] != 0).sum()
        low_var_nonzeros = (sparse_x[1] != 0).sum()
        assert high_var_nonzeros > low_var_nonzeros

    def test_uniform_variance(self):
        # When all tokens have same variance, use mean ratio
        x = torch.ones(4, 100) * 0.5
        sparse_x, sparsity = apply_adaptive_sparsity(x, min_ratio=0.3, max_ratio=0.7)
        assert abs(sparsity - 0.5) < 0.1


class TestApplySparsity:
    """Tests for the main apply_sparsity function."""

    def test_disabled(self):
        config = ActivationSparsityConfig(enabled=False)
        x = torch.randn(4, 4)
        sparse_x, sparsity = apply_sparsity(x, config)
        assert torch.allclose(sparse_x, x)
        assert sparsity == 0.0

    def test_none_mode(self):
        config = ActivationSparsityConfig(enabled=True, mode=SparsityMode.NONE)
        x = torch.randn(4, 4)
        sparse_x, sparsity = apply_sparsity(x, config)
        assert torch.allclose(sparse_x, x)
        assert sparsity == 0.0

    def test_threshold_mode(self):
        config = ActivationSparsityConfig(
            enabled=True,
            mode=SparsityMode.THRESHOLD,
            threshold=0.5,
        )
        x = torch.tensor([[0.1, 0.6, 0.4, 0.9]])
        sparse_x, sparsity = apply_sparsity(x, config)
        assert sparse_x[0, 0].item() == 0.0
        assert abs(sparse_x[0, 1].item() - 0.6) < 1e-5
        assert sparse_x[0, 2].item() == 0.0
        assert abs(sparse_x[0, 3].item() - 0.9) < 1e-5

    def test_top_k_mode(self):
        config = ActivationSparsityConfig(
            enabled=True,
            mode=SparsityMode.TOP_K,
            top_k_ratio=0.5,
        )
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        sparse_x, sparsity = apply_sparsity(x, config)
        assert sparse_x[0, 0].item() == 0.0
        assert sparse_x[0, 1].item() == 0.0
        assert sparse_x[0, 2].item() == 3.0
        assert sparse_x[0, 3].item() == 4.0

    def test_tracking(self):
        config = ActivationSparsityConfig(
            enabled=True,
            mode=SparsityMode.TOP_K,
            top_k_ratio=0.5,
            track_stats=True,
        )

        for _ in range(10):
            x = torch.randn(4, 100)
            apply_sparsity(x, config)

        assert len(config._sparsity_history) == 10
        assert abs(config.get_average_sparsity() - 0.5) < 0.01

        config.clear_stats()
        assert len(config._sparsity_history) == 0


class TestConvenienceFunctions:
    """Tests for convenience config functions."""

    def test_default_config(self):
        config = get_default_config()
        assert not config.enabled

    def test_qsparse_config(self):
        config = get_qsparse_config()
        assert config.enabled
        assert config.mode == SparsityMode.TOP_K
        assert config.top_k_ratio == 0.4  # 60% sparsity

    def test_conservative_config(self):
        config = get_conservative_config()
        assert config.enabled
        assert config.top_k_ratio == 0.7  # 30% sparsity

    def test_adaptive_config(self):
        config = get_adaptive_config()
        assert config.enabled
        assert config.mode == SparsityMode.ADAPTIVE


class TestQualityPreservation:
    """Tests to ensure output quality is preserved with sparsity."""

    def test_cosine_similarity_threshold(self):
        """Output should maintain high cosine similarity with dense baseline."""
        x = torch.randn(32, 4096)

        # Get dense output
        dense_out = x.clone()

        # Get sparse output (60% sparsity)
        sparse_out, _ = apply_top_k_sparsity(x, ratio=0.4)

        # Compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            dense_out.flatten().unsqueeze(0),
            sparse_out.flatten().unsqueeze(0),
        ).item()

        # Should be > 0.8 for 60% sparsity (conservative check)
        assert cos_sim > 0.8

    def test_conservative_sparsity_quality(self):
        """Conservative sparsity should have very high quality."""
        x = torch.randn(32, 4096)
        dense_out = x.clone()
        sparse_out, _ = apply_top_k_sparsity(x, ratio=0.7)  # 30% sparsity

        cos_sim = torch.nn.functional.cosine_similarity(
            dense_out.flatten().unsqueeze(0),
            sparse_out.flatten().unsqueeze(0),
        ).item()

        # Should be > 0.95 for 30% sparsity
        assert cos_sim > 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
