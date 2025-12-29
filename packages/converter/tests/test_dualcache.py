"""Unit tests for DualCache mechanism in Fast-dLLM v2.

Tests the DualCache implementation which maintains both prefix and suffix
KV caches for partially decoded blocks, enabling sub-block re-use.

Reference: Fast-dLLM v2 (arXiv:2509.26328)
"""

import pytest
import torch


class TestDualCacheBasics:
    """Test basic DualCache data structures and parameters."""

    def test_use_block_cache_parameter_exists(self):
        """Verify use_block_cache parameter is recognized."""
        from wf_dlm_converter.inference import generate_with_dualcache

        # Check function signature accepts use_block_cache
        import inspect

        sig = inspect.signature(generate_with_dualcache)
        assert "use_block_cache" in sig.parameters
        assert sig.parameters["use_block_cache"].default is True

    def test_small_block_size_parameter_exists(self):
        """Verify small_block_size parameter for sub-block iteration."""
        from wf_dlm_converter.inference import generate_with_dualcache

        import inspect

        sig = inspect.signature(generate_with_dualcache)
        assert "small_block_size" in sig.parameters
        assert sig.parameters["small_block_size"].default == 8

    def test_block_size_divisible_by_small_block_size(self):
        """Block size must be divisible by small block size."""
        block_size = 32
        small_block_size = 8

        assert block_size % small_block_size == 0
        num_small_blocks = block_size // small_block_size
        assert num_small_blocks == 4

    def test_replace_position_tensor_shape(self):
        """Test replace_position tensor has correct shape for cache updates."""
        batch_size = 2
        block_size = 32
        small_block_size = 8

        # replace_position marks which positions in cache to update
        replace_position = torch.zeros((batch_size, block_size), dtype=torch.bool)

        # Mark small block 1 (positions 8-16) for replacement
        start_idx = 1 * small_block_size
        end_idx = start_idx + small_block_size
        replace_position[:, start_idx:end_idx] = True

        assert replace_position.shape == (batch_size, block_size)
        assert replace_position[:, start_idx:end_idx].all()
        assert not replace_position[:, :start_idx].any()


class TestCacheReplacement:
    """Test DualCache replacement vs concatenation logic."""

    def test_cache_replacement_vs_concat_behavior(self):
        """dual_cache=True should replace positions, not concatenate."""
        # Simulate the cache update logic from modeling_dream.py:403-414
        batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 64

        # Past cache (full block)
        past_key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        past_value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # New states for subset of positions
        new_key = torch.randn(batch_size, num_heads, 8, head_dim)  # small_block
        new_value = torch.randn(batch_size, num_heads, 8, head_dim)

        # Replace indices (positions 8-15)
        replace_indices = torch.arange(8, 16)

        # DualCache behavior: replace specific positions
        past_key_replaced = past_key.clone()
        past_key_replaced[:, :, replace_indices, :] = new_key
        past_value_replaced = past_value.clone()
        past_value_replaced[:, :, replace_indices, :] = new_value

        # Shape should remain the same (replacement, not concat)
        assert past_key_replaced.shape == past_key.shape
        assert past_value_replaced.shape == past_value.shape

        # Values at replace_indices should be updated
        assert torch.allclose(past_key_replaced[:, :, replace_indices, :], new_key)

        # Values outside replace_indices should be unchanged
        assert torch.allclose(past_key_replaced[:, :, :8, :], past_key[:, :, :8, :])

    def test_standard_concat_behavior(self):
        """Without dual_cache, should concatenate KV cache."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 64

        past_key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        new_key = torch.randn(batch_size, num_heads, 8, head_dim)

        # Standard behavior: concatenate
        concat_key = torch.cat([past_key, new_key], dim=2)

        # Shape grows with concatenation
        assert concat_key.shape == (batch_size, num_heads, seq_len + 8, head_dim)


class TestSubBlockIteration:
    """Test sub-block iteration for cache reuse."""

    def test_num_small_blocks_calculation(self):
        """Verify correct number of sub-blocks."""
        test_cases = [
            (32, 8, 4),   # 32/8 = 4 sub-blocks
            (32, 4, 8),   # 32/4 = 8 sub-blocks
            (32, 16, 2),  # 32/16 = 2 sub-blocks
            (32, 32, 1),  # 32/32 = 1 sub-block (no sub-block iteration)
        ]

        for block_size, small_block_size, expected in test_cases:
            num_small_blocks = block_size // small_block_size
            assert num_small_blocks == expected

    def test_small_block_indices(self):
        """Test correct index calculation for sub-blocks."""
        block_size = 32
        small_block_size = 8
        num_small_blocks = block_size // small_block_size

        expected_ranges = [
            (0, 8),
            (8, 16),
            (16, 24),
            (24, 32),
        ]

        for small_block_idx in range(num_small_blocks):
            start_idx = small_block_idx * small_block_size
            end_idx = start_idx + small_block_size
            assert (start_idx, end_idx) == expected_ranges[small_block_idx]

    def test_cache_reuse_condition(self):
        """Cache should be reused when first token in sub-block is not masked."""
        block_size = 32
        small_block_size = 8
        mask_id = 151665

        # Simulated block tokens (some masked, some revealed)
        x_t = torch.tensor([[
            1, 2, 3, 4, 5, 6, 7, 8,  # Sub-block 0: all revealed
            mask_id, mask_id, 11, 12, mask_id, 14, 15, mask_id,  # Sub-block 1: partially masked
            17, 18, 19, 20, 21, 22, 23, 24,  # Sub-block 2: all revealed
            mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id, mask_id,  # Sub-block 3: all masked
        ]])

        # Check reuse condition for each sub-block
        for small_block_idx in range(4):
            start_idx = small_block_idx * small_block_size
            first_token = x_t[0, start_idx]

            # Cache can be reused if first token is not masked
            can_reuse = first_token != mask_id

            expected = [True, False, True, False]
            assert can_reuse == expected[small_block_idx], f"Sub-block {small_block_idx}"


@pytest.mark.gpu
class TestDualCacheGPU:
    """GPU-specific DualCache tests."""

    def test_cache_on_gpu(self, skip_if_no_gpu):
        """Verify cache tensors are on GPU when expected."""
        device = torch.device("cuda")

        batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 64
        past_key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        assert past_key.device.type == "cuda"

        # Replace operation should stay on GPU
        replace_indices = torch.arange(8, 16, device=device)
        new_key = torch.randn(batch_size, num_heads, 8, head_dim, device=device)

        past_key[:, :, replace_indices, :] = new_key
        assert past_key.device.type == "cuda"


class TestMaskTokenHandling:
    """Test mask token identification and handling."""

    def test_get_mask_id_from_tokenizer(self):
        """Should find mask token from tokenizer vocab."""
        from wf_dlm_converter.inference import _get_mask_id, DEFAULT_MASK_TOKEN
        from unittest.mock import MagicMock

        # Mock tokenizer with mask token
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {DEFAULT_MASK_TOKEN: 151665, "other": 1}
        mock_tokenizer.encode.return_value = [151665]

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        del mock_model.config.mask_token_id  # No config mask_token_id

        mask_id = _get_mask_id(mock_tokenizer, mock_model)
        assert mask_id == 151665

    def test_get_mask_id_from_config(self):
        """Should fallback to model config mask_token_id."""
        from wf_dlm_converter.inference import _get_mask_id, DEFAULT_MASK_TOKEN
        from unittest.mock import MagicMock

        # Mock tokenizer without mask token
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"other": 1}

        # Mock model with config.mask_token_id
        mock_model = MagicMock()
        mock_model.config.mask_token_id = 99999

        mask_id = _get_mask_id(mock_tokenizer, mock_model)
        assert mask_id == 99999

    def test_get_mask_id_raises_without_mask(self):
        """Should raise error if no mask token found."""
        from wf_dlm_converter.inference import _get_mask_id
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"other": 1}

        mock_model = MagicMock()
        mock_model.config = MagicMock(spec=[])  # No mask_token_id or bd_size

        with pytest.raises(ValueError, match="mask token"):
            _get_mask_id(mock_tokenizer, mock_model)


class TestSamplingFunctions:
    """Test sampling helper functions."""

    def test_sample_with_top_p_greedy(self):
        """Temperature=0 should give greedy sampling."""
        from wf_dlm_converter.inference import _sample_with_top_p

        logits = torch.tensor([[[1.0, 2.0, 3.0, 0.5]]])  # Token 2 has highest logit

        tokens, probs = _sample_with_top_p(logits, top_p=0.95, temperature=0.0)

        assert tokens[0, 0] == 2  # Should select token with highest logit

    def test_sample_with_top_p_shape(self):
        """Output should have correct shape."""
        from wf_dlm_converter.inference import _sample_with_top_p

        batch_size, seq_len, vocab_size = 2, 8, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        tokens, probs = _sample_with_top_p(logits, top_p=0.95, temperature=1.0)

        assert tokens.shape == (batch_size, seq_len)
        assert probs.shape == (batch_size, seq_len, vocab_size)


class TestGenerationResult:
    """Test GenerationResult dataclass."""

    def test_generation_result_fields(self):
        """Verify all expected fields exist."""
        from wf_dlm_converter.inference import GenerationResult

        result = GenerationResult(
            text="Hello, world!",
            tokens_generated=3,
            elapsed_seconds=0.5,
            tokens_per_second=6.0,
            nfe=10,
            used_dualcache=True,
        )

        assert result.text == "Hello, world!"
        assert result.tokens_generated == 3
        assert result.nfe == 10
        assert result.used_dualcache is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
