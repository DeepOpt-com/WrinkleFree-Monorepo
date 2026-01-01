"""Tests for sequence packing with position_ids reset in CheaperTraining.

See: https://huggingface.co/blog/sirluk/llm-sequence-packing
"""

import pytest
import torch
from torch.utils.data import IterableDataset

from cheapertraining.data.mixing import PackedDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def __call__(self, texts, **kwargs):
        """Simple tokenization: each word becomes a token."""
        input_ids = []
        for text in texts:
            tokens = [100 + i for i, _ in enumerate(text.split())]
            input_ids.append(tokens)
        return {"input_ids": input_ids}


class MockTextDataset(IterableDataset):
    """Mock dataset that yields text samples."""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __iter__(self):
        for text in self.texts:
            yield {"text": text}


class TestPackedDatasetPositionIds:
    """Tests for position_ids computation in PackedDataset."""

    def test_position_ids_reset_at_separator(self):
        """Test that position_ids reset after separator tokens."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
            separator_token_id=2,
        )

        # Simulate a packed sequence with separator tokens
        input_ids = torch.tensor([10, 11, 12, 2, 20, 21, 2, 30, 31, 32, 33, 2, 40, 41, 42, 43])

        position_ids = dataset._compute_position_ids(input_ids)

        # Expected: positions reset after each separator
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])

        assert torch.equal(position_ids, expected), f"Got {position_ids}, expected {expected}"

    def test_position_ids_with_custom_separator(self):
        """Test with separator_token_id different from eos_token_id."""
        # Use separator=99, eos=2
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
            separator_token_id=99,  # Custom separator
        )

        # Sequence with custom separator (99), not EOS (2)
        input_ids = torch.tensor([10, 11, 99, 20, 21, 22, 99, 30])

        position_ids = dataset._compute_position_ids(input_ids)

        # Should reset at 99, not at 2
        expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0])

        assert torch.equal(position_ids, expected)

    def test_position_ids_no_separator(self):
        """Test position_ids when there are no separator tokens."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
            separator_token_id=2,
        )

        input_ids = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])

        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        assert torch.equal(position_ids, expected)

    def test_position_ids_consecutive_separators(self):
        """Test position_ids with consecutive separator tokens."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
            separator_token_id=2,
        )

        input_ids = torch.tensor([10, 11, 2, 2, 20, 21, 22, 2])

        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0, 1, 2, 0, 0, 1, 2, 3])

        assert torch.equal(position_ids, expected)

    def test_position_ids_starts_with_separator(self):
        """Test position_ids when sequence starts with separator."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
            separator_token_id=2,
        )

        input_ids = torch.tensor([2, 10, 11, 12, 2, 20, 21, 22])

        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0, 0, 1, 2, 3, 0, 1, 2])

        assert torch.equal(position_ids, expected)

    def test_position_ids_dtype(self):
        """Test that position_ids have same dtype as input_ids."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10, 11, 2, 20, 21], dtype=torch.long)
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.dtype == input_ids.dtype

    def test_position_ids_shape(self):
        """Test that position_ids have same shape as input_ids."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
        )

        input_ids = torch.tensor([10, 11, 12, 2, 20, 21, 2, 30])
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.shape == input_ids.shape


class TestPackedDatasetEdgeCases:
    """Edge case tests for PackedDataset position_ids."""

    def test_all_separator_tokens(self):
        """Test sequence of only separator tokens."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
            separator_token_id=2,
        )

        input_ids = torch.tensor([2, 2, 2, 2])
        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0, 0, 0, 0])

        assert torch.equal(position_ids, expected)

    def test_single_token(self):
        """Test single token sequence."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10])
        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0])
        assert torch.equal(position_ids, expected)

    def test_empty_tensor(self):
        """Test with empty input tensor."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([], dtype=torch.long)
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.shape == (0,)
        assert position_ids.dtype == torch.long


class TestDifferentSeparatorTokenIds:
    """Test position_ids with various separator token IDs."""

    @pytest.mark.parametrize("separator_id", [1, 2, 50256, 128001])
    def test_position_ids_with_various_separators(self, separator_id):
        """Test position reset works with different separator token IDs."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
            separator_token_id=separator_id,
        )

        input_ids = torch.tensor([10, 11, 12, separator_id, 20, 21, separator_id, 30])

        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0])

        assert torch.equal(position_ids, expected)


class TestPackedDatasetIntegration:
    """Integration tests for PackedDataset."""

    def test_has_compute_position_ids_method(self):
        """Verify PackedDataset has _compute_position_ids method."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        assert hasattr(dataset, "_compute_position_ids")
        assert callable(dataset._compute_position_ids)

    def test_position_ids_device_matches_input_ids(self):
        """Verify position_ids created on same device as input_ids."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10, 11, 2, 20, 21])
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.device == input_ids.device

    def test_position_ids_contiguous(self):
        """Verify position_ids tensor is contiguous."""
        dataset = PackedDataset(
            dataset=MockTextDataset([]),
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10, 11, 2, 20, 21, 2, 30, 31])
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.is_contiguous()
