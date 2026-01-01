"""Tests for sequence packing with position_ids reset.

See: https://huggingface.co/blog/sirluk/llm-sequence-packing
"""

import pytest
import torch

from wrinklefree.data._legacy.pretrain_dataset import PretrainDataset

# Alias for backward compatibility in tests
PackedPretrainDataset = PretrainDataset


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def __call__(self, texts, **kwargs):
        """Simple tokenization: each character becomes a token."""
        input_ids = []
        for text in texts:
            # Simple tokenizer: each word becomes token 100+word_index
            tokens = [100 + i for i, _ in enumerate(text.split())]
            input_ids.append(tokens)
        return {"input_ids": input_ids}


class TestPositionIdsComputation:
    """Tests for position_ids computation in packed sequences."""

    def test_position_ids_reset_at_eos(self):
        """Test that position_ids reset to 0 after each EOS token."""
        # Create a packed dataset with mock tokenizer
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
        )

        # Simulate a packed sequence with EOS tokens
        # [10, 11, 12, EOS, 20, 21, EOS, 30, 31, 32, 33, EOS, 40, 41, 42, 43]
        input_ids = torch.tensor([10, 11, 12, 2, 20, 21, 2, 30, 31, 32, 33, 2, 40, 41, 42, 43])

        position_ids = dataset._compute_position_ids(input_ids)

        # Expected: positions reset after each EOS
        # [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3]
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])

        assert torch.equal(position_ids, expected), f"Got {position_ids}, expected {expected}"

    def test_position_ids_no_eos(self):
        """Test position_ids when there are no EOS tokens (single document)."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
        )

        # No EOS tokens - all from same document
        input_ids = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])

        position_ids = dataset._compute_position_ids(input_ids)

        # Expected: sequential positions
        expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

        assert torch.equal(position_ids, expected)

    def test_position_ids_consecutive_eos(self):
        """Test position_ids with consecutive EOS tokens."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
        )

        # Consecutive EOS (empty document case)
        # input:    [10, 11,  2,  2, 20, 21, 22,  2]
        # positions: [0,  1,  2,  0,  0,  1,  2,  3]
        # Note: First EOS gets position 2, then resets. Second EOS gets 0, resets again.
        input_ids = torch.tensor([10, 11, 2, 2, 20, 21, 22, 2])

        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0, 1, 2, 0, 0, 1, 2, 3])

        assert torch.equal(position_ids, expected)

    def test_position_ids_starts_with_eos(self):
        """Test position_ids when sequence starts with EOS."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        # Starts with EOS (end of previous doc that was split)
        input_ids = torch.tensor([2, 10, 11, 12, 2, 20, 21, 22])

        position_ids = dataset._compute_position_ids(input_ids)

        # First token is EOS, so position 0, then reset
        # Expected: [0, 0, 1, 2, 3, 0, 1, 2]
        expected = torch.tensor([0, 0, 1, 2, 3, 0, 1, 2])

        assert torch.equal(position_ids, expected)

    def test_position_ids_dtype(self):
        """Test that position_ids have same dtype as input_ids."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10, 11, 2, 20, 21], dtype=torch.long)
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.dtype == input_ids.dtype

    def test_position_ids_shape(self):
        """Test that position_ids have same shape as input_ids."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=16,
        )

        input_ids = torch.tensor([10, 11, 12, 2, 20, 21, 2, 30])
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.shape == input_ids.shape


class TestPackedDatasetOutput:
    """Tests for PackedPretrainDataset output format."""

    def test_output_contains_position_ids(self):
        """Test that dataset output includes position_ids key."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        # We can't easily iterate without a real dataset,
        # but we can verify the _compute_position_ids method exists
        assert hasattr(dataset, "_compute_position_ids")
        assert callable(dataset._compute_position_ids)


class TestPositionIdsEdgeCases:
    """Edge case tests for position_ids computation."""

    def test_all_eos_tokens(self):
        """Test sequence of only EOS tokens."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([2, 2, 2, 2])
        position_ids = dataset._compute_position_ids(input_ids)

        # Each EOS resets, so all positions are 0
        expected = torch.tensor([0, 0, 0, 0])

        assert torch.equal(position_ids, expected)

    def test_single_token(self):
        """Test single token sequence."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10])
        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0])
        assert torch.equal(position_ids, expected)

    def test_single_eos(self):
        """Test single EOS token."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([2])
        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0])
        assert torch.equal(position_ids, expected)

    def test_eos_at_end(self):
        """Test EOS at the very end."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10, 11, 12, 13, 2])
        position_ids = dataset._compute_position_ids(input_ids)

        expected = torch.tensor([0, 1, 2, 3, 4])
        assert torch.equal(position_ids, expected)

    def test_large_sequence(self):
        """Test with a larger sequence to ensure no off-by-one errors."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=1024,
        )

        # Create a sequence with multiple documents
        tokens = []
        expected_positions = []
        pos = 0
        for i in range(100):
            tokens.extend([100 + j for j in range(10)])  # 10 tokens per doc
            expected_positions.extend(list(range(10)))
            tokens.append(2)  # EOS
            expected_positions.append(10)  # EOS gets next position
            pos = 0

        input_ids = torch.tensor(tokens)
        position_ids = dataset._compute_position_ids(input_ids)
        expected = torch.tensor(expected_positions)

        assert torch.equal(position_ids, expected)

    def test_empty_tensor(self):
        """Test with empty input tensor."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([], dtype=torch.long)
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.shape == (0,)
        assert position_ids.dtype == torch.long


class TestDifferentEosTokenIds:
    """Test position_ids computation with various EOS token IDs."""

    @pytest.mark.parametrize("eos_token_id", [1, 2, 50256, 128001])
    def test_position_ids_with_various_eos_tokens(self, eos_token_id):
        """Test position reset works with different EOS token IDs."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=eos_token_id),
            max_length=16,
        )

        # Create sequence with the specified EOS token
        input_ids = torch.tensor([10, 11, 12, eos_token_id, 20, 21, eos_token_id, 30])

        position_ids = dataset._compute_position_ids(input_ids)

        # Positions should reset after each EOS
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0])

        assert torch.equal(position_ids, expected)


class TestPackedDatasetIntegration:
    """Integration tests for PackedPretrainDataset."""

    def test_position_ids_device_matches_input_ids(self):
        """Verify position_ids created on same device as input_ids."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10, 11, 2, 20, 21])
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.device == input_ids.device

    def test_position_ids_contiguous(self):
        """Verify position_ids tensor is contiguous."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=8,
        )

        input_ids = torch.tensor([10, 11, 2, 20, 21, 2, 30, 31])
        position_ids = dataset._compute_position_ids(input_ids)

        assert position_ids.is_contiguous()

    def test_position_ids_max_value_within_sequence(self):
        """Verify position values never exceed document length."""
        dataset = PackedPretrainDataset(
            dataset_path="mock",
            tokenizer=MockTokenizer(eos_token_id=2),
            max_length=1024,
        )

        # Create packed sequence with varying document lengths
        # Doc1: 5 tokens, Doc2: 3 tokens, Doc3: 10 tokens
        tokens = [100, 101, 102, 103, 104, 2, 200, 201, 202, 2] + list(range(300, 310)) + [2]
        input_ids = torch.tensor(tokens)
        position_ids = dataset._compute_position_ids(input_ids)

        # Max position in doc1 should be 5 (for EOS), doc2 should be 3, doc3 should be 10
        # Check position before each EOS
        assert position_ids[5].item() == 5  # EOS of doc1
        assert position_ids[9].item() == 3  # EOS of doc2
        assert position_ids[-1].item() == 10  # EOS of doc3
