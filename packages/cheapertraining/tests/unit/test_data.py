"""Unit tests for data loading components.

Tests for MixedDataset, PackedDataset, and DataLoader optimizations.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch

from cheapertraining.data.mixing import (
    DatasetMixture,
    MixedDataset,
    PackedDataset,
    create_mixed_dataset,
    create_mixed_dataloader,
    _worker_init_fn,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=True, truncation=False):
        """Mock encode method."""
        return list(range(len(text.split())))

    def __call__(
        self,
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=True,
    ):
        """Mock batch tokenization."""
        if isinstance(texts, str):
            texts = [texts]

        input_ids = [list(range(len(t.split()))) for t in texts]
        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = [[1] * len(ids) for ids in input_ids]
        return result


class TestDatasetMixture:
    """Tests for DatasetMixture dataclass."""

    def test_default_values(self):
        """Test default values for DatasetMixture."""
        mixture = DatasetMixture(
            name="test",
            weight=1.0,
            path="test/dataset",
        )
        assert mixture.name == "test"
        assert mixture.weight == 1.0
        assert mixture.subset is None
        assert mixture.split == "train"
        assert mixture.text_column == "text"

    def test_custom_values(self):
        """Test custom values for DatasetMixture."""
        mixture = DatasetMixture(
            name="code",
            weight=0.3,
            path="bigcode/starcoderdata",
            subset="python",
            split="validation",
            text_column="content",
        )
        assert mixture.name == "code"
        assert mixture.weight == 0.3
        assert mixture.subset == "python"
        assert mixture.split == "validation"
        assert mixture.text_column == "content"

    def test_list_text_column(self):
        """Test DatasetMixture with list of columns for concatenation."""
        mixture = DatasetMixture(
            name="synth",
            weight=0.1,
            path="PleIAs/SYNTH",
            text_column=["query", "synthetic_reasoning", "synthetic_answer"],
            text_separator="\n\n",
        )
        assert mixture.name == "synth"
        assert mixture.text_column == ["query", "synthetic_reasoning", "synthetic_answer"]
        assert mixture.text_separator == "\n\n"


class TestMixedDataset:
    """Tests for MixedDataset."""

    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        mixtures = [
            DatasetMixture(name="a", weight=2.0, path="test/a"),
            DatasetMixture(name="b", weight=3.0, path="test/b"),
        ]
        dataset = MixedDataset(mixtures=mixtures, streaming=False)

        # Weights should sum to 1
        assert abs(dataset.normalized_weights.sum().item() - 1.0) < 1e-6
        # Weight proportions should be preserved (2:3 ratio)
        assert abs(dataset.normalized_weights[0].item() - 0.4) < 1e-6
        assert abs(dataset.normalized_weights[1].item() - 0.6) < 1e-6

    def test_update_weights_from_influence(self):
        """Test dynamic weight update from influence scores."""
        mixtures = [
            DatasetMixture(name="a", weight=0.5, path="test/a"),
            DatasetMixture(name="b", weight=0.5, path="test/b"),
        ]
        dataset = MixedDataset(mixtures=mixtures, streaming=False)

        # Update weights based on influence
        dataset.update_weights_from_influence({"a": 0.8, "b": 0.2})

        # Verify new weights
        assert abs(dataset.normalized_weights[0].item() - 0.8) < 1e-6
        assert abs(dataset.normalized_weights[1].item() - 0.2) < 1e-6

    def test_update_weights_partial(self):
        """Test partial weight update (only some mixtures specified)."""
        mixtures = [
            DatasetMixture(name="a", weight=0.5, path="test/a"),
            DatasetMixture(name="b", weight=0.5, path="test/b"),
        ]
        dataset = MixedDataset(mixtures=mixtures, streaming=False)

        # Only update 'a' weight
        dataset.update_weights_from_influence({"a": 0.9})

        # 'b' should keep original weight, then normalize
        weights = dataset.normalized_weights
        # 0.9 + 0.5 = 1.4, so a = 0.9/1.4, b = 0.5/1.4
        assert abs(weights[0].item() - 0.9 / 1.4) < 1e-6
        assert abs(weights[1].item() - 0.5 / 1.4) < 1e-6

    def test_get_current_weights(self):
        """Test getting current weights as dictionary."""
        mixtures = [
            DatasetMixture(name="a", weight=0.3, path="test/a"),
            DatasetMixture(name="b", weight=0.7, path="test/b"),
        ]
        dataset = MixedDataset(mixtures=mixtures, streaming=False)

        weights = dataset.get_current_weights()

        assert "a" in weights
        assert "b" in weights
        assert abs(weights["a"] - 0.3) < 1e-6
        assert abs(weights["b"] - 0.7) < 1e-6

    def test_distributed_sharding_params(self):
        """Test that distributed sharding parameters are stored."""
        mixtures = [DatasetMixture(name="a", weight=1.0, path="test/a")]
        dataset = MixedDataset(
            mixtures=mixtures, rank=2, world_size=4, shuffle_buffer_size=50000
        )

        assert dataset.rank == 2
        assert dataset.world_size == 4
        assert dataset.shuffle_buffer_size == 50000

    def test_get_sample_with_list_columns(self):
        """Test _get_sample with list of columns for concatenation."""
        mixtures = [
            DatasetMixture(
                name="synth",
                weight=1.0,
                path="test/synth",
                text_column=["query", "reasoning", "answer"],
                text_separator=" | ",
            )
        ]
        dataset = MixedDataset(mixtures=mixtures, streaming=False)

        # Mock the datasets and iterators
        mock_sample = {
            "query": "What is 2+2?",
            "reasoning": "Adding two numbers",
            "answer": "4",
        }
        dataset._datasets = [[mock_sample]]
        dataset._iterators = [iter([mock_sample])]

        result = dataset._get_sample(0)

        assert result["text"] == "What is 2+2? | Adding two numbers | 4"
        assert result["source"] == "synth"

    def test_get_sample_with_partial_columns(self):
        """Test _get_sample when some columns are missing."""
        mixtures = [
            DatasetMixture(
                name="synth",
                weight=1.0,
                path="test/synth",
                text_column=["query", "reasoning", "answer"],
                text_separator="\n\n",
            )
        ]
        dataset = MixedDataset(mixtures=mixtures, streaming=False)

        # Sample with missing 'reasoning' column
        mock_sample = {
            "query": "What is 2+2?",
            "answer": "4",
        }
        dataset._datasets = [[mock_sample]]
        dataset._iterators = [iter([mock_sample])]

        result = dataset._get_sample(0)

        # Should only include non-empty columns
        assert result["text"] == "What is 2+2?\n\n4"
        assert result["source"] == "synth"


class TestPackedDataset:
    """Tests for PackedDataset."""

    def test_batch_tokenize(self):
        """Test batched tokenization method."""
        tokenizer = MockTokenizer()

        # Create a simple mock source dataset
        source = iter([{"text": "hello world"}, {"text": "foo bar baz"}])

        dataset = PackedDataset(
            dataset=source,
            tokenizer=tokenizer,
            max_length=10,
            tokenize_batch_size=2,
        )

        result = dataset._batch_tokenize(["hello world", "foo bar baz"])

        assert len(result) == 2
        assert len(result[0]) == 2  # "hello world" -> 2 tokens
        assert len(result[1]) == 3  # "foo bar baz" -> 3 tokens

    def test_batch_tokenize_empty(self):
        """Test batched tokenization with empty input."""
        tokenizer = MockTokenizer()
        source = iter([])

        dataset = PackedDataset(
            dataset=source,
            tokenizer=tokenizer,
            max_length=10,
        )

        result = dataset._batch_tokenize([])
        assert result == []

    def test_packed_output_format(self):
        """Test that packed dataset yields correct output format."""
        tokenizer = MockTokenizer()

        # Create source with enough text to generate at least one chunk
        texts = [{"text": " ".join(["word"] * 20)}]  # 20 tokens
        source = iter(texts)

        dataset = PackedDataset(
            dataset=source,
            tokenizer=tokenizer,
            max_length=10,
            tokenize_batch_size=1,
        )

        # Get first output
        output = next(iter(dataset))

        assert "input_ids" in output
        assert "attention_mask" in output
        assert output["input_ids"].shape == (10,)
        assert output["attention_mask"].shape == (10,)
        assert output["input_ids"].dtype == torch.long
        assert output["attention_mask"].dtype == torch.long

    def test_tokenize_batch_size_config(self):
        """Test configurable tokenize batch size."""
        tokenizer = MockTokenizer()
        source = iter([])

        dataset = PackedDataset(
            dataset=source,
            tokenizer=tokenizer,
            max_length=2048,
            tokenize_batch_size=512,
        )

        assert dataset.tokenize_batch_size == 512


class TestWorkerInitFn:
    """Tests for worker initialization function."""

    def test_worker_init_fn_updates_seed(self):
        """Test that worker_init_fn updates dataset seed."""
        # Create mock dataset with seed attribute
        # Use spec to prevent auto-creation of 'dataset' attr (avoids infinite loop)
        mock_dataset = Mock(spec=["seed", "_datasets", "_iterators"])
        mock_dataset.seed = 42
        mock_dataset._datasets = None
        mock_dataset._iterators = None

        # Create mock worker info
        mock_worker_info = Mock()
        mock_worker_info.dataset = mock_dataset

        with patch("torch.utils.data.get_worker_info", return_value=mock_worker_info):
            _worker_init_fn(worker_id=3)

        # Seed should be offset by worker_id
        assert mock_dataset.seed == 45  # 42 + 3

    def test_worker_init_fn_resets_dataset(self):
        """Test that worker_init_fn resets dataset for reload."""
        # Use spec to prevent auto-creation of 'dataset' attr (avoids infinite loop)
        mock_dataset = Mock(spec=["seed", "_datasets", "_iterators"])
        mock_dataset.seed = 42
        mock_dataset._datasets = ["some_data"]
        mock_dataset._iterators = ["some_iter"]

        mock_worker_info = Mock()
        mock_worker_info.dataset = mock_dataset

        with patch("torch.utils.data.get_worker_info", return_value=mock_worker_info):
            _worker_init_fn(worker_id=1)

        # _datasets and _iterators should be reset to None
        assert mock_dataset._datasets is None
        assert mock_dataset._iterators is None

    def test_worker_init_fn_nested_dataset(self):
        """Test worker_init_fn with nested datasets (PackedDataset wrapping MixedDataset)."""
        # Inner dataset (MixedDataset)
        inner_dataset = Mock()
        inner_dataset.seed = 100
        inner_dataset._datasets = ["data"]
        inner_dataset._iterators = ["iter"]

        # Remove 'dataset' attribute from inner to stop recursion
        del inner_dataset.dataset

        # Outer dataset (PackedDataset)
        outer_dataset = Mock()
        outer_dataset.dataset = inner_dataset

        mock_worker_info = Mock()
        mock_worker_info.dataset = outer_dataset

        with patch("torch.utils.data.get_worker_info", return_value=mock_worker_info):
            _worker_init_fn(worker_id=5)

        # Inner dataset seed should be updated
        assert inner_dataset.seed == 105  # 100 + 5


class TestCreateMixedDataset:
    """Tests for create_mixed_dataset factory function."""

    def test_creates_dataset_from_config(self):
        """Test dataset creation from config dict."""
        config = {
            "mixtures": [
                {"name": "web", "weight": 0.7, "path": "test/web"},
                {"name": "code", "weight": 0.3, "path": "test/code"},
            ],
            "seed": 123,
            "streaming": False,
            "max_length": 1024,
        }
        tokenizer = MockTokenizer()

        dataset = create_mixed_dataset(
            config=config,
            tokenizer=tokenizer,
            packing=False,  # Don't wrap in PackedDataset
        )

        assert isinstance(dataset, MixedDataset)
        assert len(dataset.mixtures) == 2
        assert dataset.seed == 123

    def test_creates_packed_dataset(self):
        """Test dataset creation with packing enabled."""
        config = {
            "mixtures": [{"name": "test", "weight": 1.0, "path": "test/data"}],
            "max_length": 512,
            "tokenize_batch_size": 128,
        }
        tokenizer = MockTokenizer()

        dataset = create_mixed_dataset(
            config=config,
            tokenizer=tokenizer,
            packing=True,
        )

        assert isinstance(dataset, PackedDataset)
        assert dataset.max_length == 512
        assert dataset.tokenize_batch_size == 128

    def test_passes_distributed_params(self):
        """Test that distributed params are passed through."""
        config = {
            "mixtures": [{"name": "test", "weight": 1.0, "path": "test/data"}],
            "shuffle_buffer_size": 25000,
        }
        tokenizer = MockTokenizer()

        dataset = create_mixed_dataset(
            config=config,
            tokenizer=tokenizer,
            packing=False,
            rank=3,
            world_size=8,
        )

        assert dataset.rank == 3
        assert dataset.world_size == 8
        assert dataset.shuffle_buffer_size == 25000


class TestCreateMixedDataloader:
    """Tests for create_mixed_dataloader factory function."""

    def test_creates_dataloader(self):
        """Test DataLoader creation with config."""
        config = {
            "mixtures": [{"name": "test", "weight": 1.0, "path": "test/data"}],
            "streaming": False,
        }
        tokenizer = MockTokenizer()

        dataloader = create_mixed_dataloader(
            config=config,
            tokenizer=tokenizer,
            batch_size=8,
            num_workers=0,  # For testing
        )

        from torch.utils.data import DataLoader

        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8

    def test_dataloader_with_workers(self):
        """Test DataLoader with worker configuration."""
        config = {
            "mixtures": [{"name": "test", "weight": 1.0, "path": "test/data"}],
        }
        tokenizer = MockTokenizer()

        dataloader = create_mixed_dataloader(
            config=config,
            tokenizer=tokenizer,
            batch_size=16,
            num_workers=4,
            prefetch_factor=3,
            pin_memory=True,
        )

        assert dataloader.num_workers == 4
        assert dataloader.prefetch_factor == 3
        assert dataloader.pin_memory is True

    def test_dataloader_zero_workers(self):
        """Test DataLoader with zero workers (main process loading)."""
        config = {
            "mixtures": [{"name": "test", "weight": 1.0, "path": "test/data"}],
        }
        tokenizer = MockTokenizer()

        dataloader = create_mixed_dataloader(
            config=config,
            tokenizer=tokenizer,
            batch_size=8,
            num_workers=0,
            prefetch_factor=2,  # Should be ignored when num_workers=0
        )

        assert dataloader.num_workers == 0
        # prefetch_factor should be None when num_workers=0
        assert dataloader.prefetch_factor is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
