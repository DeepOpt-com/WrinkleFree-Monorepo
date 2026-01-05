"""Unit tests for influence calculation components.

Tests DataInfCalculator, DiscriminativeGradientExtractor, and MixtureWeightCalculator.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wf_data.influence.config import InfluenceConfig, InfluenceTarget
from wf_data.influence.gradient import DiscriminativeGradientExtractor
from wf_data.influence.datainf import DataInfCalculator, create_influence_calculator
from wf_data.influence.mixture_calculator import MixtureWeightCalculator


class SimpleLanguageModel(nn.Module):
    """Minimal LM for testing influence functions."""

    def __init__(self, vocab_size: int = 100, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        x = self.embed_tokens(input_ids)
        x = self.linear(x)
        logits = self.lm_head(x)
        if return_dict:
            return {"logits": logits}
        return logits


class SharedWeightModel(nn.Module):
    """LM with shared embedding and output weights for testing."""

    def __init__(self, vocab_size: int = 100, embed_dim: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Share weights
        self.lm_head.weight = self.embed_tokens.weight
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        x = self.embed_tokens(input_ids)
        logits = self.lm_head(x)
        if return_dict:
            return {"logits": logits}
        return logits


class TestInfluenceConfig:
    """Tests for InfluenceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InfluenceConfig()
        assert config.lambda_reg > 0
        assert config.target_layers == InfluenceTarget.EMBEDDING_AND_OUTPUT
        assert config.max_grad_norm > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = InfluenceConfig(
            lambda_reg=1e-3,
            target_layers=InfluenceTarget.OUTPUT_ONLY,
            max_grad_norm=5.0,
        )
        assert config.lambda_reg == 1e-3
        assert config.target_layers == InfluenceTarget.OUTPUT_ONLY
        assert config.max_grad_norm == 5.0

    def test_all_target_layers(self):
        """Test all target layer options."""
        for target in InfluenceTarget:
            config = InfluenceConfig(target_layers=target)
            assert config.target_layers == target


class TestDiscriminativeGradientExtractor:
    """Tests for DiscriminativeGradientExtractor."""

    @pytest.fixture
    def model(self):
        return SimpleLanguageModel(vocab_size=100, embed_dim=32, hidden_dim=64)

    @pytest.fixture
    def shared_model(self):
        return SharedWeightModel(vocab_size=100, embed_dim=32)

    def test_init_with_model(self, model):
        """Test initialization with valid model."""
        extractor = DiscriminativeGradientExtractor(model)
        assert extractor.model is model
        assert extractor.embed_tokens is model.embed_tokens
        assert extractor.lm_head is model.lm_head

    def test_init_fails_without_embed_tokens(self):
        """Test initialization fails without embed_tokens."""
        model = nn.Linear(10, 10)
        with pytest.raises(ValueError, match="embed_tokens"):
            DiscriminativeGradientExtractor(model)

    def test_weight_sharing_detection(self, model, shared_model):
        """Test weight sharing detection."""
        extractor = DiscriminativeGradientExtractor(model)
        assert extractor.weight_sharing is False

        shared_extractor = DiscriminativeGradientExtractor(shared_model)
        assert shared_extractor.weight_sharing is True

    def test_get_target_parameters_embedding_only(self, model):
        """Test getting parameters for embedding only."""
        config = InfluenceConfig(target_layers=InfluenceTarget.EMBEDDING_ONLY)
        extractor = DiscriminativeGradientExtractor(model, config)
        params = extractor.get_target_parameters()

        assert "embed_tokens" in params
        assert "lm_head" not in params

    def test_get_target_parameters_output_only(self, model):
        """Test getting parameters for output only."""
        config = InfluenceConfig(target_layers=InfluenceTarget.OUTPUT_ONLY)
        extractor = DiscriminativeGradientExtractor(model, config)
        params = extractor.get_target_parameters()

        assert "lm_head" in params
        assert "embed_tokens" not in params

    def test_get_target_parameters_both(self, model):
        """Test getting parameters for both embedding and output."""
        config = InfluenceConfig(target_layers=InfluenceTarget.EMBEDDING_AND_OUTPUT)
        extractor = DiscriminativeGradientExtractor(model, config)
        params = extractor.get_target_parameters()

        assert "embed_tokens" in params
        assert "lm_head" in params

    def test_get_target_parameters_shared_weights(self, shared_model):
        """Test that shared weights don't get double-counted."""
        config = InfluenceConfig(target_layers=InfluenceTarget.EMBEDDING_AND_OUTPUT)
        extractor = DiscriminativeGradientExtractor(shared_model, config)
        params = extractor.get_target_parameters()

        # Should only have embed_tokens since lm_head shares weights
        assert "embed_tokens" in params
        assert "lm_head" not in params

    def test_get_gradient_dimension(self, model):
        """Test gradient dimension calculation."""
        config = InfluenceConfig(target_layers=InfluenceTarget.EMBEDDING_ONLY)
        extractor = DiscriminativeGradientExtractor(model, config)
        dim = extractor.get_gradient_dimension()

        # Embedding: vocab_size * embed_dim = 100 * 32 = 3200
        expected = 100 * 32
        assert dim == expected

    def test_compute_per_sample_gradient(self, model):
        """Test per-sample gradient computation."""
        extractor = DiscriminativeGradientExtractor(model)
        input_ids = torch.randint(0, 100, (8,))

        gradients = extractor.compute_per_sample_gradient(input_ids)

        assert "embed_tokens" in gradients
        assert gradients["embed_tokens"].shape == model.embed_tokens.weight.shape

    def test_compute_per_sample_gradient_with_labels(self, model):
        """Test gradient computation with labels."""
        extractor = DiscriminativeGradientExtractor(model)
        input_ids = torch.randint(0, 100, (8,))
        labels = torch.randint(0, 100, (8,))

        gradients = extractor.compute_per_sample_gradient(input_ids, labels=labels)
        assert "embed_tokens" in gradients

    def test_compute_per_sample_gradient_with_mask(self, model):
        """Test gradient computation with attention mask."""
        extractor = DiscriminativeGradientExtractor(model)
        input_ids = torch.randint(0, 100, (8,))
        attention_mask = torch.ones(8)

        gradients = extractor.compute_per_sample_gradient(
            input_ids, attention_mask=attention_mask
        )
        assert "embed_tokens" in gradients

    def test_flatten_gradients(self, model):
        """Test gradient flattening."""
        extractor = DiscriminativeGradientExtractor(model)
        input_ids = torch.randint(0, 100, (8,))

        gradients = extractor.compute_per_sample_gradient(input_ids)
        flat = extractor.flatten_gradients(gradients)

        # Should be 1D
        assert flat.dim() == 1
        assert flat.numel() == extractor.get_gradient_dimension()

    def test_compute_batch_gradients(self, model):
        """Test batch gradient computation."""
        extractor = DiscriminativeGradientExtractor(model)
        batch = {
            "input_ids": torch.randint(0, 100, (4, 8)),
            "attention_mask": torch.ones(4, 8),
        }

        grads = extractor.compute_batch_gradients(batch)

        assert grads.shape[0] == 4  # batch size
        assert grads.shape[1] == extractor.get_gradient_dimension()

    def test_compute_aggregated_gradient(self, model):
        """Test aggregated gradient computation."""
        extractor = DiscriminativeGradientExtractor(model)
        batch = {
            "input_ids": torch.randint(0, 100, (4, 8)),
        }

        agg_grad = extractor.compute_aggregated_gradient(batch)

        assert agg_grad.dim() == 1
        assert agg_grad.numel() == extractor.get_gradient_dimension()


class TestDataInfCalculator:
    """Tests for DataInfCalculator."""

    @pytest.fixture
    def model(self):
        return SimpleLanguageModel(vocab_size=100, embed_dim=32, hidden_dim=64)

    @pytest.fixture
    def extractor(self, model):
        return DiscriminativeGradientExtractor(model)

    @pytest.fixture
    def calculator(self, extractor):
        return DataInfCalculator(extractor)

    @pytest.fixture
    def probe_dataloader(self):
        """Create a small probe dataloader."""
        input_ids = torch.randint(0, 100, (8, 16))
        attention_mask = torch.ones(8, 16)
        dataset = TensorDataset(input_ids, attention_mask)

        def collate_fn(batch):
            ids = torch.stack([b[0] for b in batch])
            masks = torch.stack([b[1] for b in batch])
            return {"input_ids": ids, "attention_mask": masks}

        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    def test_init(self, calculator, extractor):
        """Test calculator initialization."""
        assert calculator.gradient_extractor is extractor
        assert calculator.config is not None
        assert not calculator.is_cached

    def test_cache_probe_gradients(self, calculator, probe_dataloader):
        """Test caching probe gradients."""
        probe_grads, probe_norms = calculator.cache_probe_gradients(
            probe_dataloader, show_progress=False
        )

        assert calculator.is_cached
        assert probe_grads.shape[0] == 8  # 8 probe samples
        assert probe_norms.shape[0] == 8
        assert calculator._avg_probe_gradient is not None

    def test_clear_cache(self, calculator, probe_dataloader):
        """Test clearing cache."""
        calculator.cache_probe_gradients(probe_dataloader, show_progress=False)
        assert calculator.is_cached

        calculator.clear_cache()
        assert not calculator.is_cached
        assert calculator._probe_gradients is None

    def test_compute_influence_requires_cache(self, calculator):
        """Test that compute_influence requires cached gradients."""
        train_grad = torch.randn(calculator.gradient_extractor.get_gradient_dimension())

        with pytest.raises(RuntimeError, match="Probe gradients not cached"):
            calculator.compute_influence(train_grad)

    def test_compute_influence(self, calculator, probe_dataloader):
        """Test influence computation."""
        calculator.cache_probe_gradients(probe_dataloader, show_progress=False)
        train_grad = torch.randn(calculator.gradient_extractor.get_gradient_dimension())

        influence = calculator.compute_influence(train_grad)

        assert influence.shape[0] == 8  # One score per probe sample
        # Influence can be positive or negative

    def test_compute_influence_aggregated(self, calculator, probe_dataloader):
        """Test aggregated influence computation."""
        calculator.cache_probe_gradients(probe_dataloader, show_progress=False)
        train_grad = torch.randn(calculator.gradient_extractor.get_gradient_dimension())

        agg_influence = calculator.compute_influence_aggregated(train_grad)

        assert isinstance(agg_influence, float)

    def test_compute_batch_influence(self, calculator, model, probe_dataloader):
        """Test batch influence computation."""
        calculator.cache_probe_gradients(probe_dataloader, show_progress=False)

        train_batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16),
        }

        influences = calculator.compute_batch_influence(train_batch)

        assert influences.shape == (4, 8)  # (batch_size, n_probe)

    def test_compute_batch_influence_aggregated(self, calculator, model, probe_dataloader):
        """Test batch aggregated influence."""
        calculator.cache_probe_gradients(probe_dataloader, show_progress=False)

        train_batch = {
            "input_ids": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16),
        }

        influences = calculator.compute_batch_influence_aggregated(train_batch)

        assert influences.shape == (4,)

    def test_aggregate_influence_methods(self, calculator, probe_dataloader):
        """Test different aggregation methods."""
        calculator.cache_probe_gradients(probe_dataloader, show_progress=False)
        influence_matrix = torch.randn(4, 8)

        for method in ["mean", "sum", "max", "min"]:
            result = calculator.aggregate_influence(influence_matrix, method=method)
            assert result.shape == (4,)

    def test_aggregate_influence_invalid_method(self, calculator):
        """Test invalid aggregation method."""
        influence_matrix = torch.randn(4, 8)

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            calculator.aggregate_influence(influence_matrix, method="invalid")


class TestCreateInfluenceCalculator:
    """Tests for factory function."""

    def test_create_calculator(self):
        """Test factory function."""
        model = SimpleLanguageModel()
        calculator = create_influence_calculator(model)

        assert isinstance(calculator, DataInfCalculator)
        assert calculator.gradient_extractor.model is model

    def test_create_calculator_with_config(self):
        """Test factory with custom config."""
        model = SimpleLanguageModel()
        config = InfluenceConfig(lambda_reg=0.01)
        calculator = create_influence_calculator(model, config)

        assert calculator.config.lambda_reg == 0.01


class TestMixtureWeightCalculator:
    """Tests for MixtureWeightCalculator."""

    @pytest.fixture
    def model(self):
        return SimpleLanguageModel(vocab_size=100, embed_dim=32, hidden_dim=64)

    @pytest.fixture
    def probe_dataloader(self):
        """Create a small probe dataloader."""
        input_ids = torch.randint(0, 100, (8, 16))
        attention_mask = torch.ones(8, 16)
        dataset = TensorDataset(input_ids, attention_mask)

        def collate_fn(batch):
            ids = torch.stack([b[0] for b in batch])
            masks = torch.stack([b[1] for b in batch])
            return {"input_ids": ids, "attention_mask": masks}

        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    def test_init(self, model, probe_dataloader):
        """Test MixtureWeightCalculator initialization."""
        calc = MixtureWeightCalculator(model, probe_dataloader)
        assert calc.model is model
        assert calc.probe_dataloader is probe_dataloader
        assert calc.influence_calculator is not None

    def test_cache_probe_gradients(self, model, probe_dataloader):
        """Test caching probe gradients."""
        calc = MixtureWeightCalculator(model, probe_dataloader)
        assert not calc._probe_cached

        calc.cache_probe_gradients(show_progress=False)
        assert calc._probe_cached

    def test_refresh_probe_cache(self, model, probe_dataloader):
        """Test refreshing probe cache."""
        calc = MixtureWeightCalculator(model, probe_dataloader)
        calc.cache_probe_gradients(show_progress=False)
        assert calc._probe_cached

        # Refresh should recache
        calc.refresh_probe_cache(show_progress=False)
        assert calc._probe_cached

    def test_compute_dataset_influence(self, model, probe_dataloader):
        """Test computing influence for a data source."""
        calc = MixtureWeightCalculator(model, probe_dataloader)

        # Create a small source dataloader
        source_input_ids = torch.randint(0, 100, (4, 16))
        source_mask = torch.ones(4, 16)
        source_dataset = TensorDataset(source_input_ids, source_mask)

        def collate_fn(batch):
            ids = torch.stack([b[0] for b in batch])
            masks = torch.stack([b[1] for b in batch])
            return {"input_ids": ids, "attention_mask": masks}

        source_loader = DataLoader(source_dataset, batch_size=2, collate_fn=collate_fn)

        influence = calc.compute_dataset_influence(source_loader, show_progress=False)

        # Should return a scalar
        assert isinstance(influence, float)

    def test_compute_mixture_weights(self, model, probe_dataloader):
        """Test computing optimal weights for multiple sources."""
        calc = MixtureWeightCalculator(model, probe_dataloader)

        # Create source dataloaders
        def make_source_loader(n_samples):
            input_ids = torch.randint(0, 100, (n_samples, 16))
            mask = torch.ones(n_samples, 16)
            dataset = TensorDataset(input_ids, mask)

            def collate_fn(batch):
                ids = torch.stack([b[0] for b in batch])
                masks = torch.stack([b[1] for b in batch])
                return {"input_ids": ids, "attention_mask": masks}

            return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        source_loaders = {
            "source_a": make_source_loader(4),
            "source_b": make_source_loader(4),
        }

        weights = calc.compute_mixture_weights(source_loaders, show_progress=False)

        assert "source_a" in weights
        assert "source_b" in weights
        # Weights should be normalized
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_influences_to_weights(self, model, probe_dataloader):
        """Test converting influences to weights."""
        calc = MixtureWeightCalculator(model, probe_dataloader)

        # Test with positive influences
        influences = {"a": 0.5, "b": 1.0, "c": 0.25}
        weights = calc._influences_to_weights(influences)

        assert sum(weights.values()) == pytest.approx(1.0, abs=0.01)
        # Higher influence should get higher weight
        assert weights["b"] > weights["a"]
        assert weights["a"] > weights["c"]

    def test_influences_to_weights_negative(self, model, probe_dataloader):
        """Test that negative influences are clamped."""
        calc = MixtureWeightCalculator(model, probe_dataloader)

        influences = {"a": -0.5, "b": 0.5}
        weights = calc._influences_to_weights(influences)

        # Both should have positive weights
        assert weights["a"] > 0
        assert weights["b"] > 0
        # b should have higher weight
        assert weights["b"] > weights["a"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
