"""Pytest configuration and shared fixtures for CheaperTraining tests."""

import os
from typing import Generator
from unittest.mock import MagicMock

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: Quick smoke tests")
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring GPU or external services"
    )
    config.addinivalue_line("markers", "slow: Slow tests that take >30s")
    config.addinivalue_line("markers", "gpu: Tests requiring CUDA GPU")


# =============================================================================
# Device Fixtures
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device for testing."""
    return torch.device("cpu")


# =============================================================================
# Mock Fixtures (for CPU-only testing)
# =============================================================================


@pytest.fixture
def mock_cuda(mocker) -> None:
    """Mock CUDA availability for CPU-only testing."""
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.cuda.device_count", return_value=0)


@pytest.fixture
def mock_cuda_available(mocker) -> None:
    """Mock CUDA as available (for testing GPU code paths on CPU)."""
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch("torch.cuda.current_device", return_value=0)
    mocker.patch("torch.cuda.get_device_name", return_value="Mock GPU")


@pytest.fixture
def mock_hf_model(mocker) -> MagicMock:
    """Mock HuggingFace model loading."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.config.hidden_size = 768
    mock_model.config.num_hidden_layers = 12
    mock = mocker.patch("transformers.AutoModel.from_pretrained", return_value=mock_model)
    return mock


@pytest.fixture
def mock_hf_tokenizer(mocker) -> MagicMock:
    """Mock HuggingFace tokenizer loading."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    )
    return mock


@pytest.fixture
def mock_wandb(mocker) -> MagicMock:
    """Mock W&B logging."""
    mock_init = mocker.patch("wandb.init", return_value=MagicMock())
    mocker.patch("wandb.log")
    mocker.patch("wandb.finish")
    return mock_init


@pytest.fixture
def mock_datasets(mocker) -> MagicMock:
    """Mock datasets library."""
    mock_dataset = MagicMock()
    mock_dataset.__iter__ = lambda self: iter([{"text": "sample text"}])
    mock_dataset.__len__ = lambda self: 1
    mock = mocker.patch("datasets.load_dataset", return_value=mock_dataset)
    return mock


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Generate sample input tensor for testing."""
    return torch.randn(2, 10, 64)


@pytest.fixture
def sample_batch() -> dict:
    """Generate sample training batch."""
    batch_size, seq_len = 2, 128
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, 1000, (batch_size, seq_len)),
    }


@pytest.fixture
def small_model_config() -> dict:
    """Config for a small test model."""
    return {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "vocab_size": 1000,
    }


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def checkpoint_dir(tmp_path) -> Generator:
    """Create a temporary checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    yield ckpt_dir


@pytest.fixture
def output_dir(tmp_path) -> Generator:
    """Create a temporary output directory."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    yield out_dir


# =============================================================================
# Skip Conditions
# =============================================================================


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")


@pytest.fixture
def skip_if_no_wandb():
    """Skip test if WANDB_API_KEY is not set."""
    if not os.environ.get("WANDB_API_KEY"):
        pytest.skip("WANDB_API_KEY not set")


# =============================================================================
# Auto-skip GPU tests on CPU-only machines
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with 'gpu' if CUDA is not available."""
    if torch.cuda.is_available():
        return

    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
