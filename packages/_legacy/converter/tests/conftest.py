"""Pytest configuration and shared fixtures for WrinkleFree-DLM-Converter tests."""

import os
from typing import Generator
from unittest.mock import MagicMock

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: Quick smoke tests")
    config.addinivalue_line("markers", "integration: Integration tests requiring Modal")
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
def mock_hf_model(mocker) -> MagicMock:
    """Mock HuggingFace model loading."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.config.hidden_size = 768
    mock_model.config.num_hidden_layers = 12
    mock_model.config.vocab_size = 32000
    mock = mocker.patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model
    )
    return mock


@pytest.fixture
def mock_hf_tokenizer(mocker) -> MagicMock:
    """Mock HuggingFace tokenizer loading."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.vocab_size = 32000
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
def mock_modal(mocker) -> MagicMock:
    """Mock Modal cloud functions."""
    mock_app = mocker.patch("modal.App", return_value=MagicMock())
    mocker.patch("modal.Image")
    mocker.patch("modal.Volume")
    return mock_app


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
        "bd_size": 32,  # DLM block size
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


@pytest.fixture
def skip_if_no_modal():
    """Skip test if Modal is not configured."""
    try:
        import modal

        # Check if Modal is authenticated
        modal.config.Config()
    except Exception:
        pytest.skip("Modal not configured")


# =============================================================================
# Model Fixtures for Inference Tests
# =============================================================================

# Module-level cache for model and tokenizer (shared across tests)
_model_cache = {}


@pytest.fixture(scope="module")
def small_model_and_tokenizer():
    """Load small model and tokenizer together to ensure compatibility.

    The model's embeddings are resized to include the mask token.
    Uses module scope to avoid reloading for each test.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for model loading")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "HuggingFaceTB/SmolLM2-135M"
    mask_token = "|<MASK>|"

    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add mask token if not present
        if mask_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})

        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        # Resize embeddings to include mask token
        model.resize_token_embeddings(len(tokenizer))
        model.eval()

        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Could not load model {model_name}: {e}")


@pytest.fixture(scope="module")
def small_model(small_model_and_tokenizer):
    """Get just the model from the combined fixture."""
    model, _ = small_model_and_tokenizer
    return model


@pytest.fixture(scope="module")
def small_tokenizer(small_model_and_tokenizer):
    """Get just the tokenizer from the combined fixture."""
    _, tokenizer = small_model_and_tokenizer
    return tokenizer


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
