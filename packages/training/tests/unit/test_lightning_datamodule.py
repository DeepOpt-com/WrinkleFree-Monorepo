"""Unit tests for WrinkleFreeDataModule."""

import pytest
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader

from wf_train.lightning import WrinkleFreeDataModule

# The import happens inside setup() as: from wf_train.data import create_pretraining_dataloader
# So we patch where it's defined (wrinklefree.data) not where it's used
MOCK_PATH = "wf_train.data.create_pretraining_dataloader"


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader."""
    dl = MagicMock(spec=DataLoader)
    dl.__iter__ = MagicMock(return_value=iter([{"input_ids": [1, 2, 3]}]))
    return dl


class TestWrinkleFreeDataModuleInit:
    """Test DataModule initialization."""

    def test_init_defaults(self, mock_tokenizer):
        """Test initialization with default parameters."""
        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)

        assert dm.tokenizer == mock_tokenizer
        assert dm.batch_size == 32
        assert dm.max_length == 2048
        assert dm.config_name == "default"
        assert dm.with_probes is False
        assert dm.world_size == 1
        assert dm.rank == 0
        assert dm.packed is True
        assert dm.num_workers == 4
        assert dm.val_config_name is None
        assert dm.val_batch_size == 32  # defaults to batch_size

    def test_init_custom_params(self, mock_tokenizer):
        """Test initialization with custom parameters."""
        dm = WrinkleFreeDataModule(
            tokenizer=mock_tokenizer,
            batch_size=64,
            max_length=4096,
            config_name="mixed_pretrain",
            with_probes=True,
            world_size=4,
            rank=2,
            packed=False,
            num_workers=8,
            val_config_name="fineweb",
            val_batch_size=16,
        )

        assert dm.batch_size == 64
        assert dm.max_length == 4096
        assert dm.config_name == "mixed_pretrain"
        assert dm.with_probes is True
        assert dm.world_size == 4
        assert dm.rank == 2
        assert dm.packed is False
        assert dm.num_workers == 8
        assert dm.val_config_name == "fineweb"
        assert dm.val_batch_size == 16

    def test_init_val_batch_size_defaults_to_batch_size(self, mock_tokenizer):
        """Test that val_batch_size defaults to batch_size if not specified."""
        dm = WrinkleFreeDataModule(
            tokenizer=mock_tokenizer,
            batch_size=128,
            val_config_name="fineweb",
        )

        assert dm.val_batch_size == 128

    def test_init_dataloaders_are_none(self, mock_tokenizer):
        """Test that dataloaders are None before setup."""
        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)

        assert dm.train_dataloader_instance is None
        assert dm.val_dataloader_instance is None
        assert dm.mixed_dataset is None
        assert dm.probe_dataloaders is None


class TestWrinkleFreeDataModuleSetup:
    """Test DataModule setup."""

    @patch(MOCK_PATH)
    def test_setup_creates_train_dataloader(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that setup creates the training dataloader."""
        mock_dataset = MagicMock()
        mock_probes = {"probe1": MagicMock()}
        mock_create_dl.return_value = (mock_dataloader, mock_dataset, mock_probes)

        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer, config_name="default")
        dm.setup(stage="fit")

        mock_create_dl.assert_called_once_with(
            tokenizer=mock_tokenizer,
            batch_size=32,
            max_length=2048,
            config_name="default",
            with_probes=False,
            world_size=1,
            rank=0,
            packed=True,
        )
        assert dm.train_dataloader_instance == mock_dataloader
        assert dm.mixed_dataset == mock_dataset
        assert dm.probe_dataloaders == mock_probes

    @patch(MOCK_PATH)
    def test_setup_with_none_stage(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that setup with stage=None also creates dataloaders."""
        mock_create_dl.return_value = (mock_dataloader, None, None)

        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)
        dm.setup(stage=None)

        assert mock_create_dl.called
        assert dm.train_dataloader_instance == mock_dataloader

    @patch(MOCK_PATH)
    def test_setup_creates_val_dataloader_when_configured(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that setup creates validation dataloader when val_config_name is set."""
        mock_val_dl = MagicMock(spec=DataLoader)
        mock_create_dl.side_effect = [
            (mock_dataloader, None, None),  # train
            (mock_val_dl, None, None),  # val
        ]

        dm = WrinkleFreeDataModule(
            tokenizer=mock_tokenizer,
            val_config_name="fineweb",
            val_batch_size=16,
        )
        dm.setup(stage="fit")

        assert mock_create_dl.call_count == 2
        # Check val dataloader was created with correct params
        val_call = mock_create_dl.call_args_list[1]
        assert val_call.kwargs["batch_size"] == 16
        assert val_call.kwargs["config_name"] == "fineweb"
        assert val_call.kwargs["with_probes"] is False

        assert dm.val_dataloader_instance == mock_val_dl

    @patch(MOCK_PATH)
    def test_setup_does_not_run_for_test_stage(
        self, mock_create_dl, mock_tokenizer
    ):
        """Test that setup doesn't create dataloaders for test stage."""
        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)
        dm.setup(stage="test")

        mock_create_dl.assert_not_called()


class TestWrinkleFreeDataModuleDataloaders:
    """Test DataModule dataloader access."""

    @patch(MOCK_PATH)
    def test_train_dataloader_returns_dataloader(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that train_dataloader returns the dataloader after setup."""
        mock_create_dl.return_value = (mock_dataloader, None, None)

        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)
        dm.setup(stage="fit")

        assert dm.train_dataloader() == mock_dataloader

    def test_train_dataloader_raises_without_setup(self, mock_tokenizer):
        """Test that train_dataloader raises error if setup not called."""
        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)

        with pytest.raises(RuntimeError, match="Call setup()"):
            dm.train_dataloader()

    def test_val_dataloader_returns_empty_list_when_no_val_config(
        self, mock_tokenizer
    ):
        """Test that val_dataloader returns empty list when no val config."""
        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)

        result = dm.val_dataloader()

        assert result == []

    @patch(MOCK_PATH)
    def test_val_dataloader_returns_dataloader_when_configured(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that val_dataloader returns dataloader when configured."""
        mock_val_dl = MagicMock(spec=DataLoader)
        mock_create_dl.side_effect = [
            (mock_dataloader, None, None),
            (mock_val_dl, None, None),
        ]

        dm = WrinkleFreeDataModule(
            tokenizer=mock_tokenizer,
            val_config_name="fineweb",
        )
        dm.setup(stage="fit")

        assert dm.val_dataloader() == mock_val_dl


class TestWrinkleFreeDataModuleAccessors:
    """Test DataModule accessor methods."""

    @patch(MOCK_PATH)
    def test_get_mixed_dataset(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test get_mixed_dataset returns the dataset."""
        mock_dataset = MagicMock()
        mock_create_dl.return_value = (mock_dataloader, mock_dataset, None)

        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)
        dm.setup(stage="fit")

        assert dm.get_mixed_dataset() == mock_dataset

    @patch(MOCK_PATH)
    def test_get_probe_dataloaders(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test get_probe_dataloaders returns probe loaders."""
        mock_probes = {"web_edu": MagicMock(), "code": MagicMock()}
        mock_create_dl.return_value = (mock_dataloader, None, mock_probes)

        dm = WrinkleFreeDataModule(
            tokenizer=mock_tokenizer,
            with_probes=True,
        )
        dm.setup(stage="fit")

        assert dm.get_probe_dataloaders() == mock_probes

    def test_get_mixed_dataset_returns_none_before_setup(self, mock_tokenizer):
        """Test that get_mixed_dataset returns None before setup."""
        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)

        assert dm.get_mixed_dataset() is None

    def test_get_probe_dataloaders_returns_none_before_setup(self, mock_tokenizer):
        """Test that get_probe_dataloaders returns None before setup."""
        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer)

        assert dm.get_probe_dataloaders() is None


class TestWrinkleFreeDataModuleUpdateBatchSize:
    """Test DataModule batch size updates."""

    @patch(MOCK_PATH)
    def test_update_batch_size_recreates_dataloader(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that update_batch_size recreates the dataloader."""
        mock_new_dl = MagicMock(spec=DataLoader)
        mock_create_dl.side_effect = [
            (mock_dataloader, None, None),  # initial setup
            (mock_new_dl, None, None),  # after update
        ]

        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer, batch_size=32)
        dm.setup(stage="fit")
        assert dm.train_dataloader_instance == mock_dataloader

        dm.update_batch_size(64)

        assert dm.batch_size == 64
        assert dm.train_dataloader_instance == mock_new_dl
        # Verify create_pretraining_dataloader was called twice
        assert mock_create_dl.call_count == 2
        # Check the second call used the new batch size
        second_call = mock_create_dl.call_args_list[1]
        assert second_call.kwargs["batch_size"] == 64

    @patch(MOCK_PATH)
    def test_update_batch_size_same_value_no_recreation(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that update_batch_size with same value doesn't recreate."""
        mock_create_dl.return_value = (mock_dataloader, None, None)

        dm = WrinkleFreeDataModule(tokenizer=mock_tokenizer, batch_size=32)
        dm.setup(stage="fit")

        dm.update_batch_size(32)  # Same value

        # Should only be called once (initial setup)
        assert mock_create_dl.call_count == 1


class TestWrinkleFreeDataModuleDistributed:
    """Test DataModule distributed training parameters."""

    @patch(MOCK_PATH)
    def test_distributed_params_passed_to_dataloader(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that world_size and rank are passed to create_pretraining_dataloader."""
        mock_create_dl.return_value = (mock_dataloader, None, None)

        dm = WrinkleFreeDataModule(
            tokenizer=mock_tokenizer,
            world_size=8,
            rank=3,
        )
        dm.setup(stage="fit")

        call_kwargs = mock_create_dl.call_args.kwargs
        assert call_kwargs["world_size"] == 8
        assert call_kwargs["rank"] == 3

    @patch(MOCK_PATH)
    def test_packed_param_passed_to_dataloader(
        self, mock_create_dl, mock_tokenizer, mock_dataloader
    ):
        """Test that packed parameter is passed correctly."""
        mock_create_dl.return_value = (mock_dataloader, None, None)

        dm = WrinkleFreeDataModule(
            tokenizer=mock_tokenizer,
            packed=False,
        )
        dm.setup(stage="fit")

        call_kwargs = mock_create_dl.call_args.kwargs
        assert call_kwargs["packed"] is False
