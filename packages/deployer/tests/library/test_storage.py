"""Tests for wf_deploy.utils.storage module."""

import pytest
from pathlib import Path

from wf_deploy.utils.storage import (
    StorageType,
    StoragePath,
    parse_storage_path,
    validate_model_path,
)


class TestParseStoragePath:
    """Tests for parse_storage_path function."""

    def test_s3_path(self):
        """Test parsing S3 path."""
        result = parse_storage_path("s3://my-bucket/models/model.gguf")

        assert result.storage_type == StorageType.S3
        assert result.bucket == "my-bucket"
        assert result.key == "models/model.gguf"
        assert result.is_remote
        assert not result.is_local

    def test_s3_path_no_key(self):
        """Test parsing S3 path with no key."""
        result = parse_storage_path("s3://my-bucket")

        assert result.storage_type == StorageType.S3
        assert result.bucket == "my-bucket"
        assert result.key == ""

    def test_gcs_path(self):
        """Test parsing GCS path."""
        result = parse_storage_path("gs://my-bucket/models/model.gguf")

        assert result.storage_type == StorageType.GCS
        assert result.bucket == "my-bucket"
        assert result.key == "models/model.gguf"
        assert result.is_remote

    def test_huggingface_path(self):
        """Test parsing HuggingFace path."""
        result = parse_storage_path("hf://HuggingFaceTB/SmolLM2-135M-Instruct")

        assert result.storage_type == StorageType.HUGGINGFACE
        assert result.bucket is None
        assert result.key == "HuggingFaceTB/SmolLM2-135M-Instruct"
        assert result.is_remote

    def test_azure_path(self):
        """Test parsing Azure path."""
        result = parse_storage_path("azure://container/models/model.gguf")

        assert result.storage_type == StorageType.AZURE
        assert result.bucket == "container"
        assert result.key == "models/model.gguf"
        assert result.is_remote

    def test_r2_path(self):
        """Test parsing R2 path."""
        result = parse_storage_path("r2://my-bucket/models/model.gguf")

        assert result.storage_type == StorageType.R2
        assert result.bucket == "my-bucket"
        assert result.key == "models/model.gguf"

    def test_local_path_absolute(self):
        """Test parsing absolute local path."""
        result = parse_storage_path("/home/user/models/model.gguf")

        assert result.storage_type == StorageType.LOCAL
        assert result.bucket is None
        assert result.key == "/home/user/models/model.gguf"
        assert result.is_local
        assert not result.is_remote

    def test_local_path_relative(self):
        """Test parsing relative local path."""
        result = parse_storage_path("./models/model.gguf")

        assert result.storage_type == StorageType.LOCAL
        assert result.key == "./models/model.gguf"

    def test_whitespace_handling(self):
        """Test whitespace is trimmed."""
        result = parse_storage_path("  s3://bucket/key  ")

        assert result.storage_type == StorageType.S3
        assert result.bucket == "bucket"
        assert result.key == "key"


class TestStoragePath:
    """Tests for StoragePath class."""

    def test_to_uri_s3(self):
        """Test converting S3 path back to URI."""
        path = StoragePath(
            StorageType.S3, "my-bucket", "models/model.gguf", "s3://my-bucket/models/model.gguf"
        )
        assert path.to_uri() == "s3://my-bucket/models/model.gguf"

    def test_to_uri_gcs(self):
        """Test converting GCS path back to URI."""
        path = StoragePath(
            StorageType.GCS, "my-bucket", "models/model.gguf", "gs://my-bucket/models/model.gguf"
        )
        assert path.to_uri() == "gs://my-bucket/models/model.gguf"

    def test_to_uri_hf(self):
        """Test converting HuggingFace path back to URI."""
        path = StoragePath(
            StorageType.HUGGINGFACE, None, "org/model", "hf://org/model"
        )
        assert path.to_uri() == "hf://org/model"

    def test_to_uri_local(self):
        """Test converting local path back to URI."""
        path = StoragePath(
            StorageType.LOCAL, None, "/path/to/model", "/path/to/model"
        )
        assert path.to_uri() == "/path/to/model"


class TestValidateModelPath:
    """Tests for validate_model_path function."""

    def test_remote_path_valid(self):
        """Test remote paths are always valid."""
        assert validate_model_path("s3://bucket/model.gguf")
        assert validate_model_path("gs://bucket/model.gguf")
        assert validate_model_path("hf://org/model")

    def test_local_path_exists(self, tmp_path):
        """Test local path that exists is valid."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()

        assert validate_model_path(str(model_file))

    def test_local_path_not_exists(self, tmp_path):
        """Test local path that doesn't exist raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            validate_model_path(str(tmp_path / "nonexistent.gguf"))
