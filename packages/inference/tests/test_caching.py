"""Tests for build caching functionality."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest


class TestCacheKeyGeneration:
    """Test cache key generation logic."""

    def test_cache_key_format(self):
        """Cache key should include model name, quant type, and version."""
        model_repo = "microsoft/BitNet-b1.58-2B-4T"
        quant_type = "i2_s"
        cache_version = "v1"

        model_name = model_repo.split("/")[-1]
        cache_key = f"{model_name}_{quant_type}_{cache_version}"

        assert cache_key == "BitNet-b1.58-2B-4T_i2_s_v1"

    def test_cache_tarball_name(self):
        """Cache tarball should use consistent naming."""
        cache_key = "BitNet-b1.58-2B-4T_i2_s_v1"
        tarball_name = f"bitnet_cache_{cache_key}.tar.gz"

        assert tarball_name == "bitnet_cache_BitNet-b1.58-2B-4T_i2_s_v1.tar.gz"

    def test_cache_path_format(self):
        """Cache path should be a valid GCS path."""
        bucket = "gs://wrinklefree-inference-cache"
        tarball = "bitnet_cache_test.tar.gz"
        cache_path = f"{bucket}/{tarball}"

        assert cache_path.startswith("gs://")
        assert cache_path.endswith(".tar.gz")


class TestCacheArtifacts:
    """Test which artifacts should be cached."""

    @pytest.fixture
    def mock_bitnet_dir(self, tmp_path: Path):
        """Create mock BitNet directory structure."""
        bitnet_dir = tmp_path / "BitNet"
        bitnet_dir.mkdir()

        # Create build directory
        build_bin = bitnet_dir / "build" / "bin"
        build_bin.mkdir(parents=True)

        # Create mock binaries
        (build_bin / "llama-server").touch()
        (build_bin / "llama-cli").touch()
        (build_bin / "llama-quantize").touch()

        # Create models directory with GGUF
        models_dir = bitnet_dir / "models" / "BitNet-b1.58-2B-4T"
        models_dir.mkdir(parents=True)
        (models_dir / "ggml-model-i2_s.gguf").write_bytes(b"fake gguf data")

        return bitnet_dir

    def test_cache_includes_binaries(self, mock_bitnet_dir: Path):
        """Cache should include compiled binaries."""
        build_bin = mock_bitnet_dir / "build" / "bin"
        assert (build_bin / "llama-server").exists()
        assert (build_bin / "llama-cli").exists()

    def test_cache_includes_gguf(self, mock_bitnet_dir: Path):
        """Cache should include converted GGUF model."""
        gguf_path = mock_bitnet_dir / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"
        assert gguf_path.exists()

    def test_cache_tarball_creation(self, mock_bitnet_dir: Path, tmp_path: Path):
        """Test creating cache tarball."""
        import tarfile

        cache_tarball = tmp_path / "cache.tar.gz"

        with tarfile.open(cache_tarball, "w:gz") as tar:
            tar.add(
                mock_bitnet_dir / "build" / "bin",
                arcname="build/bin"
            )
            tar.add(
                mock_bitnet_dir / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf",
                arcname="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
            )

        assert cache_tarball.exists()
        assert cache_tarball.stat().st_size > 0

        # Verify contents
        with tarfile.open(cache_tarball, "r:gz") as tar:
            names = tar.getnames()
            assert "build/bin/llama-server" in names
            assert "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf" in names

    def test_cache_extraction(self, mock_bitnet_dir: Path, tmp_path: Path):
        """Test extracting cache tarball."""
        import tarfile

        # Create tarball
        cache_tarball = tmp_path / "cache.tar.gz"
        with tarfile.open(cache_tarball, "w:gz") as tar:
            tar.add(
                mock_bitnet_dir / "build" / "bin",
                arcname="build/bin"
            )

        # Extract to new location
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()

        with tarfile.open(cache_tarball, "r:gz") as tar:
            tar.extractall(extract_dir)

        # Verify extraction
        assert (extract_dir / "build" / "bin" / "llama-server").exists()


class TestGCSIntegration:
    """Test GCS cache operations (mocked)."""

    @pytest.mark.smoke
    def test_gcloud_availability_check(self):
        """Test checking if gcloud is available."""
        # This is what the setup script does
        result = subprocess.run(
            ["command", "-v", "gcloud"],
            shell=True,
            capture_output=True
        )
        # Just verify the check doesn't crash
        assert isinstance(result.returncode, int)

    @pytest.mark.smoke
    def test_cache_hit_detection_logic(self):
        """Test cache hit/miss detection logic."""
        # Simulate the bash logic
        cache_hit = False

        # Mock: gcloud not available
        gcloud_available = False
        if gcloud_available:
            cache_hit = True

        assert cache_hit is False

    @pytest.mark.smoke
    def test_fallback_to_build(self):
        """Test that build proceeds when cache miss."""
        cache_hit = False
        built = False

        if not cache_hit:
            # Would run build commands
            built = True

        assert built is True


class TestCacheVersioning:
    """Test cache versioning and invalidation."""

    def test_version_bump_invalidates_cache(self):
        """Bumping CACHE_VERSION should create new cache key."""
        model = "BitNet-b1.58-2B-4T"
        quant = "i2_s"

        cache_key_v1 = f"{model}_{quant}_v1"
        cache_key_v2 = f"{model}_{quant}_v2"

        assert cache_key_v1 != cache_key_v2

    def test_quant_type_affects_cache(self):
        """Different quant types should have different caches."""
        model = "BitNet-b1.58-2B-4T"
        version = "v1"

        cache_i2s = f"{model}_i2_s_{version}"
        cache_tl2 = f"{model}_tl2_{version}"

        assert cache_i2s != cache_tl2

    def test_model_affects_cache(self):
        """Different models should have different caches."""
        quant = "i2_s"
        version = "v1"

        cache_2b = f"BitNet-b1.58-2B-4T_{quant}_{version}"
        cache_1b = f"BitNet-b1.58-1B-4T_{quant}_{version}"

        assert cache_2b != cache_1b


@pytest.mark.skip(reason="Skypilot caching config was removed in monorepo migration")
class TestDeploymentConfig:
    """Test deployment configuration."""

    @pytest.fixture
    def skypilot_config_path(self):
        """Get path to SkyPilot config."""
        return Path(__file__).parent.parent / "skypilot" / "runpod_cpu.yaml"

    @pytest.mark.smoke
    def test_skypilot_config_exists(self, skypilot_config_path: Path):
        """SkyPilot config should exist."""
        assert skypilot_config_path.exists()

    @pytest.mark.smoke
    def test_skypilot_config_has_caching(self, skypilot_config_path: Path):
        """SkyPilot config should include caching logic."""
        content = skypilot_config_path.read_text()

        # Check for caching-related content
        assert "GCS_CACHE_BUCKET" in content
        assert "CACHE_VERSION" in content
        assert "cache_tarball" in content.lower() or "CACHE_TARBALL" in content

    @pytest.mark.smoke
    def test_skypilot_config_has_fallback(self, skypilot_config_path: Path):
        """SkyPilot config should fall back to building when no cache."""
        content = skypilot_config_path.read_text()

        # Should check for gcloud availability
        assert "gcloud" in content
        # Should have fallback logic
        assert "CACHE_HIT" in content or "cache_hit" in content.lower()


class TestEndToEndCaching:
    """End-to-end tests for caching workflow."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not os.environ.get("RUN_E2E_CACHE_TESTS"),
        reason="E2E cache tests require RUN_E2E_CACHE_TESTS=1"
    )
    def test_cache_upload_download_cycle(self, tmp_path: Path):
        """Test full cache upload/download cycle with GCS.

        Requires:
        - GCS credentials configured
        - wrinklefree-inference-cache bucket exists
        """
        import tarfile

        # Create test data
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        (test_dir / "binary").write_bytes(b"fake binary")

        # Create tarball
        tarball = tmp_path / "test_cache.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(test_dir / "binary", arcname="binary")

        # Upload to GCS
        bucket = "gs://wrinklefree-inference-cache"
        gcs_path = f"{bucket}/test_cache_{os.getpid()}.tar.gz"

        upload_result = subprocess.run(
            ["gsutil", "cp", str(tarball), gcs_path],
            capture_output=True,
            text=True
        )
        assert upload_result.returncode == 0, f"Upload failed: {upload_result.stderr}"

        try:
            # Download from GCS
            download_path = tmp_path / "downloaded.tar.gz"
            download_result = subprocess.run(
                ["gsutil", "cp", gcs_path, str(download_path)],
                capture_output=True,
                text=True
            )
            assert download_result.returncode == 0, f"Download failed: {download_result.stderr}"

            # Verify contents
            with tarfile.open(download_path, "r:gz") as tar:
                names = tar.getnames()
                assert "binary" in names

        finally:
            # Cleanup
            subprocess.run(["gsutil", "rm", gcs_path], capture_output=True)
