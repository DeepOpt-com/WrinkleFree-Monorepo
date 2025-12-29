"""GCS smoke test verification.

Verifies that the smoke test uploaded checkpoints to GCS correctly.

Run after completing smoke test:
    uv run pytest tests/test_gcs_smoke.py -v

Requires:
    - GOOGLE_APPLICATION_CREDENTIALS set or GCP default credentials
"""

import os
from pathlib import Path

import pytest

# Skip all tests if GCS credentials not available
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    and not Path.home().joinpath(".config/gcloud/application_default_credentials.json").exists(),
    reason="GCS credentials not configured",
)


@pytest.fixture
def gcs_client():
    """Get GCS client."""
    try:
        from google.cloud import storage

        return storage.Client()
    except ImportError:
        pytest.skip("google-cloud-storage not installed")


@pytest.fixture
def smoke_test_bucket(gcs_client):
    """Get smoke test bucket."""
    bucket_name = os.getenv("GCS_BUCKET", "wrinklefree-checkpoints")
    return gcs_client.bucket(bucket_name)


class TestSmokeTestCheckpoints:
    """Test that smoke test checkpoints exist in GCS."""

    def test_stage1_checkpoint_exists(self, smoke_test_bucket):
        """Verify Stage 1 checkpoint was uploaded."""
        blob = smoke_test_bucket.blob(
            "checkpoints/smoke-test/stage1_checkpoint/model.safetensors"
        )
        assert blob.exists(), "Stage 1 model.safetensors not found in GCS"

    def test_stage1_tokenizer_exists(self, smoke_test_bucket):
        """Verify Stage 1 tokenizer was uploaded."""
        blob = smoke_test_bucket.blob(
            "checkpoints/smoke-test/stage1_checkpoint/tokenizer.json"
        )
        assert blob.exists(), "Stage 1 tokenizer.json not found in GCS"

    def test_stage1_9_final_checkpoint_exists(self, smoke_test_bucket):
        """Verify Stage 1.9 final checkpoint was uploaded."""
        blob = smoke_test_bucket.blob(
            "checkpoints/smoke-test/stage1_9_checkpoint/checkpoints/final/checkpoint.pt"
        )
        assert blob.exists(), "Stage 1.9 final checkpoint.pt not found in GCS"

    def test_stage1_9_step_checkpoints_exist(self, smoke_test_bucket):
        """Verify Stage 1.9 intermediate checkpoints were uploaded."""
        for step in [50, 100]:
            blob = smoke_test_bucket.blob(
                f"checkpoints/smoke-test/stage1_9_checkpoint/checkpoints/step_{step}/checkpoint.pt"
            )
            assert blob.exists(), f"Stage 1.9 step_{step} checkpoint not found in GCS"

    def test_hydra_config_exists(self, smoke_test_bucket):
        """Verify Hydra config was uploaded."""
        # List blobs to find the timestamped config directory
        blobs = list(
            smoke_test_bucket.list_blobs(
                prefix="checkpoints/smoke-test/", max_results=50
            )
        )
        config_blobs = [b for b in blobs if ".hydra/config.yaml" in b.name]
        assert len(config_blobs) > 0, "Hydra config.yaml not found in GCS"


class TestSmokeTestSize:
    """Test that checkpoint sizes are reasonable."""

    def test_stage1_model_size(self, smoke_test_bucket):
        """Verify Stage 1 model is reasonable size (SmolLM2-135M ~300MB)."""
        blob = smoke_test_bucket.blob(
            "checkpoints/smoke-test/stage1_checkpoint/model.safetensors"
        )
        blob.reload()
        size_mb = blob.size / (1024 * 1024)
        assert 100 < size_mb < 500, f"Stage 1 model unexpected size: {size_mb:.1f}MB"

    def test_stage1_9_checkpoint_size(self, smoke_test_bucket):
        """Verify Stage 1.9 checkpoint is reasonable size."""
        blob = smoke_test_bucket.blob(
            "checkpoints/smoke-test/stage1_9_checkpoint/checkpoints/final/checkpoint.pt"
        )
        blob.reload()
        size_mb = blob.size / (1024 * 1024)
        # Checkpoint includes model + optimizer state, so larger
        assert 100 < size_mb < 2000, f"Stage 1.9 checkpoint unexpected size: {size_mb:.1f}MB"
