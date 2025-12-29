"""Pytest configuration and shared fixtures."""

import os
from typing import Generator

import pytest

from wrinklefree_inference.client.bitnet_client import BitNetClient


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: Quick smoke tests")
    config.addinivalue_line("markers", "integration: Integration tests requiring running server")
    config.addinivalue_line("markers", "kv_cache: KV cache validation tests")
    config.addinivalue_line("markers", "stress: Stress tests with concurrent load")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")
    config.addinivalue_line("markers", "slow: Slow tests that take longer to run")
    config.addinivalue_line("markers", "caching: Tests for GCS build caching")


@pytest.fixture
def inference_url() -> str:
    """Get inference server URL from environment or use default."""
    return os.environ.get("INFERENCE_URL", "http://localhost:8080")


@pytest.fixture
def client(inference_url: str) -> Generator[BitNetClient, None, None]:
    """Create a BitNet client for testing."""
    # Parse URL to extract host and port
    url = inference_url.replace("http://", "").replace("https://", "")
    if ":" in url:
        host, port = url.split(":")
        port = int(port.split("/")[0])
    else:
        host = url.split("/")[0]
        port = 8080

    client = BitNetClient(host=host, port=port, timeout=60)
    yield client


@pytest.fixture
def skip_if_no_server(client: BitNetClient):
    """Skip test if no server is running."""
    if not client.health_check():
        pytest.skip("Inference server not available")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests requiring server if no server available."""
    # Check if server is available
    url = os.environ.get("INFERENCE_URL", "http://localhost:8080")
    server_available = False

    try:
        import requests
        response = requests.get(f"{url}/health", timeout=2)
        server_available = response.status_code == 200
    except Exception:
        pass

    if not server_available:
        skip_marker = pytest.mark.skip(reason="Inference server not available")
        server_required_markers = {"integration", "kv_cache", "stress", "benchmark"}
        for item in items:
            if any(marker in item.keywords for marker in server_required_markers):
                item.add_marker(skip_marker)
