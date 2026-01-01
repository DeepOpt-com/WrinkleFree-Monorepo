"""Tests for BitNet server deployment.

These tests verify the deployed server works correctly.
Run with:
    pytest tests/test_deployment.py -v -m deployment

For E2E tests that spin up actual GCP instances:
    pytest tests/test_deployment.py -v -m e2e
"""

import json
import os
import subprocess
import time

import pytest
import requests

# Mark all tests in this module
pytestmark = pytest.mark.deployment


def get_server_url() -> str:
    """Get the server URL from environment or default."""
    return os.environ.get("BITNET_SERVER_URL", "http://localhost:30000")


class TestServerHealth:
    """Test server health and basic functionality."""

    def test_health_endpoint(self):
        """Test that health endpoint returns ok."""
        url = get_server_url()
        try:
            response = requests.get(f"{url}/health", timeout=5)
            assert response.status_code == 200
            assert response.text == "ok"
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running")

    def test_models_endpoint(self):
        """Test that models endpoint returns model info."""
        url = get_server_url()
        try:
            response = requests.get(f"{url}/v1/models", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert len(data["data"]) > 0
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running")


class TestChatCompletions:
    """Test chat completion functionality."""

    @pytest.fixture
    def server_url(self):
        url = get_server_url()
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Server not healthy")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running")
        return url

    def test_simple_math(self, server_url):
        """Test simple arithmetic question."""
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
            },
            timeout=60,
        )
        assert response.status_code == 200
        data = response.json()

        assert "choices" in data
        assert len(data["choices"]) == 1

        content = data["choices"][0]["message"]["content"]
        assert "4" in content, f"Expected '4' in response, got: {content}"

    def test_coherent_response(self, server_url):
        """Test that response is coherent (not garbage like 'GGGGG...')."""
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is your name?"}],
                "max_tokens": 50,
            },
            timeout=60,
        )
        assert response.status_code == 200
        data = response.json()

        content = data["choices"][0]["message"]["content"]

        # Check for common garbage patterns
        assert "GGGGG" not in content, f"Response contains garbage: {content}"
        assert "AAAAA" not in content, f"Response contains garbage: {content}"

        # Check that response contains actual words
        words = content.split()
        assert len(words) > 0, "Response should contain words"

    def test_usage_stats(self, server_url):
        """Test that usage stats are returned."""
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 20,
            },
            timeout=60,
        )
        assert response.status_code == 200
        data = response.json()

        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] == (
            data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
        )


class TestPerformance:
    """Performance benchmarks."""

    @pytest.fixture
    def server_url(self):
        url = get_server_url()
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code != 200:
                pytest.skip("Server not healthy")
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running")
        return url

    @pytest.mark.slow
    def test_throughput(self, server_url):
        """Measure tokens per second."""
        num_tokens = 100
        start_time = time.time()

        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Explain quantum computing in detail"}
                ],
                "max_tokens": num_tokens,
            },
            timeout=120,
        )

        elapsed = time.time() - start_time
        assert response.status_code == 200

        data = response.json()
        completion_tokens = data["usage"]["completion_tokens"]
        tokens_per_second = completion_tokens / elapsed

        print(f"\nPerformance: {tokens_per_second:.1f} tok/s")
        print(f"Generated {completion_tokens} tokens in {elapsed:.1f}s")

        # Minimum acceptable performance
        assert tokens_per_second > 5.0, f"Too slow: {tokens_per_second:.1f} tok/s"


@pytest.mark.e2e
class TestE2EDeployment:
    """End-to-end deployment tests.

    These tests spin up actual GCP instances using SkyPilot.
    Only run explicitly with: pytest -m e2e

    Estimated cost: ~$0.10-0.20 per test run
    """

    @pytest.fixture(scope="class")
    def deployed_server(self):
        """Deploy server and return its URL."""
        # Check if we should use an existing cluster
        existing_url = os.environ.get("BITNET_SERVER_URL")
        if existing_url:
            yield existing_url
            return

        # Deploy new cluster
        script_dir = os.path.dirname(os.path.dirname(__file__))
        deploy_script = os.path.join(script_dir, "scripts", "deploy_bitnet.sh")

        print("\n=== Deploying BitNet server ===")
        result = subprocess.run(
            [deploy_script, "--server-type", "batch"],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout for deployment
        )

        if result.returncode != 0:
            pytest.fail(f"Deployment failed: {result.stderr}")

        # Get cluster IP
        ip_result = subprocess.run(
            ["sky", "status", "bitnet-batch", "--endpoint", "30000"],
            capture_output=True,
            text=True,
        )
        server_ip = ip_result.stdout.strip()

        if not server_ip:
            pytest.fail("Could not get server IP")

        server_url = f"http://{server_ip}:30000"

        # Wait for server to be ready
        for _ in range(30):
            try:
                response = requests.get(f"{server_url}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(10)
        else:
            pytest.fail("Server did not become healthy in time")

        yield server_url

        # Cleanup
        print("\n=== Tearing down BitNet server ===")
        subprocess.run([deploy_script, "--down"], timeout=300)

    def test_e2e_coherent_output(self, deployed_server):
        """Test that deployed server produces coherent output."""
        response = requests.post(
            f"{deployed_server}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
            },
            timeout=60,
        )

        assert response.status_code == 200
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        assert "4" in content, f"Expected '4' in response: {content}"
        assert "GGGGG" not in content, "Response should not be garbage"

    def test_e2e_performance(self, deployed_server):
        """Test deployed server performance."""
        start_time = time.time()

        response = requests.post(
            f"{deployed_server}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Explain quantum computing briefly"}
                ],
                "max_tokens": 100,
            },
            timeout=120,
        )

        elapsed = time.time() - start_time
        assert response.status_code == 200

        data = response.json()
        tokens = data["usage"]["completion_tokens"]
        tps = tokens / elapsed

        print(f"\nE2E Performance: {tps:.1f} tok/s ({tokens} tokens in {elapsed:.1f}s)")

        # Should get at least 5 tok/s on GCP C3D-8
        assert tps > 5.0, f"Performance too low: {tps:.1f} tok/s"
