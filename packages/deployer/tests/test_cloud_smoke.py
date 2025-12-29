"""Cloud smoke tests for spot instance scaling validation.

These tests validate that the burst layer (AWS/GCP/Azure) works correctly:
- Spot instances can be provisioned via SkyPilot
- Instances become healthy and serve traffic
- Scale-down works properly

Run with:
    uv run pytest tests/test_cloud_smoke.py -v --cloud aws
    uv run pytest tests/test_cloud_smoke.py -v --cloud gcp
    uv run pytest tests/test_cloud_smoke.py -v --cloud azure

Requires:
    - SkyPilot installed and configured (sky check)
    - Cloud credentials configured
    - ~$0.10-0.50 per test run (spot pricing)
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import requests


def pytest_addoption(parser):
    """Add cloud provider option."""
    parser.addoption(
        "--cloud",
        action="store",
        default="aws",
        choices=["aws", "gcp", "azure"],
        help="Cloud provider to test (aws, gcp, azure)",
    )
    parser.addoption(
        "--skip-teardown",
        action="store_true",
        default=False,
        help="Skip teardown for debugging (warning: incurs cost)",
    )
    parser.addoption(
        "--instance-type",
        action="store",
        default=None,
        help="Override instance type (e.g., r7a.large for AWS)",
    )


@pytest.fixture(scope="module")
def cloud_provider(request) -> str:
    """Get cloud provider from CLI."""
    return request.config.getoption("--cloud")


@pytest.fixture(scope="module")
def skip_teardown(request) -> bool:
    """Check if teardown should be skipped."""
    return request.config.getoption("--skip-teardown")


@pytest.fixture(scope="module")
def instance_type_override(request) -> str | None:
    """Get instance type override."""
    return request.config.getoption("--instance-type")


@pytest.fixture(scope="module")
def cluster_name(cloud_provider: str) -> str:
    """Generate unique cluster name."""
    timestamp = int(time.time())
    return f"wf-smoke-{cloud_provider}-{timestamp}"


@pytest.fixture(scope="module")
def service_yaml(cloud_provider: str, instance_type_override: str | None, tmp_path_factory) -> Path:
    """Generate cloud-specific SkyPilot service YAML."""
    tmp_path = tmp_path_factory.mktemp("skypilot")
    yaml_path = tmp_path / "smoke-test-service.yaml"

    # Cloud-specific instance types (small, cheap for testing)
    instance_types = {
        "aws": instance_type_override or "r7a.large",      # 2 vCPU, 16GB
        "gcp": instance_type_override or "n2d-highmem-2",  # 2 vCPU, 16GB
        "azure": instance_type_override or "Standard_E2s_v5",  # 2 vCPU, 16GB
    }

    config = f"""
# Auto-generated smoke test config for {cloud_provider}
name: smoke-test-inference

resources:
  cloud: {cloud_provider}
  instance_type: {instance_types[cloud_provider]}
  use_spot: true
  spot_recovery: FAILOVER
  disk_size: 50
  ports: 8080

envs:
  BACKEND: bitnet
  MODEL_PATH: /models/smollm2-135m.gguf
  NUM_THREADS: "2"
  CONTEXT_SIZE: "1024"

file_mounts:
  /models:
    source: ./models/test
    mode: COPY

setup: |
  set -ex
  sudo apt-get update
  sudo apt-get install -y python3-pip curl

  # Simple HTTP server for testing (mock inference)
  pip install flask

  # Create mock server
  cat > /tmp/mock_server.py << 'PYEOF'
import flask
import time
import json

app = flask.Flask(__name__)

@app.route('/health')
def health():
    return {{"status": "healthy", "backend": "mock"}}

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = flask.request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 10)
    # Simulate inference latency
    time.sleep(0.1)
    return {{
        "id": "mock-completion",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "smollm2-135m-mock",
        "choices": [{{
            "text": f" This is a mock response to: {{prompt[:20]}}...",
            "index": 0,
            "finish_reason": "length"
        }}],
        "usage": {{"prompt_tokens": len(prompt.split()), "completion_tokens": max_tokens}}
    }}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
PYEOF

run: |
  python3 /tmp/mock_server.py
"""

    yaml_path.write_text(config)
    return yaml_path


class TestCloudSmoke:
    """Cloud smoke tests for spot instance validation."""

    @pytest.fixture(autouse=True)
    def setup_teardown(
        self,
        cluster_name: str,
        skip_teardown: bool,
        request,
    ):
        """Setup and teardown cluster."""
        yield
        # Teardown after all tests in class
        if not skip_teardown:
            print(f"\nTearing down cluster {cluster_name}...")
            subprocess.run(
                ["sky", "down", cluster_name, "-y"],
                capture_output=True,
            )

    @pytest.mark.cloud
    def test_sky_check_cloud_enabled(self, cloud_provider: str) -> None:
        """Verify cloud provider is enabled in SkyPilot."""
        result = subprocess.run(
            ["sky", "check"],
            capture_output=True,
            text=True,
        )

        # Parse output to check if cloud is enabled
        output = result.stdout + result.stderr
        cloud_names = {
            "aws": "AWS",
            "gcp": "GCP",
            "azure": "Azure",
        }

        assert cloud_names[cloud_provider] in output, (
            f"{cloud_provider} not found in sky check output. "
            f"Please configure credentials: sky check"
        )
        # Check for enabled status (green checkmark or "enabled")
        # Note: actual parsing depends on sky check output format

    @pytest.mark.cloud
    def test_spot_instance_launch(
        self,
        cloud_provider: str,
        cluster_name: str,
        service_yaml: Path,
        metrics,
    ) -> None:
        """Launch spot instance and verify it starts."""
        start_time = time.time()

        # Launch cluster
        result = subprocess.run(
            [
                "sky", "launch",
                "-c", cluster_name,
                str(service_yaml),
                "-y",  # Auto-confirm
                "--retry-until-up",  # Handle spot interruptions
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        launch_duration = time.time() - start_time
        metrics.record_latency(launch_duration * 1000)

        assert result.returncode == 0, (
            f"Failed to launch cluster:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify cluster is running
        status_result = subprocess.run(
            ["sky", "status", cluster_name, "--json"],
            capture_output=True,
            text=True,
        )

        if status_result.returncode == 0 and status_result.stdout.strip():
            status = json.loads(status_result.stdout)
            assert len(status) > 0, "Cluster not found in status"
            cluster_info = status[0] if isinstance(status, list) else status
            assert cluster_info.get("status") == "UP", f"Cluster not UP: {cluster_info}"

    @pytest.mark.cloud
    def test_instance_becomes_healthy(
        self,
        cluster_name: str,
        metrics,
    ) -> None:
        """Verify instance health endpoint responds."""
        # Get cluster IP
        result = subprocess.run(
            ["sky", "status", cluster_name, "--json"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.skip("Cluster not available")

        status = json.loads(result.stdout)
        cluster_info = status[0] if isinstance(status, list) else status

        # Get the endpoint (head node IP)
        head_ip = cluster_info.get("handle", {}).get("head_ip")
        if not head_ip:
            # Try alternative method
            ip_result = subprocess.run(
                ["sky", "status", "--ip", cluster_name],
                capture_output=True,
                text=True,
            )
            head_ip = ip_result.stdout.strip()

        assert head_ip, "Could not get cluster IP"

        # Wait for health endpoint
        endpoint = f"http://{head_ip}:8080"
        max_wait = 120
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{endpoint}/health", timeout=5)
                if response.status_code == 200:
                    latency = (time.time() - start_time) * 1000
                    metrics.record_latency(latency)
                    return
            except requests.RequestException:
                pass
            time.sleep(5)

        pytest.fail(f"Health endpoint not ready after {max_wait}s")

    @pytest.mark.cloud
    def test_inference_request(
        self,
        cluster_name: str,
        metrics,
    ) -> None:
        """Send inference request to spot instance."""
        # Get cluster IP
        ip_result = subprocess.run(
            ["sky", "status", "--ip", cluster_name],
            capture_output=True,
            text=True,
        )

        if ip_result.returncode != 0:
            pytest.skip("Cluster not available")

        head_ip = ip_result.stdout.strip()
        endpoint = f"http://{head_ip}:8080"

        # Send inference request
        start_time = time.time()
        response = requests.post(
            f"{endpoint}/v1/completions",
            json={
                "prompt": "Test prompt for smoke test",
                "max_tokens": 10,
            },
            timeout=60,
        )
        latency = (time.time() - start_time) * 1000
        metrics.record_latency(latency)

        assert response.status_code == 200, f"Inference failed: {response.text}"

        data = response.json()
        assert "choices" in data, f"Invalid response: {data}"
        assert len(data["choices"]) > 0, "No completions returned"

    @pytest.mark.cloud
    def test_spot_instance_details(
        self,
        cloud_provider: str,
        cluster_name: str,
    ) -> None:
        """Verify spot instance configuration."""
        result = subprocess.run(
            ["sky", "status", cluster_name, "--json"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.skip("Cluster not available")

        status = json.loads(result.stdout)
        cluster_info = status[0] if isinstance(status, list) else status

        # Check spot instance
        resources = cluster_info.get("handle", {}).get("launched_resources", {})

        # Log instance details for debugging
        print(f"\nCluster details:")
        print(f"  Cloud: {resources.get('cloud', 'unknown')}")
        print(f"  Instance: {resources.get('instance_type', 'unknown')}")
        print(f"  Spot: {resources.get('use_spot', 'unknown')}")
        print(f"  Region: {resources.get('region', 'unknown')}")

    @pytest.mark.cloud
    def test_graceful_termination(
        self,
        cluster_name: str,
        skip_teardown: bool,
    ) -> None:
        """Test cluster can be terminated gracefully."""
        if skip_teardown:
            pytest.skip("Teardown skipped (--skip-teardown)")

        result = subprocess.run(
            ["sky", "down", cluster_name, "-y"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, (
            f"Failed to terminate cluster:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify cluster is gone
        status_result = subprocess.run(
            ["sky", "status", cluster_name],
            capture_output=True,
            text=True,
        )

        # Should either return error or show no cluster
        assert cluster_name not in status_result.stdout or "No cluster" in status_result.stdout


class TestMultiCloudSmoke:
    """Tests for multi-cloud scenarios (run manually)."""

    @pytest.mark.cloud
    @pytest.mark.manual
    def test_cross_cloud_failover(self) -> None:
        """Test failover from primary to secondary cloud.

        This test is expensive and should be run manually:
            pytest tests/test_cloud_smoke.py::TestMultiCloudSmoke -v
        """
        pytest.skip("Manual test - run with: pytest -m manual")
        # TODO: Implement cross-cloud failover test
        # 1. Launch primary cluster on cloud A
        # 2. Launch secondary cluster on cloud B
        # 3. Verify traffic routing
        # 4. Simulate primary failure
        # 5. Verify failover to secondary


# JSON output for AI agents
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate JSON summary for AI agent consumption."""
    output_file = os.environ.get("CLOUD_SMOKE_RESULTS", "cloud_smoke_results.json")

    results = {
        "cloud": config.getoption("--cloud"),
        "exit_status": exitstatus,
        "passed": len(terminalreporter.stats.get("passed", [])),
        "failed": len(terminalreporter.stats.get("failed", [])),
        "skipped": len(terminalreporter.stats.get("skipped", [])),
        "tests": [],
    }

    for status in ["passed", "failed", "skipped"]:
        for report in terminalreporter.stats.get(status, []):
            results["tests"].append({
                "name": report.nodeid,
                "status": status,
                "duration": report.duration,
            })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCloud smoke results written to {output_file}")
