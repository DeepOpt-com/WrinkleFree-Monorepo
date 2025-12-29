"""Python wrapper for BitNet.cpp inference server."""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class BitNetServer:
    """
    Manager for BitNet.cpp inference server.

    Handles starting, stopping, and health checking of the server.

    Args:
        bitnet_path: Path to BitNet.cpp installation
        model_path: Path to GGUF model file
        port: Server port
        num_threads: Number of CPU threads (0 = auto-detect)
        context_size: KV cache context size (default 4096)
        continuous_batching: Enable continuous batching for concurrent requests
        host: Host to bind to
    """

    def __init__(
        self,
        bitnet_path: Path,
        model_path: Path,
        port: int = 8080,
        num_threads: int = 0,
        context_size: int = 4096,
        continuous_batching: bool = True,
        host: str = "0.0.0.0",
    ):
        self.bitnet_path = Path(bitnet_path)
        self.model_path = Path(model_path)
        self.port = port
        self.num_threads = num_threads
        self.context_size = context_size
        self.continuous_batching = continuous_batching
        self.host = host
        self._process: Optional[subprocess.Popen] = None

    def start(self, wait_for_ready: bool = True, timeout: int = 120) -> None:
        """
        Start the BitNet inference server.

        Args:
            wait_for_ready: Wait for server to be ready
            timeout: Timeout in seconds for server startup
        """
        if self._process is not None:
            logger.warning("Server already running")
            return

        server_script = self.bitnet_path / "run_inference_server.py"

        if not server_script.exists():
            raise FileNotFoundError(f"BitNet server script not found: {server_script}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        cmd = [
            "python",
            str(server_script),
            "-m", str(self.model_path),
            "-t", str(self.num_threads),
            "-c", str(self.context_size),
            "--host", self.host,
            "--port", str(self.port),
        ]

        if self.continuous_batching:
            cmd.append("-cb")

        logger.info(f"Starting BitNet server: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.bitnet_path),
        )

        if wait_for_ready:
            self._wait_for_ready(timeout)

    def _wait_for_ready(self, timeout: int) -> None:
        """Wait for server to be ready."""
        start_time = time.time()
        check_url = f"http://localhost:{self.port}/health"

        while time.time() - start_time < timeout:
            try:
                response = requests.get(check_url, timeout=2)
                if response.status_code == 200:
                    logger.info("BitNet server is ready")
                    return
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                raise RuntimeError(f"Server process died: {stderr}")

            time.sleep(1)

        raise TimeoutError(f"Server did not start within {timeout} seconds")

    def stop(self) -> None:
        """Stop the BitNet inference server."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            logger.info("BitNet server stopped")

    def is_running(self) -> bool:
        """Check if server is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def health_check(self) -> bool:
        """Check server health via HTTP endpoint."""
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @property
    def base_url(self) -> str:
        """Get the server base URL."""
        return f"http://localhost:{self.port}"

    def __enter__(self) -> "BitNetServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def get_default_bitnet_path() -> Path:
    """Get the default BitNet path relative to this package."""
    return Path(__file__).parent.parent.parent.parent / "extern" / "BitNet"
