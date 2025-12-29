"""Python wrapper for BitNet.cpp inference server."""

import json
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
        num_threads: Number of CPU threads
    """

    def __init__(
        self,
        bitnet_path: Path,
        model_path: Path,
        port: int = 8080,
        num_threads: int = 4,
    ):
        self.bitnet_path = Path(bitnet_path)
        self.model_path = Path(model_path)
        self.port = port
        self.num_threads = num_threads
        self._process: Optional[subprocess.Popen] = None

    def start(self, wait_for_ready: bool = True, timeout: int = 60) -> None:
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

        cmd = [
            "python",
            str(server_script),
            "-m", str(self.model_path),
            "-t", str(self.num_threads),
            "--port", str(self.port),
        ]

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
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=1)
                if response.status_code == 200:
                    logger.info("BitNet server is ready")
                    return
            except requests.exceptions.RequestException:
                pass

            # Check if process died
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode()
                raise RuntimeError(f"Server process died: {stderr}")

            time.sleep(1)

        raise TimeoutError(f"Server did not start within {timeout} seconds")

    def stop(self) -> None:
        """Stop the BitNet inference server."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._process = None
            logger.info("BitNet server stopped")

    def is_running(self) -> bool:
        """Check if server is running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class BitNetClient:
    """
    Python client for BitNet.cpp inference server.

    Provides a simple interface for text generation using the
    quantized 1.58-bit model served by BitNet.cpp.

    Args:
        host: Server hostname
        port: Server port
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        timeout: int = 60,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[list[str]] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            stop: Stop sequences
            stream: Whether to stream response

        Returns:
            Generated text completion
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
        }

        if stop:
            payload["stop"] = stop

        try:
            response = requests.post(
                f"{self.base_url}/completion",
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)
            else:
                return response.json()["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Generation request failed: {e}")
            raise

    def _handle_stream(self, response) -> str:
        """Handle streaming response."""
        full_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "content" in data:
                        full_text += data["content"]
        return full_text

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Chat completion with message history.

        Args:
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Assistant's response
        """
        # Format messages into prompt
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        prompt += "Assistant:"

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        try:
            response = requests.post(
                f"{self.base_url}/tokenize",
                json={"content": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["tokens"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Tokenization request failed: {e}")
            raise

    def detokenize(self, tokens: list[int]) -> str:
        """
        Detokenize token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        try:
            response = requests.post(
                f"{self.base_url}/detokenize",
                json={"tokens": tokens},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Detokenization request failed: {e}")
            raise
