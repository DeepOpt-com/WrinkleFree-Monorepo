"""Python client for BitNet.cpp inference server."""

import asyncio
import json
import logging
from typing import AsyncIterator, Iterator, Optional

import requests

logger = logging.getLogger(__name__)


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

    @classmethod
    def from_url(cls, url: str, timeout: int = 60) -> "BitNetClient":
        """Create client from a full URL."""
        # Strip trailing slash
        url = url.rstrip("/")
        return cls.__new__(cls, base_url=url, timeout=timeout)

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

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """
        Generate text completion with streaming.

        Yields tokens as they are generated.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            stop: Stop sequences

        Yields:
            Generated tokens
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
        }

        if stop:
            payload["stop"] = stop

        response = requests.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])
                        if "content" in data:
                            yield data["content"]
                    except json.JSONDecodeError:
                        continue

    def _handle_stream(self, response: requests.Response) -> str:
        """Handle streaming response and return full text."""
        full_text = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])
                        if "content" in data:
                            full_text += data["content"]
                    except json.JSONDecodeError:
                        continue
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
        prompt = self._format_chat_messages(messages)

        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def _format_chat_messages(self, messages: list[dict[str, str]]) -> str:
        """Format chat messages into a prompt string."""
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
        return prompt

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


class AsyncBitNetClient:
    """
    Async Python client for BitNet.cpp inference server.

    For concurrent request handling and load testing.

    Args:
        base_url: Server base URL
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional["httpx.AsyncClient"] = None

    async def _get_session(self):
        """Get or create async HTTP session."""
        if self._session is None:
            import httpx
            self._session = httpx.AsyncClient(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the async session."""
        if self._session is not None:
            await self._session.aclose()
            self._session = None

    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            session = await self._get_session()
            response = await session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Generate text completion asynchronously.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            stop: Stop sequences

        Returns:
            Generated text completion
        """
        session = await self._get_session()

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        response = await session.post(
            f"{self.base_url}/completion",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["content"]

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate text completion with async streaming.

        Yields tokens as they are generated.
        """
        session = await self._get_session()

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        async with session.stream(
            "POST",
            f"{self.base_url}/completion",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "content" in data:
                            yield data["content"]
                    except json.JSONDecodeError:
                        continue

    async def __aenter__(self) -> "AsyncBitNetClient":
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
