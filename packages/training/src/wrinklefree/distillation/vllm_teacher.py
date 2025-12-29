"""vLLM-based teacher wrapper for efficient knowledge distillation.

This module provides a teacher wrapper that uses vLLM to serve the teacher model,
enabling efficient batched inference with PagedAttention.

Usage:
    # Start vLLM server (separate process):
    # python -m vllm.entrypoints.openai.api_server --model HuggingFaceTB/SmolLM2-135M

    from wrinklefree.distillation.vllm_teacher import VLLMTeacherWrapper

    teacher = VLLMTeacherWrapper(
        model_name="HuggingFaceTB/SmolLM2-135M",
        base_url="http://localhost:8000",
    )

    # Get teacher logits (top-k)
    logits = teacher.get_logits(input_ids)
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for vLLM teacher wrapper."""

    model_name: str
    base_url: str = "http://localhost:8000"
    top_k_logprobs: int = 100  # Number of top logprobs to request
    timeout: float = 30.0
    max_concurrent_requests: int = 8
    use_async: bool = True


class VLLMTeacherWrapper(nn.Module):
    """
    Teacher wrapper that queries a vLLM server for logits.

    Uses vLLM's OpenAI-compatible API to get teacher outputs.
    Since vLLM returns sparse top-k logprobs, this provides
    approximate distillation (works well in practice).

    Args:
        model_name: Name of the model served by vLLM
        base_url: Base URL of the vLLM server
        top_k_logprobs: Number of top logprobs to request (default: 100)
        vocab_size: Vocabulary size for reconstructing logits tensor
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000",
        top_k_logprobs: int = 100,
        vocab_size: Optional[int] = None,
        timeout: float = 30.0,
        use_async: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.top_k_logprobs = top_k_logprobs
        self.vocab_size = vocab_size
        self.timeout = timeout
        self.use_async = use_async

        # For synchronous fallback
        self._executor = ThreadPoolExecutor(max_workers=8)

        # Lazy import to avoid dependency issues
        self._client = None
        self._async_client = None

        logger.info(
            f"VLLMTeacherWrapper initialized: model={model_name}, "
            f"url={base_url}, top_k={top_k_logprobs}"
        )

    def _get_client(self):
        """Get or create synchronous HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=self.timeout)
            except ImportError:
                raise ImportError("httpx is required for VLLMTeacherWrapper. Install with: pip install httpx")
        return self._client

    def _get_async_client(self):
        """Get or create async HTTP client."""
        if self._async_client is None:
            try:
                import httpx
                self._async_client = httpx.AsyncClient(timeout=self.timeout)
            except ImportError:
                raise ImportError("httpx is required for VLLMTeacherWrapper. Install with: pip install httpx")
        return self._async_client

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> dict:
        """
        Forward pass through vLLM server.

        Args:
            input_ids: Input token IDs (batch, seq)
            attention_mask: Attention mask (unused, for API compatibility)
            output_attentions: If True, returns None for attentions (vLLM doesn't support)

        Returns:
            Dictionary with:
                - logits: Sparse logits tensor (top-k values, rest are -inf)
                - attentions: None (not supported by vLLM)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get logits from vLLM
        if self.use_async:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, use sync method instead
                    logits = self._get_logits_sync(input_ids)
                else:
                    logits = loop.run_until_complete(self._get_logits_async(input_ids))
            except RuntimeError:
                # No event loop, use sync method
                logits = self._get_logits_sync(input_ids)
        else:
            logits = self._get_logits_sync(input_ids)

        logits = logits.to(device)

        result = {
            "logits": logits,
            "attentions": None,  # vLLM doesn't expose attention weights
        }

        if output_attentions:
            logger.warning(
                "VLLMTeacherWrapper does not support output_attentions. "
                "Returning None for attentions."
            )

        return result

    def _get_logits_sync(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits synchronously."""
        import json

        client = self._get_client()
        batch_size, seq_len = input_ids.shape

        # Prepare requests for each sequence in batch
        all_logits = []

        for i in range(batch_size):
            # Convert tokens to prompt (vLLM expects text or token list)
            tokens = input_ids[i].tolist()

            # Use completion endpoint with logprobs
            response = client.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": tokens,  # vLLM accepts token IDs directly
                    "max_tokens": 1,  # We only need logits for existing tokens
                    "logprobs": self.top_k_logprobs,
                    "echo": True,  # Include prompt tokens in response
                    "temperature": 0.0,  # Deterministic
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"vLLM request failed: {response.text}")

            data = response.json()

            # Extract logprobs for all positions
            logits = self._parse_logprobs_response(data, seq_len)
            all_logits.append(logits)

        return torch.stack(all_logits)

    async def _get_logits_async(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits asynchronously for better throughput."""
        client = self._get_async_client()
        batch_size, seq_len = input_ids.shape

        async def get_single_logits(tokens: list) -> torch.Tensor:
            response = await client.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": tokens,
                    "max_tokens": 1,
                    "logprobs": self.top_k_logprobs,
                    "echo": True,
                    "temperature": 0.0,
                },
            )

            if response.status_code != 200:
                raise RuntimeError(f"vLLM request failed: {response.text}")

            data = response.json()
            return self._parse_logprobs_response(data, len(tokens))

        # Run all requests concurrently
        tasks = [
            get_single_logits(input_ids[i].tolist())
            for i in range(batch_size)
        ]
        results = await asyncio.gather(*tasks)

        return torch.stack(results)

    def _parse_logprobs_response(
        self,
        response_data: dict,
        seq_len: int,
    ) -> torch.Tensor:
        """Parse vLLM response and construct sparse logits tensor.

        Args:
            response_data: JSON response from vLLM
            seq_len: Expected sequence length

        Returns:
            Logits tensor of shape (seq_len, vocab_size)
        """
        # Extract logprobs from response
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("No choices in vLLM response")

        logprobs_data = choices[0].get("logprobs", {})
        top_logprobs = logprobs_data.get("top_logprobs", [])
        token_logprobs = logprobs_data.get("token_logprobs", [])

        # Infer vocab size if not set
        vocab_size = self.vocab_size or 32000  # Default for many models

        # Create sparse logits tensor (fill with -inf)
        logits = torch.full((seq_len, vocab_size), float("-inf"))

        # Fill in top-k logprobs
        for pos, (pos_logprobs, _) in enumerate(zip(top_logprobs, token_logprobs)):
            if pos_logprobs is None:
                continue

            for token_str, logprob in pos_logprobs.items():
                try:
                    # Token string might be the actual token or token ID
                    token_id = int(token_str) if token_str.isdigit() else None
                    if token_id is not None and 0 <= token_id < vocab_size:
                        logits[pos, token_id] = logprob
                except (ValueError, TypeError):
                    pass

        return logits

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method to get just logits."""
        result = self.forward(input_ids, attention_mask)
        return result["logits"]


class VLLMTeacherWithPrefetch(VLLMTeacherWrapper):
    """
    vLLM teacher with async prefetching for pipelined training.

    Prefetches teacher outputs for the next batch while the student
    is training on the current batch, reducing latency.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefetch_future = None
        self._prefetch_result = None

    def prefetch(self, input_ids: torch.Tensor) -> None:
        """Start prefetching logits for the given input."""
        # Submit prefetch task
        self._prefetch_future = self._executor.submit(
            self._get_logits_sync, input_ids
        )

    def get_prefetched_or_compute(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Get prefetched result if available, otherwise compute synchronously.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (unused)

        Returns:
            Dictionary with logits
        """
        if self._prefetch_future is not None:
            try:
                # Wait for prefetch to complete
                logits = self._prefetch_future.result(timeout=self.timeout)
                self._prefetch_future = None
                return {"logits": logits.to(input_ids.device), "attentions": None}
            except Exception as e:
                logger.warning(f"Prefetch failed, computing synchronously: {e}")
                self._prefetch_future = None

        # Fall back to synchronous computation
        return self.forward(input_ids, attention_mask)


def start_vllm_server(
    model_name: str,
    port: int = 8000,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "bfloat16",
) -> None:
    """
    Start vLLM server as a subprocess.

    This is a convenience function for testing. In production,
    run vLLM server separately.

    Args:
        model_name: HuggingFace model name
        port: Port to serve on
        gpu_memory_utilization: Fraction of GPU memory to use
        dtype: Data type (bfloat16, float16, float32)
    """
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
    ]

    logger.info(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.Popen(cmd)


def create_vllm_or_inprocess_teacher(
    model_name: str,
    use_vllm: bool = False,
    vllm_base_url: str = "http://localhost:8000",
    device: Optional[torch.device] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create either vLLM or in-process teacher.

    Args:
        model_name: HuggingFace model name
        use_vllm: If True, use vLLM server; else use in-process
        vllm_base_url: vLLM server URL
        device: Device for in-process teacher
        **kwargs: Additional arguments for teacher wrapper

    Returns:
        Teacher wrapper module
    """
    if use_vllm:
        return VLLMTeacherWrapper(
            model_name=model_name,
            base_url=vllm_base_url,
            **kwargs,
        )
    else:
        # Import in-process teacher from stage3
        from wrinklefree.training.stage3 import TeacherWrapper

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return TeacherWrapper(
            model_name_or_path=model_name,
            device=device,
            **kwargs,
        )
