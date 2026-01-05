"""Python clients for BitNet and DLM inference servers.

This module provides synchronous and asynchronous clients for communicating
with BitNet inference servers (native, DLM, SGLang backends).

Classes:
    BitNetClient: Synchronous client for BitNet/DLM inference
    AsyncBitNetClient: Asynchronous client using httpx

Example:
    >>> from wf_infer.client import BitNetClient
    >>> client = BitNetClient.from_url("http://localhost:30000")
    >>> # For DLM server (OpenAI-compatible API)
    >>> response = client.chat_openai([
    ...     {"role": "user", "content": "Hello!"}
    ... ])
    >>> # For streaming
    >>> for token in client.chat_openai_stream(messages):
    ...     print(token, end="", flush=True)
"""

from wf_infer.client.bitnet_client import AsyncBitNetClient, BitNetClient

__all__ = ["BitNetClient", "AsyncBitNetClient"]
