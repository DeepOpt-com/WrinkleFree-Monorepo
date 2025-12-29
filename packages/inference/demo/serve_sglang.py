#!/usr/bin/env python3
"""Streamlit chat interface for BitNet models.

Supports two backends:
- bitnet_cpp: Pure C++ backend (BitNet.cpp) - ~26 tok/s, recommended
- sglang: Python-based backend (SGLang-BitNet) - ~16 tok/s

Environment variables:
  BITNET_BACKEND - "bitnet_cpp" (default) or "sglang"
  BITNET_URL - BitNet.cpp server URL (default: http://127.0.0.1:8080)
  SGLANG_URL - SGLang server URL (default: http://127.0.0.1:30000)

Run:
  # BitNet.cpp backend (recommended)
  1. Start server: ./scripts/launch_bitnet_cpp.sh
  2. Start Streamlit: BITNET_BACKEND=bitnet_cpp uv run streamlit run demo/serve_sglang.py

  # SGLang backend
  1. Start server: ./scripts/launch_sglang_bitnet.sh
  2. Start Streamlit: BITNET_BACKEND=sglang uv run streamlit run demo/serve_sglang.py
"""

import html
import json
import os
import time
from typing import Generator

import requests
import streamlit as st

# Backend selection
BACKEND = os.environ.get("BITNET_BACKEND", "bitnet_cpp")  # "bitnet_cpp" or "sglang"

# Server URLs based on backend
if BACKEND == "bitnet_cpp":
    API_URL = os.environ.get("BITNET_URL", "http://127.0.0.1:8080")
else:
    API_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")

# Legacy compatibility
SGLANG_URL = API_URL


def check_server() -> dict:
    """Check if SGLang server is running and get model info."""
    try:
        resp = requests.get(f"{SGLANG_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return {"status": "ok", "model": models[0].get("id", "unknown")}
        return {"status": "error", "message": "No models loaded"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": f"Cannot connect to {SGLANG_URL}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_streaming(
    messages: list,
    max_tokens: int = 256,
    temperature: float = 0.7,
    repetition_penalty: float = 1.1,
) -> Generator[str, None, None]:
    """Stream tokens from SGLang server using SSE."""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "stream": True,
    }

    try:
        with requests.post(
            f"{SGLANG_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")

                # Skip non-data lines
                if not line_str.startswith("data: "):
                    continue

                data_str = line_str[6:]  # Remove "data: " prefix

                # Check for stream end
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.RequestException as e:
        yield f"\n\n**Error:** {e}"


def generate_sync(
    messages: list,
    max_tokens: int = 256,
    temperature: float = 0.7,
    repetition_penalty: float = 1.1,
) -> tuple[str, dict]:
    """Generate response synchronously (non-streaming)."""
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "stream": False,
    }

    start = time.perf_counter()
    resp = requests.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    elapsed = time.perf_counter() - start

    data = resp.json()
    choices = data.get("choices", [])
    content = choices[0]["message"]["content"] if choices else ""

    usage = data.get("usage", {})
    stats = {
        "tokens": usage.get("completion_tokens", len(content.split())),
        "elapsed": elapsed,
        "tok_per_sec": usage.get("completion_tokens", 0) / elapsed if elapsed > 0 else 0,
    }

    return content, stats


# Page config
st.set_page_config(
    page_title="BitNet Chat (SGLang)",
    page_icon="",
    layout="wide",
)

# Custom CSS for fade-in animation on new tokens
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
.token-new {
    animation: fadeIn 1.2s ease-out forwards;
}
.token-recent {
    animation: fadeIn 0.8s ease-out forwards;
}
.token-fading {
    animation: fadeIn 0.5s ease-out forwards;
}
.streaming-text {
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.title("BitNet-b1.58-2B-4T Chat")
st.caption("SGLang backend | Native SIMD kernels | Streaming generation")

# Sidebar
with st.sidebar:
    st.header("Settings")
    max_tokens = st.slider("Max tokens", 32, 512, 256)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1)
    use_streaming = st.checkbox("Streaming", value=True)

    st.divider()

    # Server status
    server_info = check_server()
    if server_info["status"] == "ok":
        backend_name = "BitNet.cpp" if BACKEND == "bitnet_cpp" else "SGLang"
        st.success(f"Connected to {backend_name}")
        st.caption(f"Model: {server_info['model']}")
        st.caption(f"URL: {API_URL}")
    else:
        st.error(f"Server Error: {server_info['message']}")
        if BACKEND == "bitnet_cpp":
            st.info(
                "Start server with:\n```bash\n./scripts/launch_bitnet_cpp.sh\n```"
            )
        else:
            st.info(
                "Start server with:\n```bash\n./scripts/launch_sglang_bitnet.sh\n```"
            )

    st.divider()

    if BACKEND == "bitnet_cpp":
        st.caption("Backend: BitNet.cpp (C++)")
        st.caption("Performance: ~26 tok/s")
    else:
        st.caption("Backend: SGLang-BitNet (Python)")
        st.caption("Performance: ~16 tok/s")
    st.caption("Quantization: 1.58-bit ternary")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type a message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages for API (include system prompt)
    api_messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        *st.session_state.messages,
    ]

    # Generate response
    with st.chat_message("assistant"):
        if use_streaming:
            output_placeholder = st.empty()
            stats_placeholder = st.empty()
            full_response = ""
            token_count = 0
            start_time = time.perf_counter()

            # Track recent tokens for cascading fade effect
            recent_tokens = []  # List of (token, timestamp) tuples

            for token in generate_streaming(api_messages, max_tokens, temperature, repetition_penalty):
                current_time = time.perf_counter()
                recent_tokens.append((token, current_time))

                # Build HTML with fading tokens
                # Tokens older than 1.5s get no animation, recent ones get cascading fade
                html_parts = []
                animated_start_idx = 0

                for i, (t, ts) in enumerate(recent_tokens):
                    age = current_time - ts
                    escaped_t = html.escape(t)

                    if age > 1.2:
                        # Old token - no animation
                        html_parts.append(escaped_t)
                        animated_start_idx = i + 1
                    elif age > 0.6:
                        # Fading token
                        html_parts.append(f'<span class="token-fading">{escaped_t}</span>')
                    elif age > 0.2:
                        # Recent token
                        html_parts.append(f'<span class="token-recent">{escaped_t}</span>')
                    else:
                        # New token
                        html_parts.append(f'<span class="token-new">{escaped_t}</span>')

                html_content = f'''<div class="streaming-text" style="white-space: pre-wrap; font-family: inherit;">{''.join(html_parts)}</div>'''
                output_placeholder.markdown(html_content, unsafe_allow_html=True)

                full_response += token
                token_count += 1

            # Final update (plain markdown for proper rendering)
            output_placeholder.markdown(full_response)
            elapsed = time.perf_counter() - start_time
            tok_per_sec = token_count / elapsed if elapsed > 0 else 0
            stats_placeholder.caption(
                f"Generated {token_count} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)"
            )
        else:
            with st.spinner("Generating..."):
                full_response, stats = generate_sync(api_messages, max_tokens, temperature, repetition_penalty)
            st.markdown(full_response)
            st.caption(
                f"Generated {stats['tokens']} tokens in {stats['elapsed']:.1f}s "
                f"({stats['tok_per_sec']:.1f} tok/s)"
            )

        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
