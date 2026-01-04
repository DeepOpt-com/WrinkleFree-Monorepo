#!/usr/bin/env python3
"""Streamlit chat interface using BitNet.cpp backend.

Prerequisites:
1. Build BitNet.cpp:
   cd extern/BitNet
   python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

2. Start the BitNet.cpp server:
   cd extern/BitNet
   python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -t 8 --port 8080

3. Run this Streamlit app:
   uv run streamlit run demo/serve_bitnet_cpp.py --server.port 7860
"""

import requests
import streamlit as st
import time
from typing import Generator


BITNET_URL = "http://localhost:8080"


def check_server() -> bool:
    """Check if BitNet.cpp server is running."""
    try:
        resp = requests.get(f"{BITNET_URL}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def generate_streaming(prompt: str, max_tokens: int, temperature: float) -> Generator[str, None, None]:
    """Stream tokens from BitNet.cpp server."""
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stop": ["<|eot_id|>", "<|end|>", "</s>"],
    }

    with requests.post(
        f"{BITNET_URL}/completion",
        json=payload,
        stream=True,
        timeout=120,
    ) as resp:
        for line in resp.iter_lines():
            if line:
                try:
                    import json
                    data = json.loads(line.decode().removeprefix("data: "))
                    if "content" in data:
                        yield data["content"]
                    if data.get("stop", False):
                        break
                except:
                    continue


def generate_sync(prompt: str, max_tokens: int, temperature: float) -> tuple[str, float]:
    """Generate text synchronously."""
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": False,
        "stop": ["<|eot_id|>", "<|end|>", "</s>"],
    }

    start = time.time()
    resp = requests.post(f"{BITNET_URL}/completion", json=payload, timeout=120)
    elapsed = time.time() - start

    data = resp.json()
    content = data.get("content", "")
    tokens = data.get("tokens_predicted", len(content.split()))

    tok_per_sec = tokens / elapsed if elapsed > 0 else 0
    return content, tok_per_sec


def main():
    st.set_page_config(
        page_title="BitNet-2B Chat",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 BitNet-b1.58-2B-4T Chat")
    st.caption("Using BitNet.cpp backend | Native 1.58-bit inference")

    # Check server
    if not check_server():
        st.error("""
        **BitNet.cpp server not running!**

        Start it with:
        ```bash
        cd extern/BitNet
        python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -t 8 --port 8080
        ```
        """)
        return

    st.success("Connected to BitNet.cpp server")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        max_tokens = st.slider("Max tokens", 10, 500, 100, step=10)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.1)
        use_streaming = st.checkbox("Streaming", value=True)

        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Format prompt with chat template
        formatted = f"User: {prompt}<|eot_id|>Assistant:"

        with st.chat_message("assistant"):
            if use_streaming:
                output_placeholder = st.empty()
                stats_placeholder = st.empty()
                full_response = ""
                start = time.time()
                token_count = 0

                for token in generate_streaming(formatted, max_tokens, temperature):
                    full_response += token
                    token_count += 1
                    output_placeholder.markdown(full_response + "▌")

                output_placeholder.markdown(full_response)
                elapsed = time.time() - start
                tok_per_sec = token_count / elapsed if elapsed > 0 else 0
                stats_placeholder.caption(f"Generated {token_count} tokens @ {tok_per_sec:.1f} tok/s")
            else:
                with st.spinner("Generating..."):
                    full_response, tok_per_sec = generate_sync(formatted, max_tokens, temperature)
                st.markdown(full_response)
                st.caption(f"Generated @ {tok_per_sec:.1f} tok/s")

        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
