#!/usr/bin/env python3
"""Streamlit chat interface using BitNet.cpp backend.

Uses the BitNet.cpp HTTP server for inference (much more memory efficient).

Run:
  1. Start BitNet.cpp server: python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
  2. Start Streamlit: uv run streamlit run demo/serve_streamlit_api.py --server.port 7860 --server.address 0.0.0.0
"""

import os
import time
import json
import requests
import streamlit as st

# BitNet.cpp server URL
BITNET_URL = os.environ.get("BITNET_URL", "http://127.0.0.1:8080")


def check_server():
    """Check if BitNet.cpp server is running."""
    try:
        resp = requests.get(f"{BITNET_URL}/health", timeout=2)
        return resp.status_code == 200
    except:
        return False


def generate_completion(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> dict:
    """Generate completion using BitNet.cpp server."""
    try:
        resp = requests.post(
            f"{BITNET_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def format_chat_prompt(messages: list) -> str:
    """Format messages for BitNet chat template."""
    # Simple Llama-style chat template
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "user":
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == "assistant":
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

    # Add assistant prompt
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


# Page config
st.set_page_config(
    page_title="BitNet Chat",
    page_icon="",
    layout="wide"
)

st.title("BitNet-b1.58-2B-4T Chat")

# Sidebar
with st.sidebar:
    st.header("Settings")
    max_tokens = st.slider("Max tokens", 32, 512, 256)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    st.divider()

    if check_server():
        st.success(f"Connected to {BITNET_URL}")
    else:
        st.error(f"Server not running at {BITNET_URL}")
        st.info("Start server with:\n```\ncd extern/BitNet\npython run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf\n```")

    st.divider()

    st.caption("Model: BitNet-b1.58-2B-4T (1.1GB)")
    st.caption("Quantization: i2_s (2-bit ternary)")
    st.caption("Context: 4096 tokens")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# Display chat history
for msg in st.session_state.messages[1:]:  # Skip system message
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type a message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.perf_counter()

            # Format full conversation
            formatted_prompt = format_chat_prompt(st.session_state.messages)

            # Generate
            result = generate_completion(formatted_prompt, max_tokens, temperature)

            elapsed = time.perf_counter() - start_time

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                response = result.get("content", "").strip()

                # Clean up response (stop at eot tokens)
                for stop in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]:
                    if stop in response:
                        response = response.split(stop)[0]

                st.markdown(response)

                # Stats
                tokens = result.get("tokens_predicted", 0)
                tok_per_sec = tokens / elapsed if elapsed > 0 else 0
                st.caption(f"Generated {tokens} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")

                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    st.rerun()
