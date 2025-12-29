"""Streamlit chat interface for BitNet inference.

Run with:
    streamlit run src/wrinklefree_inference/ui/chat.py

Environment variables:
    INFERENCE_URL: BitNet server URL (default: http://localhost:8080)
"""

import os
import streamlit as st
import requests
from typing import Generator

# Configuration
DEFAULT_URL = os.environ.get("INFERENCE_URL", "http://localhost:8080")


def generate_response(
    prompt: str,
    server_url: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    stream: bool = True,
) -> Generator[str, None, None]:
    """Generate response from BitNet server with streaming."""
    endpoint = f"{server_url}/completion"

    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    if stream:
        with requests.post(endpoint, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    # BitNet.cpp streams data as JSON lines
                    import json

                    try:
                        data = json.loads(line.decode("utf-8").removeprefix("data: "))
                        if "content" in data:
                            yield data["content"]
                    except json.JSONDecodeError:
                        continue
    else:
        response = requests.post(endpoint, json=payload, timeout=120)
        response.raise_for_status()
        yield response.json().get("content", "")


def check_server_health(server_url: str) -> bool:
    """Check if server is healthy."""
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def main():
    st.set_page_config(
        page_title="WrinkleFree BitNet Chat",
        page_icon="",
        layout="wide",
    )

    st.title("WrinkleFree BitNet Chat")
    st.caption("1.58-bit quantized LLM inference powered by BitNet.cpp")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        server_url = st.text_input(
            "Server URL",
            value=DEFAULT_URL,
            help="BitNet inference server endpoint",
        )

        # Check server status
        if st.button("Check Connection"):
            if check_server_health(server_url):
                st.success("Server is healthy!")
            else:
                st.error("Cannot connect to server")

        st.divider()

        max_tokens = st.slider(
            "Max Tokens",
            min_value=32,
            max_value=2048,
            value=256,
            step=32,
            help="Maximum tokens to generate",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Sampling temperature (0 = deterministic)",
        )

        stream_output = st.checkbox(
            "Stream output",
            value=True,
            help="Stream tokens as they're generated",
        )

        st.divider()

        st.markdown(
            """
        ### About

        This is a simple chat interface for BitNet 1.58-bit models.

        **Model**: microsoft/BitNet-b1.58-2B-4T

        **Features**:
        - INT8 activations
        - KV caching
        - Streaming output

        Built with WrinkleFree Inference Engine.
        """
        )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build full prompt with chat history
        full_prompt = ""
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                full_prompt += f"User: {content}\n"
            else:
                full_prompt += f"Assistant: {content}\n"
        full_prompt += "Assistant:"

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in generate_response(
                    prompt=full_prompt,
                    server_url=server_url,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream_output,
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "")

                response_placeholder.markdown(full_response)

            except requests.RequestException as e:
                st.error(f"Error connecting to server: {e}")
                full_response = "[Error: Could not connect to inference server]"

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
