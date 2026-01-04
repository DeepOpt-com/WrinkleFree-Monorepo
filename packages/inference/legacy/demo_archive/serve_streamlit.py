#!/usr/bin/env python3
"""Streamlit chat interface for BitNet-b1.58-2B-4T with streaming.

Run: uv run streamlit run demo/serve_streamlit.py --server.port 7860
"""

import time
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


@st.cache_resource
def load_model():
    """Load model and tokenizer (cached)."""
    model_id = "microsoft/bitnet-b1.58-2B-4T"

    with st.spinner("Loading tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    with st.spinner("Loading BitNet-2B model (this takes ~30s)..."):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cpu",
        )

    return model, tokenizer


def generate_streaming(model, tokenizer, prompt: str, max_tokens: int, temperature: float):
    """Generate tokens with streaming output."""
    # Format as chat
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    # Set up streamer
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    # Generation kwargs
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.01,
        "temperature": max(temperature, 0.01),
        "top_p": 0.9,
        "streamer": streamer,
    }

    # Start generation in background thread
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Stream output
    output_placeholder = st.empty()
    stats_placeholder = st.empty()
    generated_text = ""
    num_tokens = 0
    start_time = time.perf_counter()

    for new_text in streamer:
        generated_text += new_text
        num_tokens += 1
        elapsed = time.perf_counter() - start_time
        tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0

        output_placeholder.markdown(f"**Assistant:** {generated_text}▌")
        stats_placeholder.caption(f"Generated {num_tokens} tokens @ {tok_per_sec:.1f} tok/s")

    thread.join()

    # Final update
    elapsed = time.perf_counter() - start_time
    tok_per_sec = num_tokens / elapsed if elapsed > 0 else 0
    output_placeholder.markdown(f"**Assistant:** {generated_text}")
    stats_placeholder.caption(f"Generated {num_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

    return generated_text


def main():
    st.set_page_config(
        page_title="BitNet-2B Chat",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 BitNet-b1.58-2B-4T Chat")
    st.caption("1.58-bit ternary weights | Transformers backend | Streaming generation")

    # Load model
    model, tokenizer = load_model()

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        max_tokens = st.slider("Max tokens", 10, 500, 100, step=10)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)

        st.divider()
        st.header("Model Info")
        st.write(f"**Layers:** {model.config.num_hidden_layers}")
        st.write(f"**Hidden:** {model.config.hidden_size}")
        st.write(f"**Vocab:** {model.config.vocab_size:,}")
        st.write(f"**Device:** {model.device}")

        st.divider()
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response = generate_streaming(model, tokenizer, prompt, max_tokens, temperature)

        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
