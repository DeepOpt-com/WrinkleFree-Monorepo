#!/usr/bin/env python3
"""Simple Flask server for BitNet inference using transformers directly.

This server bypasses SGLang's TransformersForCausalLM wrapper which has issues
with BitNet's special weight handling. It uses the transformers library directly
which properly handles the bf16 -> ternary quantization during inference.

Usage:
    python scripts/serve_bitnet_flask.py [--model-path PATH] [--port PORT]

API:
    POST /v1/chat/completions  - OpenAI-compatible chat API
    POST /generate             - Raw text generation
    GET /health               - Health check
"""

import argparse
import logging
import time
from threading import Lock
from typing import Generator

import torch
from flask import Flask, Response, jsonify, request, stream_with_context
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model and tokenizer
model = None
tokenizer = None
model_lock = Lock()


def load_model(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """Load BitNet model and tokenizer."""
    global model, tokenizer

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_path} with dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map='cpu',
        low_cpu_mem_usage=True,
    )
    model.eval()

    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")


def generate_text(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate text completion."""
    with model_lock:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)


def stream_generate(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """Stream text generation token by token."""
    with model_lock:
        inputs = tokenizer(prompt, return_tensors="pt")

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route('/generate', methods=['POST'])
def generate():
    """Raw text generation endpoint."""
    data = request.json
    prompt = data.get('text', data.get('prompt', ''))
    max_tokens = data.get('max_new_tokens', data.get('max_tokens', 128))
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)

    start_time = time.time()
    result = generate_text(prompt, max_tokens, temperature, top_p)
    latency = time.time() - start_time

    return jsonify({
        "text": result,
        "meta_info": {
            "latency": latency,
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(result)),
        }
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 128)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    stream = data.get('stream', False)

    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if stream:
        def generate_stream():
            for token in stream_generate(prompt, max_tokens, temperature, top_p):
                chunk = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "bitnet",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }]
                }
                yield f"data: {jsonify(chunk).get_data(as_text=True)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate_stream()),
            mimetype='text/event-stream'
        )

    start_time = time.time()
    result = generate_text(prompt, max_tokens, temperature, top_p)
    latency = time.time() - start_time

    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "bitnet",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result,
            },
            "finish_reason": "stop" if len(tokenizer.encode(result)) < max_tokens else "length",
        }],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(result)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(result)),
        }
    })


def main():
    parser = argparse.ArgumentParser(description='BitNet Flask Server')
    parser.add_argument('--model-path', type=str,
                        default='models/dlm-bitnet-2b',
                        help='Path to model')
    parser.add_argument('--port', type=int, default=30000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    load_model(args.model_path)

    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
