#!/usr/bin/env python3
"""Standalone BitNet server with native SIMD kernels.

Uses sgl-kernel BitNet ops (AVX2/AVX512) for fast CPU inference.
OpenAI-compatible API for easy integration.

Run: uv run python demo/serve_bitnet.py --port 30000
"""

import argparse
import asyncio
import time
import uuid
from threading import Thread
from typing import AsyncGenerator, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# Try to import BitNet kernels - FAIL LOUDLY if not available
try:
    from sgl_kernel.quantization import bitnet_check_kernel_available
    BITNET_KERNEL_AVAILABLE = bitnet_check_kernel_available()
    if not BITNET_KERNEL_AVAILABLE:
        raise RuntimeError(
            "BitNet native kernels not available! "
            "Build sgl-kernel with: cd extern/sglang-bitnet/sgl-kernel && uv pip install -e . --no-build-isolation"
        )
except ImportError as e:
    raise ImportError(
        f"sgl_kernel not installed! "
        f"Build sgl-kernel with: cd extern/sglang-bitnet/sgl-kernel && uv pip install -e . --no-build-isolation\n"
        f"Original error: {e}"
    ) from e

# Try to import kernel patcher
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from wrinklefree_inference.kernels import patch_model_with_native_kernels
    PATCHER_AVAILABLE = True
except ImportError:
    PATCHER_AVAILABLE = False
    patch_model_with_native_kernels = None

app = FastAPI(title="BitNet Inference Server")

# Global model and tokenizer
model = None
tokenizer = None
model_name = "microsoft/bitnet-b1.58-2B-4T"


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "bitnet"
    messages: list[Message]
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "bitnet"
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False


def load_model(model_id: str, use_native_kernels: bool = True):
    """Load BitNet model and tokenizer."""
    global model, tokenizer, model_name

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id}")
    print(f"BitNet native kernels available: {BITNET_KERNEL_AVAILABLE}")
    print(f"Kernel patcher available: {PATCHER_AVAILABLE}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model_name = model_id
    print(f"Model loaded. Device: {model.device}")

    # Try to patch with native kernels
    if use_native_kernels and PATCHER_AVAILABLE and BITNET_KERNEL_AVAILABLE:
        try:
            patched = patch_model_with_native_kernels(model)
            print(f"Patched {patched} layers with native BitNet kernels")
        except Exception as e:
            print(f"Warning: Failed to patch model: {e}")


def generate_text(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> tuple[str, dict]:
    """Generate text without streaming."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.01,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start_time
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    stats = {
        "tokens": len(generated_ids),
        "time_s": elapsed,
        "tok_per_s": len(generated_ids) / elapsed if elapsed > 0 else 0,
    }

    return text, stats


async def generate_stream(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Generate text with streaming."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.01,
        "temperature": max(temperature, 0.01),
        "top_p": 0.9,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    # Start generation in background thread
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Stream tokens
    for text in streamer:
        yield text

    thread.join()


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name, "bitnet_kernels": BITNET_KERNEL_AVAILABLE}


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "microsoft",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    # Format messages as chat
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if request.stream:
        async def stream_response():
            request_id = str(uuid.uuid4())
            async for token in generate_stream(
                prompt, request.max_tokens, request.temperature
            ):
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {chunk}\n\n"

            # Final chunk
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )

    # Non-streaming
    text, stats = generate_text(prompt, request.max_tokens, request.temperature)

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,  # Not tracked
            "completion_tokens": stats["tokens"],
            "total_tokens": stats["tokens"],
        },
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    if request.stream:
        async def stream_response():
            request_id = str(uuid.uuid4())
            async for token in generate_stream(
                request.prompt, request.max_tokens, request.temperature
            ):
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": token,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )

    text, stats = generate_text(
        request.prompt, request.max_tokens, request.temperature
    )

    return {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": stats["tokens"],
            "total_tokens": stats["tokens"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="BitNet Inference Server")
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    load_model(args.model)

    print(f"\nStarting server at http://{args.host}:{args.port}")
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1/chat/completions")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

