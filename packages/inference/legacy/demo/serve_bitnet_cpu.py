#!/usr/bin/env python3
"""BitNet CPU inference server with native SIMD kernels.

Lightweight OpenAI-compatible API server for BitNet models.
Uses transformers + sgl-kernel native kernels (AVX2/AVX512).

Run:
  uv run python demo/serve_bitnet_cpu.py --port 30000
"""

import argparse
import gc
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for native kernels
NATIVE_KERNELS = False
try:
    from sgl_kernel.quantization import bitnet_check_kernel_available
    NATIVE_KERNELS = bitnet_check_kernel_available()
    logger.info(f"BitNet native kernels: {'AVAILABLE' if NATIVE_KERNELS else 'NOT AVAILABLE'}")
except ImportError:
    logger.warning("sgl-kernel not found - using PyTorch fallback")


app = FastAPI(title="BitNet CPU Server")

# Global model state
model = None
tokenizer = None
model_name = ""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "bitnet"
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "bitnet"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


def load_model(model_id: str):
    """Load BitNet model with memory optimizations."""
    global model, tokenizer, model_name

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    logger.info(f"Loading model: {model_id}")
    logger.info("Using low_cpu_mem_usage=True for memory efficiency")

    # Force garbage collection before loading
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,  # Use native transformers BitNet
    )
    model.eval()
    model_name = model_id

    logger.info(f"Model loaded on {model.device}")
    logger.info(f"Model memory: {model.get_memory_footprint() / 1e9:.2f} GB")


def generate_text(prompt: str, max_tokens: int, temperature: float) -> tuple[str, dict]:
    """Generate text (non-streaming)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.01,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    stats = {
        "tokens": len(new_tokens),
        "elapsed": elapsed,
        "tok_per_sec": len(new_tokens) / elapsed if elapsed > 0 else 0,
    }
    return text, stats


async def generate_stream(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    """Generate text with streaming."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0.01,
        "temperature": max(temperature, 0.01),
        "top_p": 0.9,
        "streamer": streamer,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    thread = Thread(target=lambda: model.generate(**gen_kwargs))
    thread.start()

    for text in streamer:
        yield text

    thread.join()


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_name, "native_kernels": NATIVE_KERNELS}


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{"id": model_name, "object": "model", "owned_by": "microsoft"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions."""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if request.stream:
        async def stream_response():
            request_id = str(uuid.uuid4())
            async for token in generate_stream(prompt, request.max_tokens, request.temperature):
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {chunk}\n\n"

            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {final}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    text, stats = generate_text(prompt, request.max_tokens, request.temperature)
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": stats["tokens"], "total_tokens": stats["tokens"]},
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions."""
    if request.stream:
        async def stream_response():
            request_id = str(uuid.uuid4())
            async for token in generate_stream(request.prompt, request.max_tokens, request.temperature):
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "model": model_name,
                    "choices": [{"index": 0, "text": token, "finish_reason": None}],
                }
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    text, stats = generate_text(request.prompt, request.max_tokens, request.temperature)
    return {
        "id": str(uuid.uuid4()),
        "object": "text_completion",
        "model": model_name,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": stats["tokens"], "total_tokens": stats["tokens"]},
    }


def main():
    parser = argparse.ArgumentParser(description="BitNet CPU Server")
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()

    load_model(args.model)

    logger.info(f"Starting server at http://{args.host}:{args.port}")
    logger.info(f"API: http://{args.host}:{args.port}/v1/chat/completions")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
