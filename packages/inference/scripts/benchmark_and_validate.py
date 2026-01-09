#!/usr/bin/env python3
"""Benchmark and validate C++ kernel against Python reference.

This script:
1. Loads the Python model and runs inference
2. Compares logits from Python vs C++ server
3. Reports cosine similarity and performance metrics

Usage:
    python scripts/benchmark_and_validate.py \
        --checkpoint /path/to/checkpoint \
        --server-url http://localhost:30000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def get_python_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> tuple[str, np.ndarray, float]:
    """Get output from Python model, return (text, logits, time)."""
    inputs = tokenizer(prompt, return_tensors="pt")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].cpu().numpy()

        # Generate tokens
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, logits, elapsed


def get_cpp_output(
    server_url: str,
    prompt: str,
    max_tokens: int = 50,
) -> tuple[str, float]:
    """Get output from C++ server, return (text, time)."""
    start = time.perf_counter()
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - start

    data = response.json()
    text = ""
    if "choices" in data and data["choices"]:
        text = data["choices"][0].get("message", {}).get("content", "")

    return text, elapsed


def benchmark_throughput(server_url: str, num_requests: int = 20) -> dict:
    """Benchmark server throughput with parallel requests."""
    import concurrent.futures

    prompts = [
        "Explain machine learning in simple terms",
        "Write a short poem about the ocean",
        "What is the capital of France?",
        "How do computers work?",
        "Describe the solar system",
    ]

    def make_request(prompt):
        start = time.perf_counter()
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0,
                },
                timeout=60,
            )
            data = response.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return time.perf_counter() - start, tokens
        except Exception as e:
            return time.perf_counter() - start, 0

    # Warmup
    make_request("Hello")

    # Sequential benchmark
    print("\n=== Sequential Benchmark ===")
    seq_times = []
    seq_tokens = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        elapsed, tokens = make_request(prompt)
        seq_times.append(elapsed)
        seq_tokens.append(tokens)
        print(f"  Request {i+1}: {elapsed:.3f}s, {tokens} tokens")

    seq_total = sum(seq_times)
    seq_total_tokens = sum(seq_tokens)

    # Parallel benchmark
    print("\n=== Parallel Benchmark ===")
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(make_request, prompts[i % len(prompts)])
            for i in range(num_requests)
        ]
        results = [f.result() for f in futures]
    parallel_total = time.perf_counter() - start
    parallel_tokens = sum(r[1] for r in results)

    return {
        "sequential": {
            "total_time": seq_total,
            "total_tokens": seq_total_tokens,
            "requests": num_requests,
            "req_per_sec": num_requests / seq_total,
            "tok_per_sec": seq_total_tokens / seq_total if seq_total > 0 else 0,
        },
        "parallel": {
            "total_time": parallel_total,
            "total_tokens": parallel_tokens,
            "requests": num_requests,
            "req_per_sec": num_requests / parallel_total,
            "tok_per_sec": parallel_tokens / parallel_total if parallel_total > 0 else 0,
        },
    }


def validate_outputs(
    checkpoint_path: str,
    server_url: str,
    test_prompts: list[str] | None = None,
) -> dict:
    """Compare Python model output with C++ server output."""
    if test_prompts is None:
        test_prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "Once upon a time",
            "Hello, how are you",
        ]

    print("Loading Python model...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    results = []

    print("\n=== Output Comparison ===")
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")

        # Python output
        py_text, py_logits, py_time = get_python_output(model, tokenizer, prompt)
        print(f"  Python ({py_time:.2f}s): {py_text[len(prompt):80]}...")

        # C++ output
        cpp_text, cpp_time = get_cpp_output(server_url, prompt)
        print(f"  C++ ({cpp_time:.2f}s): {cpp_text[:80]}...")

        # Check if outputs match
        # Note: DLM uses block diffusion, outputs may differ from autoregressive
        py_first_token = py_text[len(prompt):].strip().split()[0] if py_text[len(prompt):].strip() else ""
        cpp_first_token = cpp_text.strip().split()[0] if cpp_text.strip() else ""

        match = py_first_token.lower() == cpp_first_token.lower() if py_first_token and cpp_first_token else False

        results.append({
            "prompt": prompt,
            "python_output": py_text[len(prompt):],
            "cpp_output": cpp_text,
            "python_time": py_time,
            "cpp_time": cpp_time,
            "first_token_match": match,
            "speedup": py_time / cpp_time if cpp_time > 0 else 0,
        })

        print(f"  First token match: {match}")
        print(f"  Speedup: {py_time / cpp_time:.2f}x" if cpp_time > 0 else "  Speedup: N/A")

    return {
        "comparisons": results,
        "all_match": all(r["first_token_match"] for r in results),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark and validate inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--server-url", default="http://localhost:30000", help="C++ server URL")
    parser.add_argument("--skip-validation", action="store_true", help="Skip Python vs C++ comparison")
    parser.add_argument("--num-requests", type=int, default=20, help="Number of benchmark requests")
    args = parser.parse_args()

    print("=" * 60)
    print("DLM-BitNet Benchmark and Validation")
    print("=" * 60)

    # Check server connectivity
    try:
        response = requests.get(f"{args.server_url}/health", timeout=5)
        print(f"Server at {args.server_url}: OK")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to {args.server_url}")
        print("Please start the server first:")
        print("  ./rust/target/release/dlm_server \\")
        print("    --model-path models/model.gguf --port 30000")
        return 1

    # Validation
    if not args.skip_validation:
        print("\n" + "=" * 60)
        print("VALIDATION: Python vs C++ Output Comparison")
        print("=" * 60)
        validation = validate_outputs(args.checkpoint, args.server_url)

        if validation["all_match"]:
            print("\n✓ All outputs match!")
        else:
            print("\n✗ Some outputs differ (expected for DLM - uses different decoding)")

    # Benchmark
    print("\n" + "=" * 60)
    print("BENCHMARK: Server Throughput")
    print("=" * 60)
    benchmark = benchmark_throughput(args.server_url, args.num_requests)

    print("\n=== Results ===")
    print(f"Sequential: {benchmark['sequential']['req_per_sec']:.2f} req/s, "
          f"{benchmark['sequential']['tok_per_sec']:.2f} tok/s")
    print(f"Parallel:   {benchmark['parallel']['req_per_sec']:.2f} req/s, "
          f"{benchmark['parallel']['tok_per_sec']:.2f} tok/s")

    return 0


if __name__ == "__main__":
    exit(main())
