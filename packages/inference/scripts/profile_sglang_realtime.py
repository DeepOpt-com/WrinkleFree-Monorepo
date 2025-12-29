#!/usr/bin/env python
"""Profile SGLang decode latency in real-time to identify actual bottlenecks.

This script connects to a running SGLang server and measures:
1. End-to-end latency per token
2. Server-side processing time (from logs)
3. Network overhead

Run on Desktop after starting the SGLang server.
"""

import sys
import os
import time
import json
import requests
import statistics

# Configuration
SGLANG_URL = os.getenv("SGLANG_URL", "http://localhost:30000")
N_WARMUP = 3
N_RUNS = 10

def measure_single_token_latency():
    """Measure time to generate a single token."""
    prompt = "Hello"

    payload = {
        "model": "bitnet",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,  # Deterministic
    }

    start = time.perf_counter()
    response = requests.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    latency = time.perf_counter() - start

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None

    return latency * 1000  # Convert to ms


def measure_streaming_latency(n_tokens=20):
    """Measure token-by-token latency in streaming mode."""
    prompt = "Count from 1 to 50."

    payload = {
        "model": "bitnet",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_tokens,
        "temperature": 0,
        "stream": True,
    }

    token_times = []
    last_time = time.perf_counter()
    first_token_time = None
    tokens_received = 0

    response = requests.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                    content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if content:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now - last_time
                        else:
                            token_times.append((now - last_time) * 1000)
                        last_time = now
                        tokens_received += 1
                except json.JSONDecodeError:
                    pass

    return {
        'first_token_latency_ms': first_token_time * 1000 if first_token_time else None,
        'inter_token_latencies_ms': token_times,
        'tokens_received': tokens_received,
    }


def main():
    print("="*70)
    print("SGLANG REAL-TIME LATENCY PROFILER")
    print("="*70)
    print(f"\nServer: {SGLANG_URL}")

    # Check server health
    try:
        health = requests.get(f"{SGLANG_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"Server not healthy: {health.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("Cannot connect to SGLang server. Start it first with:")
        print("  ./scripts/launch_sglang_bitnet.sh")
        return

    print("Server: OK")

    # Warmup
    print(f"\nWarming up ({N_WARMUP} requests)...")
    for _ in range(N_WARMUP):
        measure_single_token_latency()

    # Single token latency
    print(f"\n{'='*70}")
    print("1. SINGLE TOKEN LATENCY")
    print("="*70)

    single_latencies = []
    for i in range(N_RUNS):
        lat = measure_single_token_latency()
        if lat:
            single_latencies.append(lat)
            print(f"  Run {i+1}: {lat:.1f}ms")

    if single_latencies:
        print(f"\n  Mean:   {statistics.mean(single_latencies):.1f}ms")
        print(f"  Median: {statistics.median(single_latencies):.1f}ms")
        print(f"  Stdev:  {statistics.stdev(single_latencies) if len(single_latencies) > 1 else 0:.1f}ms")

    # Streaming latency
    print(f"\n{'='*70}")
    print("2. STREAMING LATENCY (decode phase)")
    print("="*70)

    all_inter_token = []
    first_token_latencies = []

    for i in range(N_RUNS):
        result = measure_streaming_latency(n_tokens=30)
        if result['first_token_latency_ms']:
            first_token_latencies.append(result['first_token_latency_ms'])
        all_inter_token.extend(result['inter_token_latencies_ms'])

        mean_inter = statistics.mean(result['inter_token_latencies_ms']) if result['inter_token_latencies_ms'] else 0
        print(f"  Run {i+1}: TTFT={result['first_token_latency_ms']:.0f}ms, "
              f"mean inter-token={mean_inter:.1f}ms, "
              f"tokens={result['tokens_received']}")

    print(f"\n  --- Summary ---")
    if first_token_latencies:
        print(f"  Time to First Token (TTFT):")
        print(f"    Mean:   {statistics.mean(first_token_latencies):.1f}ms")
        print(f"    Median: {statistics.median(first_token_latencies):.1f}ms")

    if all_inter_token:
        print(f"\n  Inter-token latency (decode speed):")
        print(f"    Mean:   {statistics.mean(all_inter_token):.1f}ms")
        print(f"    Median: {statistics.median(all_inter_token):.1f}ms")
        print(f"    Min:    {min(all_inter_token):.1f}ms")
        print(f"    Max:    {max(all_inter_token):.1f}ms")

        mean_latency = statistics.mean(all_inter_token)
        throughput = 1000 / mean_latency if mean_latency > 0 else 0
        print(f"\n  Effective throughput: {throughput:.1f} tok/s")
        print(f"  (BitNet.cpp baseline: 26 tok/s)")
        print(f"  Gap: {26 - throughput:.1f} tok/s ({((26-throughput)/26)*100:.0f}%)")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("="*70)

    if all_inter_token:
        mean_latency = statistics.mean(all_inter_token)
        kernel_time = 3.34  # ms, from previous profiling
        overhead = mean_latency - kernel_time

        print(f"""
  Kernel time (measured):     {kernel_time:.1f}ms
  Total per-token latency:    {mean_latency:.1f}ms
  Framework overhead:         {overhead:.1f}ms ({overhead/mean_latency*100:.0f}%)

  To match BitNet.cpp (26 tok/s = 38ms/token), we need:
  - Target latency: 38ms
  - Current: {mean_latency:.0f}ms
  - Need to reduce: {mean_latency - 38:.0f}ms
""")


if __name__ == "__main__":
    main()
