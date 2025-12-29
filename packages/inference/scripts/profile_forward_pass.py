#!/usr/bin/env python
"""Profile the actual forward pass overhead in HuggingFace BitNet.

This directly loads the model and profiles the forward pass,
bypassing HTTP overhead to isolate framework overhead.
"""

import sys
sys.path.insert(0, "/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/sglang-bitnet/python")

import torch
import time
import json
import os

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("FORWARD PASS OVERHEAD PROFILER")
print("="*70)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_name = "microsoft/bitnet-b1.58-2B-4T"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

print(f"Model: {model_name}")
print(f"Layers: {config.num_hidden_layers}")
print(f"Hidden: {config.hidden_size}")

print("\nLoading model (HuggingFace)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
model.eval()

# Prepare input
prompt = "Hello, how are you?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(f"Input tokens: {input_ids.shape[1]}")

# Warmup
print("\nWarming up...")
with torch.no_grad():
    for _ in range(3):
        _ = model.generate(input_ids, max_new_tokens=1, do_sample=False)

# Profile generation
print("\nProfiling forward pass (no HTTP, no scheduler)...")
n_tokens = 20
n_runs = 5

all_times = []
for run in range(n_runs):
    gen_times = []
    with torch.no_grad():
        generated = input_ids.clone()
        for i in range(n_tokens):
            start = time.perf_counter()
            outputs = model(generated)
            next_token = outputs.logits[0, -1:].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            elapsed = (time.perf_counter() - start) * 1000
            gen_times.append(elapsed)

    mean_time = sum(gen_times[1:]) / len(gen_times[1:])  # Skip first (prefill)
    all_times.extend(gen_times[1:])
    print(f"  Run {run+1}: {mean_time:.1f}ms/token (prefill: {gen_times[0]:.1f}ms)")

mean_latency = sum(all_times) / len(all_times)
throughput = 1000 / mean_latency

print(f"\n{'='*70}")
print("RESULTS (Pure Forward Pass - No HTTP/Scheduler)")
print("="*70)
print(f"  Mean latency:  {mean_latency:.1f}ms/token")
print(f"  Throughput:    {throughput:.1f} tok/s")
print(f"")
print(f"  Comparison:")
print(f"    BitNet.cpp:        26.0 tok/s (38.5ms/token)")
print(f"    SGLang server:     19.2 tok/s (52.2ms/token)")
print(f"    HF pure forward:   {throughput:.1f} tok/s ({mean_latency:.1f}ms/token)")
print(f"")

# Calculate overhead components
kernel_time = 3.3  # ms, measured
python_overhead = mean_latency - kernel_time
http_overhead = 52.2 - mean_latency  # Difference from server

print(f"  Overhead breakdown (estimated):")
print(f"    Kernel execution:   {kernel_time:.1f}ms")
print(f"    Python/framework:   {python_overhead:.1f}ms ({python_overhead/mean_latency*100:.0f}%)")
if http_overhead > 0:
    print(f"    HTTP/scheduler:     {http_overhead:.1f}ms (from server measurement)")

print(f"\n{'='*70}")
print("ANALYSIS")
print("="*70)

if mean_latency < 52.2:
    print(f"""
  The pure forward pass ({mean_latency:.1f}ms) is faster than SGLang server (52.2ms).
  This confirms HTTP/scheduler adds {http_overhead:.1f}ms overhead.

  To match BitNet.cpp (38.5ms), we need to reduce:
  - Python forward pass: {mean_latency - 38.5:.1f}ms (type conversions, allocations)
  - Or use BitNet.cpp backend directly
""")
else:
    print(f"""
  The pure forward pass ({mean_latency:.1f}ms) is similar to server (52.2ms).
  The overhead is primarily in the model forward pass, not HTTP.

  Key optimizations needed:
  - Reduce type conversions (float32 <-> bfloat16 <-> int8)
  - Pre-allocate intermediate tensors
  - Use fused operations
""")
