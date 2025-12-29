#!/usr/bin/env python
"""Test sglang model directly without HTTP server."""

import sys
sys.path.insert(0, "/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/sglang-bitnet/python")

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("SGLANG DIRECT MODEL TEST (NO HTTP)")
print("="*70)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

# Load model using HuggingFace transformers
print("Loading model via transformers...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/bitnet-b1.58-2B-4T",
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
model.eval()

print(f"Model loaded: {model.config.num_hidden_layers} layers")

# Test generation
prompt = "Hello, how are you today?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(f"Prompt: {len(input_ids[0])} tokens")

# Warmup
print("Warming up...")
with torch.no_grad():
    _ = model.generate(
        input_ids,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Benchmark
print("\nBenchmarking generation...")
max_tokens = 30

start = time.perf_counter()
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
gen_time = time.perf_counter() - start

tokens_generated = len(output_ids[0]) - len(input_ids[0])
throughput = tokens_generated / gen_time

print(f"\nGenerated {tokens_generated} tokens in {gen_time:.2f}s")
print(f"Throughput: {throughput:.1f} tok/s")
print(f"Latency: {gen_time / tokens_generated * 1000:.1f}ms/token")

# Decode output
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\n{'='*70}")
print("OUTPUT")
print("="*70)
print(output)

print(f"\n{'='*70}")
print("SUMMARY")
print("="*70)
print(f"HuggingFace transformers: {throughput:.1f} tok/s")
print(f"sglang HTTP server:       ~16 tok/s")
print(f"Direct BitNet:            ~20 tok/s")
