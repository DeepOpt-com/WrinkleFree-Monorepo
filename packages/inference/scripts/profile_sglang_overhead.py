#!/usr/bin/env python
"""Profile sglang framework overhead to understand where 59ms/token is spent.

Kernels: 3.34ms/token → 300 tok/s theoretical
Actual:  62ms/token → 16 tok/s
Overhead: 59ms/token (94% of time)

This script profiles the major components of sglang to identify bottlenecks.
"""

import sys
sys.path.insert(0, "/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/sglang-bitnet/python")

import torch
import time
import json
import os
from transformers import AutoTokenizer
from safetensors.torch import load_file

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("PROFILING SGLANG FRAMEWORK OVERHEAD")
print("="*70)

# Load model config
model_path = "/home/lev/.cache/huggingface/hub/models--microsoft--BitNet-b1.58-2B-4T/snapshots"
snapshots = os.listdir(model_path)
latest = sorted(snapshots)[-1]
model_dir = os.path.join(model_path, latest)

with open(os.path.join(model_dir, "config.json")) as f:
    config = json.load(f)

n_layers = config["num_hidden_layers"]
hidden_dim = config["hidden_size"]
num_heads = config["num_attention_heads"]
head_dim = hidden_dim // num_heads

print(f"\nModel: {n_layers} layers, {hidden_dim} hidden, {num_heads} heads")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

# Create sample input
test_prompt = "Hello, how are you today?"
input_ids = tokenizer.encode(test_prompt, return_tensors="pt")[0]
print(f"Input: {len(input_ids)} tokens")

# Profile components
n_runs = 100

print("\n" + "="*70)
print("1. TOKENIZATION OVERHEAD")
print("="*70)

# Warmup
for _ in range(10):
    _ = tokenizer.encode("Hello, how are you today?")

start = time.perf_counter()
for _ in range(n_runs):
    _ = tokenizer.encode("Hello, how are you today?")
tokenize_time = (time.perf_counter() - start) / n_runs * 1000
print(f"Tokenization time: {tokenize_time:.3f}ms")

print("\n" + "="*70)
print("2. TENSOR CREATION OVERHEAD")
print("="*70)

# Pre-allocate sizes
seq_len = 50
batch_size = 1

# Measure tensor allocation times
start = time.perf_counter()
for _ in range(n_runs):
    x = torch.randn(batch_size, hidden_dim, dtype=torch.bfloat16)
alloc_time = (time.perf_counter() - start) / n_runs * 1000
print(f"Activation tensor allocation: {alloc_time:.4f}ms")

start = time.perf_counter()
for _ in range(n_runs):
    kv = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
kv_alloc_time = (time.perf_counter() - start) / n_runs * 1000
print(f"KV cache tensor allocation: {kv_alloc_time:.4f}ms")

# Per-layer allocation for 30 layers
per_token_alloc = (alloc_time + kv_alloc_time * 2) * n_layers
print(f"Per-token allocation (30 layers): {per_token_alloc:.2f}ms")

print("\n" + "="*70)
print("3. SCHEDULING OVERHEAD")
print("="*70)

# Simulate sglang's batch construction overhead
import torch.nn.functional as F

# Create tensors that would be used in batch management
req_pool_indices = torch.zeros(1, dtype=torch.int64)
seq_lens = torch.tensor([seq_len], dtype=torch.int64)
extend_prefix_lens = torch.zeros(1, dtype=torch.int64)
extend_seq_lens = torch.tensor([1], dtype=torch.int64)
req_to_token = torch.zeros(1, seq_len, dtype=torch.int64)

# Measure forward_batch info construction
start = time.perf_counter()
for _ in range(n_runs):
    # Simulating ForwardBatch construction
    _ = {
        'req_pool_indices': req_pool_indices.clone(),
        'seq_lens': seq_lens.clone(),
        'extend_prefix_lens': extend_prefix_lens.clone(),
        'extend_seq_lens': extend_seq_lens.clone(),
    }
batch_info_time = (time.perf_counter() - start) / n_runs * 1000
print(f"Batch info construction: {batch_info_time:.4f}ms")

# Per-layer scheduling overhead
per_token_batch = batch_info_time * n_layers
print(f"Per-token batch overhead (30 layers): {per_token_batch:.2f}ms")

print("\n" + "="*70)
print("4. PYTHON FUNCTION CALL OVERHEAD")
print("="*70)

def empty_function():
    pass

start = time.perf_counter()
for _ in range(n_runs * 100):
    empty_function()
call_time = (time.perf_counter() - start) / (n_runs * 100) * 1000
print(f"Empty function call: {call_time:.6f}ms")

# Each layer has many function calls
# Estimate: ~50 function calls per layer
calls_per_layer = 50
per_token_calls = call_time * calls_per_layer * n_layers
print(f"Per-token Python calls (est 50/layer): {per_token_calls:.2f}ms")

print("\n" + "="*70)
print("5. TYPE CONVERSION OVERHEAD")
print("="*70)

x_bf16 = torch.randn(1, hidden_dim, dtype=torch.bfloat16)
x_int8 = torch.randint(-127, 128, (hidden_dim,), dtype=torch.int8)

# Measure bfloat16 → float32
start = time.perf_counter()
for _ in range(n_runs):
    _ = x_bf16.float()
bf16_f32_time = (time.perf_counter() - start) / n_runs * 1000
print(f"bfloat16 → float32: {bf16_f32_time:.4f}ms")

# Measure int8 quantization
start = time.perf_counter()
for _ in range(n_runs):
    scale = x_bf16.abs().max().item() / 127.0
    _ = (x_bf16 / scale).to(torch.int8)
quant_time = (time.perf_counter() - start) / n_runs * 1000
print(f"Quantization (bfloat16 → int8): {quant_time:.4f}ms")

# Per-token type conversions (multiple per layer)
conversions_per_layer = 4  # input quant, output dequant, etc.
per_token_conv = quant_time * conversions_per_layer * n_layers
print(f"Per-token conversions (4/layer): {per_token_conv:.2f}ms")

print("\n" + "="*70)
print("6. KV CACHE MANAGEMENT OVERHEAD")
print("="*70)

# Simulate KV cache indexing
max_tokens = 4096
k_cache = torch.randn(max_tokens, num_heads, head_dim, dtype=torch.bfloat16)
v_cache = torch.randn(max_tokens, num_heads, head_dim, dtype=torch.bfloat16)
token_indices = torch.randint(0, max_tokens, (seq_len,))

# Python indexing (slow path)
start = time.perf_counter()
for _ in range(n_runs):
    k = k_cache[token_indices]
    v = v_cache[token_indices]
py_gather_time = (time.perf_counter() - start) / n_runs * 1000
print(f"KV gather (Python): {py_gather_time:.4f}ms")

# Per-token KV management
per_token_kv = py_gather_time * n_layers
print(f"Per-token KV management (30 layers): {per_token_kv:.2f}ms")

print("\n" + "="*70)
print("7. TENSOR VIEW/RESHAPE OVERHEAD")
print("="*70)

q = torch.randn(1, hidden_dim, dtype=torch.bfloat16)

start = time.perf_counter()
for _ in range(n_runs):
    q_reshaped = q.view(1, num_heads, head_dim).unsqueeze(2)
    _ = q_reshaped.squeeze(2).view(1, -1)
reshape_time = (time.perf_counter() - start) / n_runs * 1000
print(f"View/reshape operations: {reshape_time:.4f}ms")

# Per-token reshapes
reshapes_per_layer = 6
per_token_reshape = reshape_time * reshapes_per_layer * n_layers
print(f"Per-token reshapes (6/layer): {per_token_reshape:.2f}ms")

print("\n" + "="*70)
print("SUMMARY: ESTIMATED OVERHEAD BREAKDOWN")
print("="*70)

total_overhead = (
    per_token_alloc +
    per_token_batch +
    per_token_calls +
    per_token_conv +
    per_token_kv +
    per_token_reshape
)

print(f"""
Component                    Time/token
-----------------------------------------
Tensor allocation:           {per_token_alloc:>6.2f}ms
Batch info construction:     {per_token_batch:>6.2f}ms
Python function calls:       {per_token_calls:>6.2f}ms
Type conversions:            {per_token_conv:>6.2f}ms
KV cache management:         {per_token_kv:>6.2f}ms
Tensor reshapes:             {per_token_reshape:>6.2f}ms
-----------------------------------------
ESTIMATED TOTAL:             {total_overhead:>6.2f}ms

Kernel time (measured):       3.34ms
Actual overhead (measured):  ~59ms
Estimated overhead:          {total_overhead:>5.2f}ms
Unexplained gap:             {59 - total_overhead:.2f}ms
""")

print("="*70)
print("ANALYSIS")
print("="*70)

if total_overhead < 59:
    print(f"""
The unexplained {59 - total_overhead:.0f}ms is likely from:
- HTTP server overhead (request parsing, response formatting)
- sglang scheduler (radix tree, prefix caching, batching logic)
- Token sampling and detokenization
- Memory pool management
- Inter-process communication
- GIL contention

To achieve theoretical 300 tok/s, we need to eliminate this framework overhead.
Options:
1. Direct inference without sglang HTTP server
2. Custom inference loop with pre-allocated tensors
3. BitNet.cpp (if model conversion issues are resolved)
4. Torch.compile on entire forward pass
""")
