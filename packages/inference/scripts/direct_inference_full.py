#!/usr/bin/env python
"""Full direct BitNet inference bypassing sglang HTTP server.

This implements the complete forward pass with autoregressive generation,
measuring actual throughput without framework overhead.

Target: ~100 tok/s (vs sglang's 16 tok/s)
"""

import sys
sys.path.insert(0, "/home/lev/code/WrinkleFree/WrinkleFree-Inference-Engine/extern/sglang-bitnet/python")

import torch
import torch.nn.functional as F
import time
import json
import os

from transformers import AutoTokenizer
from safetensors.torch import load_file
from sglang.srt.models.bitnet import _unpack_ternary_weights, _pack_ternary_weights
from sgl_kernel.quantization.bitnet import bitnet_gemv, quantize_activations_i8

import warnings
warnings.filterwarnings("ignore")


class RotaryEmbedding:
    """Rotary Position Embedding."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 500000.0):
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.inv_freq = inv_freq

        # Pre-compute sin/cos cache
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, head_dim/2]
        self.cos_cache = torch.cos(freqs).to(torch.bfloat16)
        self.sin_cache = torch.sin(freqs).to(torch.bfloat16)

    def apply(self, q, k, pos):
        """Apply rotary embedding to q and k at given position."""
        cos = self.cos_cache[pos]  # [head_dim/2]
        sin = self.sin_cache[pos]  # [head_dim/2]

        # Apply to Q
        q = self._rotate(q, cos, sin)
        # Apply to K
        k = self._rotate(k, cos, sin)

        return q, k

    def _rotate(self, x, cos, sin):
        """Apply rotary embedding to tensor x with shape [n_heads, head_dim]."""
        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]

        # Rotate
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return rotated


class DirectBitNetInference:
    """Direct BitNet inference without sglang framework."""

    def __init__(self, model_path: str):
        self.device = "cpu"
        self.dtype = torch.bfloat16

        # Load config
        with open(os.path.join(model_path, "config.json")) as f:
            self.config = json.load(f)

        self.hidden_dim = self.config["hidden_size"]  # 2560
        self.n_layers = self.config["num_hidden_layers"]  # 30
        self.n_heads = self.config["num_attention_heads"]  # 20
        self.n_kv_heads = self.config.get("num_key_value_heads", self.n_heads)  # 4 for GQA
        self.head_dim = self.hidden_dim // self.n_heads  # 128
        self.mlp_hidden = self.config["intermediate_size"]  # 6912
        self.vocab_size = self.config["vocab_size"]  # 152064
        self.rms_eps = self.config.get("rms_norm_eps", 1e-5)
        self.rope_theta = self.config.get("rope_theta", 500000.0)

        print(f"Model: {self.n_layers}L, {self.hidden_dim}H, {self.n_heads}Q/{self.n_kv_heads}KV")
        print(f"MLP hidden: {self.mlp_hidden}, Vocab: {self.vocab_size}")

        # Initialize RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=4096, base=self.rope_theta)

        # Load weights
        print("Loading weights...")
        self.weights = load_file(os.path.join(model_path, "model.safetensors"))
        print(f"Loaded {len(self.weights)} tensors")

        # Pre-process weights
        self._preprocess_weights()

        # Pre-allocate KV cache
        self.max_seq_len = 4096
        self._init_kv_cache()

    def _repack_hf_weight(self, hf_weight):
        """Convert HF weight format [out/4, in] to kernel format [out, in/4]."""
        unpacked = _unpack_ternary_weights(hf_weight)
        packed, _ = _pack_ternary_weights(unpacked)
        return packed

    def _preprocess_weights(self):
        """Repack all weights to kernel format."""
        print("Repacking weights to kernel format...")

        # Embedding (keep as-is, it's bfloat16)
        self.embed_tokens = self.weights["model.embed_tokens.weight"].to(self.dtype)

        # Output projection - uses tied embeddings (same as embed_tokens)
        # lm_head.weight doesn't exist, we reuse embed_tokens
        self.lm_head = self.embed_tokens  # Weight tying

        # Final norm
        self.final_norm = self.weights["model.norm.weight"].to(self.dtype)

        # Per-layer weights
        self.layers = []
        for i in range(self.n_layers):
            prefix = f"model.layers.{i}"
            layer = {
                # Attention norms (including sub-norm)
                "input_layernorm": self.weights[f"{prefix}.input_layernorm.weight"].to(self.dtype),
                "post_attention_layernorm": self.weights[f"{prefix}.post_attention_layernorm.weight"].to(self.dtype),
                "attn_sub_norm": self.weights[f"{prefix}.self_attn.attn_sub_norm.weight"].to(self.dtype),
                "ffn_sub_norm": self.weights[f"{prefix}.mlp.ffn_sub_norm.weight"].to(self.dtype),

                # QKV projections (packed)
                "q_proj": self._repack_hf_weight(self.weights[f"{prefix}.self_attn.q_proj.weight"]),
                "q_scale": self.weights[f"{prefix}.self_attn.q_proj.weight_scale"].item(),
                "k_proj": self._repack_hf_weight(self.weights[f"{prefix}.self_attn.k_proj.weight"]),
                "k_scale": self.weights[f"{prefix}.self_attn.k_proj.weight_scale"].item(),
                "v_proj": self._repack_hf_weight(self.weights[f"{prefix}.self_attn.v_proj.weight"]),
                "v_scale": self.weights[f"{prefix}.self_attn.v_proj.weight_scale"].item(),
                "o_proj": self._repack_hf_weight(self.weights[f"{prefix}.self_attn.o_proj.weight"]),
                "o_scale": self.weights[f"{prefix}.self_attn.o_proj.weight_scale"].item(),

                # MLP (packed)
                "gate_proj": self._repack_hf_weight(self.weights[f"{prefix}.mlp.gate_proj.weight"]),
                "gate_scale": self.weights[f"{prefix}.mlp.gate_proj.weight_scale"].item(),
                "up_proj": self._repack_hf_weight(self.weights[f"{prefix}.mlp.up_proj.weight"]),
                "up_scale": self.weights[f"{prefix}.mlp.up_proj.weight_scale"].item(),
                "down_proj": self._repack_hf_weight(self.weights[f"{prefix}.mlp.down_proj.weight"]),
                "down_scale": self.weights[f"{prefix}.mlp.down_proj.weight_scale"].item(),
            }
            self.layers.append(layer)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{self.n_layers} layers")

        # Free original weights to save memory
        del self.weights
        print("Weights preprocessed")

    def _init_kv_cache(self):
        """Initialize KV cache for all layers."""
        self.k_cache = torch.zeros(
            self.n_layers, self.max_seq_len, self.n_kv_heads, self.head_dim,
            dtype=self.dtype
        )
        self.v_cache = torch.zeros(
            self.n_layers, self.max_seq_len, self.n_kv_heads, self.head_dim,
            dtype=self.dtype
        )
        print(f"KV cache allocated: {self.k_cache.nbytes / 1e6:.1f}MB per K/V")

    def rms_norm(self, x, weight):
        """RMS normalization."""
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.rms_eps).to(x.dtype)) * weight

    def bitnet_linear(self, x, weight, scale):
        """BitNet linear using GEMV kernel with encoding correction.

        The kernel computes: weight_scale * sum((weight + 1) * activation)
        where (weight + 1) is the encoded value {0, 1, 2} for {-1, 0, +1}
        So: kernel_out = weight_scale * (sum(weight * act) + sum(act))
        We want: weight_scale * sum(weight * act)
        Therefore: correct = kernel_out - weight_scale * sum(act)
        """
        # Quantize input to int8
        x_int8, act_scale = quantize_activations_i8(x.float())
        # GEMV (handles batch dim internally)
        if x_int8.dim() > 1:
            x_int8 = x_int8.squeeze(0)

        # Kernel call
        out = bitnet_gemv(weight, x_int8, scale)

        # Correction: subtract weight_scale * sum(activations) for encoding offset
        act_sum = x_int8.float().sum().item()
        out = (out - scale * act_sum) * act_scale

        return out.to(self.dtype)

    def attention(self, hidden, layer_idx, pos, layer):
        """Single-head attention with KV cache."""
        # QKV projections
        q = self.bitnet_linear(hidden, layer["q_proj"], layer["q_scale"])
        k = self.bitnet_linear(hidden, layer["k_proj"], layer["k_scale"])
        v = self.bitnet_linear(hidden, layer["v_proj"], layer["v_scale"])

        # Reshape for attention
        q = q.view(self.n_heads, self.head_dim)  # [n_heads, head_dim]
        k = k.view(self.n_kv_heads, self.head_dim)  # [n_kv_heads, head_dim]
        v = v.view(self.n_kv_heads, self.head_dim)  # [n_kv_heads, head_dim]

        # Apply RoPE to Q and K
        q, k = self.rope.apply(q, k, pos)

        # Store in KV cache (after RoPE)
        self.k_cache[layer_idx, pos] = k
        self.v_cache[layer_idx, pos] = v

        # Get cached K/V up to current position
        k_seq = self.k_cache[layer_idx, :pos + 1]  # [seq, n_kv_heads, head_dim]
        v_seq = self.v_cache[layer_idx, :pos + 1]  # [seq, n_kv_heads, head_dim]

        # Reshape for SDPA with GQA
        # Q: [1, n_heads, 1, head_dim]
        # K/V: [1, n_kv_heads, seq, head_dim]
        q_attn = q.unsqueeze(0).unsqueeze(2)  # [1, n_heads, 1, head_dim]
        k_attn = k_seq.transpose(0, 1).unsqueeze(0)  # [1, n_kv_heads, seq, head_dim]
        v_attn = v_seq.transpose(0, 1).unsqueeze(0)  # [1, n_kv_heads, seq, head_dim]

        # SDPA with GQA
        attn_out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, enable_gqa=True)

        # Reshape back [1, n_heads, 1, head_dim] -> [hidden_dim]
        attn_out = attn_out.squeeze(0).squeeze(1).flatten()

        # Apply attention sub-norm before output projection (BitNet-specific)
        attn_out = self.rms_norm(attn_out, layer["attn_sub_norm"])

        # Output projection
        out = self.bitnet_linear(attn_out.unsqueeze(0), layer["o_proj"], layer["o_scale"])
        return out

    def mlp(self, hidden, layer):
        """MLP with ReLU² activation (BitNet-specific, NOT SiLU!)."""
        gate = self.bitnet_linear(hidden, layer["gate_proj"], layer["gate_scale"])
        up = self.bitnet_linear(hidden, layer["up_proj"], layer["up_scale"])

        # ReLU² activation: relu(gate)^2 * up (NOT SiLU!)
        hidden_act = torch.relu(gate.to(self.dtype)).square() * up.to(self.dtype)

        # Apply FFN sub-norm before down projection (BitNet-specific)
        hidden_act = self.rms_norm(hidden_act, layer["ffn_sub_norm"])

        down = self.bitnet_linear(hidden_act.unsqueeze(0), layer["down_proj"], layer["down_scale"])
        return down

    def forward_layer(self, hidden, layer_idx, pos):
        """Forward pass through one transformer layer."""
        layer = self.layers[layer_idx]

        # Pre-attention norm
        normed = self.rms_norm(hidden, layer["input_layernorm"])

        # Attention with residual
        attn_out = self.attention(normed, layer_idx, pos, layer)
        hidden = hidden + attn_out

        # Pre-MLP norm
        normed = self.rms_norm(hidden, layer["post_attention_layernorm"])

        # MLP with residual
        mlp_out = self.mlp(normed, layer)
        hidden = hidden + mlp_out

        return hidden

    def forward(self, token_id: int, pos: int):
        """Full forward pass for one token."""
        # Embedding
        hidden = self.embed_tokens[token_id]  # [hidden_dim]

        # All layers
        for i in range(self.n_layers):
            hidden = self.forward_layer(hidden, i, pos)

        # Final norm
        hidden = self.rms_norm(hidden, self.final_norm)

        # LM head (dense matmul, not BitNet)
        logits = F.linear(hidden, self.lm_head)

        return logits

    def sample(self, logits, temperature=0.7, top_p=0.9):
        """Sample from logits."""
        if temperature == 0:
            return logits.argmax().item()

        # Apply temperature
        logits = logits / temperature

        # Top-p sampling
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff
        cutoff_idx = (cumsum > top_p).nonzero()
        if len(cutoff_idx) > 0:
            cutoff = cutoff_idx[0].item()
            sorted_probs[cutoff + 1:] = 0

        # Renormalize and sample
        sorted_probs = sorted_probs / sorted_probs.sum()
        token = torch.multinomial(sorted_probs, 1).item()
        return sorted_indices[token].item()

    def generate(self, prompt: str, max_tokens: int = 50, tokenizer=None):
        """Generate tokens autoregressively."""
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
        generated = input_ids.tolist()

        print(f"Prompt: {len(input_ids)} tokens")

        # Prefill
        print("Prefilling...")
        prefill_start = time.perf_counter()
        for pos, token_id in enumerate(input_ids):
            _ = self.forward(token_id.item(), pos)
        prefill_time = time.perf_counter() - prefill_start
        print(f"Prefill: {prefill_time:.2f}s ({len(input_ids) / prefill_time:.1f} tok/s)")

        # Generate
        print(f"Generating {max_tokens} tokens...")
        gen_start = time.perf_counter()

        pos = len(input_ids)
        next_token = generated[-1]

        for i in range(max_tokens):
            logits = self.forward(next_token, pos)
            next_token = self.sample(logits)
            generated.append(next_token)
            pos += 1

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                break

        gen_time = time.perf_counter() - gen_start
        tokens_generated = pos - len(input_ids)
        throughput = tokens_generated / gen_time

        print(f"\nGenerated {tokens_generated} tokens in {gen_time:.2f}s")
        print(f"Throughput: {throughput:.1f} tok/s")
        print(f"Latency: {gen_time / tokens_generated * 1000:.1f}ms/token")

        # Decode
        output = tokenizer.decode(generated, skip_special_tokens=True)
        return output, throughput


def main():
    print("="*70)
    print("DIRECT BITNET INFERENCE (NO SGLANG)")
    print("="*70)

    # Find model
    model_base = "/home/lev/.cache/huggingface/hub/models--microsoft--BitNet-b1.58-2B-4T/snapshots"
    snapshots = os.listdir(model_base)
    model_path = os.path.join(model_base, sorted(snapshots)[-1])

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

    # Initialize model
    model = DirectBitNetInference(model_path)

    # Try torch.compile
    print("\nCompiling forward pass with torch.compile...")
    try:
        model.forward = torch.compile(model.forward, mode="reduce-overhead")
        print("torch.compile applied successfully")
    except Exception as e:
        print(f"torch.compile failed: {e}")

    # Warmup (more iterations for compile warmup)
    print("\nWarming up (extra iterations for compile)...")
    for i in range(20):
        _ = model.forward(1, 0)
        _ = model.forward(2, 1)
    print("Warmup complete")

    # Test generation
    print("\n" + "="*70)
    print("GENERATION TEST")
    print("="*70)

    prompt = "Hello, how are you today?"
    output, throughput = model.generate(prompt, max_tokens=30, tokenizer=tokenizer)

    print("\n" + "="*70)
    print("OUTPUT")
    print("="*70)
    print(output)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Direct inference: {throughput:.1f} tok/s")
    print(f"sglang baseline:  ~16 tok/s")
    print(f"Speedup:          {throughput / 16:.1f}x")
    print(f"BitNet.cpp target: ~47 tok/s")

    return throughput


if __name__ == "__main__":
    main()
