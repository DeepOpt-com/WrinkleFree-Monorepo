#!/usr/bin/env python3
"""Native BitNet inference server using sgl-kernel TL2 kernels.

This server uses packed 1.58-bit weights with sgl-kernel's optimized SIMD
kernels (AVX2/AVX512) for maximum CPU throughput.

Performance: ~29 tok/s on GCP c3d-standard-32 (AMD EPYC Genoa with AVX512)

Key optimizations:
    - bitnet_gemv for single-token decode (8x faster than gemm for M=1)
    - Greedy decoding by default (eliminates sampling overhead)
    - Repetition penalty to reduce output loops
    - KV cache for efficient autoregressive generation
    - SIMD block-interleaved weight packing (128-element blocks)

Usage:
    # Step 1: Convert checkpoint to packed format (one-time)
    python scripts/convert_to_sglkernel.py models/my-checkpoint models/my-checkpoint.bin

    # Step 2: Start server
    python scripts/serve_bitnet_native.py \\
        --model models/my-checkpoint.bin \\
        --tokenizer models/my-checkpoint

    # Step 3: Test
    curl http://localhost:30000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

Requirements:
    - sgl-kernel with BitNet support (run scripts/setup-cpu.sh)
    - Converted .bin file (from convert_to_sglkernel.py) + tokenizer directory

API Endpoints:
    GET  /health              - Health check (returns {"status": "healthy"})
    POST /generate            - Raw text generation
    POST /v1/chat/completions - OpenAI-compatible chat API
"""

import argparse
import json
import logging
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request
from safetensors import safe_open
from transformers import AutoTokenizer, PretrainedConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Import sgl-kernel BitNet operations
try:
    from sgl_kernel.quantization.bitnet import (
        bitnet_gemm,
        bitnet_gemv,
        quantize_activations_i8,
        check_kernel_available,
    )
    NATIVE_KERNEL = check_kernel_available()
except ImportError:
    NATIVE_KERNEL = False
    logger.warning("sgl-kernel BitNet not available, using fallback")


# sgl-kernel binary format constants
MAGIC = b"SGLBITNT"
QK_I2_S = 128  # SIMD block size


@dataclass
class BitNetConfig:
    """BitNet model configuration."""
    vocab_size: int = 128256
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    head_dim: int = 128
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    bos_token_id: int = 128000
    eos_token_id: int = 128001

    @classmethod
    def from_json(cls, path: str) -> "BitNetConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, data: dict) -> "BitNetConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_sglkernel_binary(path: Path) -> Tuple[BitNetConfig, dict]:
    """Load model from sgl-kernel binary format (.bin file).

    Returns:
        config: BitNetConfig
        tensors: dict mapping name -> {'tensor': tensor, 'scale': float or None}
    """
    logger.info(f"Loading sgl-kernel binary: {path}")

    tensors = {}

    with open(path, 'rb') as f:
        # Magic
        magic = f.read(8)
        assert magic == MAGIC, f"Invalid magic: {magic}"

        # Version
        version = struct.unpack('<I', f.read(4))[0]
        assert version == 1, f"Unsupported version: {version}"

        # Config
        config_len = struct.unpack('<I', f.read(4))[0]
        config_json = json.loads(f.read(config_len).decode('utf-8'))
        config = BitNetConfig.from_dict(config_json)

        # Tensors
        num_tensors = struct.unpack('<I', f.read(4))[0]
        logger.info(f"Loading {num_tensors} tensors...")

        for i in range(num_tensors):
            # Name
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            # Dtype
            dtype_id = struct.unpack('<I', f.read(4))[0]
            dtype_map = {0: torch.uint8, 1: torch.float32, 2: torch.float16, 3: torch.bfloat16}
            dtype = dtype_map[dtype_id]

            # Shape
            ndims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]

            # Scale
            has_scale = struct.unpack('<I', f.read(4))[0]
            scale = struct.unpack('<f', f.read(4))[0] if has_scale else None

            # Data
            data_size = struct.unpack('<Q', f.read(8))[0]
            data = f.read(data_size)

            # Convert to tensor
            tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
            tensors[name] = {'tensor': tensor.clone(), 'scale': scale}

            if (i + 1) % 50 == 0:
                logger.info(f"  Loaded {i + 1}/{num_tensors} tensors")

    logger.info(f"Model loaded: {len(tensors)} tensors")
    return config, tensors


class RMSNorm(nn.Module):
    """RMS normalization layer."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class BitLinear(nn.Module):
    """Linear layer using packed 1.58-bit weights with native kernels.

    Weights are packed in SIMD block-interleaved format (from convert_to_sglkernel.py):
    - Block size: 128 elements (QK_I2_S)
    - 32 packed bytes per 128-element block
    - byte[j].bits[6:7] = weight[j+0]
    - byte[j].bits[4:5] = weight[j+32]
    - byte[j].bits[2:3] = weight[j+64]
    - byte[j].bits[0:1] = weight[j+96]
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Packed weights: 4 weights per byte, SIMD block-interleaved
        packed_in = in_features // 4
        self.weight = nn.Parameter(
            torch.zeros(out_features, packed_in, dtype=torch.uint8),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic kernel selection.

        Uses bitnet_gemv for single-token decode (8x faster) and
        bitnet_gemm for prefill/batch processing.
        """
        # Flatten batch dimensions
        orig_shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        if NATIVE_KERNEL:
            # Use optimized SIMD kernel (expects SIMD block-interleaved packing)
            scale = self.weight_scale.item()

            if x_flat.shape[0] == 1:
                # Single token decode: use gemv (8x faster than gemm)
                x_int8, act_scale = quantize_activations_i8(x_flat.squeeze(0).float())
                out = bitnet_gemv(self.weight, x_int8, scale)
                out = out.unsqueeze(0)
            else:
                # Prefill/batch: use gemm
                x_int8, act_scale = quantize_activations_i8(x_flat.float())
                out = bitnet_gemm(self.weight, x_int8, scale)

            out = (out * act_scale).to(x.dtype)
        else:
            # Fallback: unpack weights (slow)
            out = self._fallback_forward(x_flat)

        logger.debug(f"BitLinear: reshape {out.shape} -> {orig_shape} + (-1,)")
        return out.view(*orig_shape, -1)

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback using weight unpacking (slower)."""
        weight = self._unpack_weights_simd()
        return F.linear(x, weight * self.weight_scale.item())

    def _unpack_weights_simd(self) -> torch.Tensor:
        """Unpack SIMD block-interleaved 2-bit weights to float tensor."""
        packed = self.weight
        M, K_packed = packed.shape
        K = K_packed * 4

        out = torch.zeros(M, K, dtype=torch.float32)

        # SIMD block-interleaved unpacking
        # For each 128-element block in K: 32 packed bytes
        num_blocks = K // QK_I2_S
        for block_idx in range(num_blocks):
            base_w = block_idx * QK_I2_S  # Start of 128-element block in output
            base_p = block_idx * 32       # Start of 32-byte block in packed

            for j in range(32):
                byte_val = packed[:, base_p + j]
                # Unpack 4 weights from positions j, j+32, j+64, j+96 within block
                out[:, base_w + j + 0] = ((byte_val >> 6) & 0x03).float() - 1.0
                out[:, base_w + j + 32] = ((byte_val >> 4) & 0x03).float() - 1.0
                out[:, base_w + j + 64] = ((byte_val >> 2) & 0x03).float() - 1.0
                out[:, base_w + j + 96] = ((byte_val >> 0) & 0x03).float() - 1.0

        return out


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 500000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BitNetAttention(nn.Module):
    """Multi-head attention with BitNet linear layers."""

    def __init__(self, config: BitNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = BitLinear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = BitLinear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = BitLinear(self.num_heads * self.head_dim, self.hidden_size)

        self.attn_sub_norm = RMSNorm(self.num_heads * self.head_dim)
        self.rotary_emb = RotaryEmbedding(self.head_dim, base=config.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        logger.debug(f"Attention input shape: {hidden_states.shape}, ndim={hidden_states.ndim}")
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected 3D hidden_states, got {hidden_states.ndim}D: {hidden_states.shape}")
        bsz, seq_len, _ = hidden_states.shape

        # Q/K/V projections (no norm before - that's in the decoder layer)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Rotary embeddings
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)

        # GQA expansion
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and normalize (attn_sub_norm is AFTER attention, BEFORE o_proj)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.attn_sub_norm(attn_output)  # BitNet-specific: norm before o_proj
        logger.debug(f"Attention o_proj input: {attn_output.shape}")
        output = self.o_proj(attn_output)
        logger.debug(f"Attention output: {output.shape}")
        return output, past_key_value


class BitNetMLP(nn.Module):
    """MLP with BitNet linear layers and ReLU^2."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size)
        self.ffn_sub_norm = RMSNorm(config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # ReLU^2 activation
        hidden = F.relu(gate).pow(2) * up
        hidden = self.ffn_sub_norm(hidden)
        return self.down_proj(hidden)


class BitNetDecoderLayer(nn.Module):
    """Single BitNet transformer layer."""

    def __init__(self, config: BitNetConfig, layer_idx: int):
        super().__init__()
        self.self_attn = BitNetAttention(config, layer_idx)
        self.mlp = BitNetMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        logger.debug(f"DecoderLayer input shape: {hidden_states.shape}")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        logger.debug(f"After input_layernorm: {hidden_states.shape}")
        hidden_states, past_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value
        )
        logger.debug(f"After attention: {hidden_states.shape}")
        hidden_states = residual + hidden_states
        logger.debug(f"After attn residual: {hidden_states.shape}")

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        logger.debug(f"After post_attn_ln: {hidden_states.shape}")
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class BitNetModel(nn.Module):
    """BitNet transformer model using native kernels."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            BitNetDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        hidden_states = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            hidden_states, past_kv = layer(
                hidden_states, attention_mask, past_key_value=past_key_values[i]
            )
            new_past_key_values.append(past_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_past_key_values


class BitNetForCausalLM(nn.Module):
    """BitNet for causal language modeling."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.model = BitNetModel(config)
        # Tied embeddings
        self.lm_head_weight = None  # Will be set to embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        hidden_states, past_key_values = self.model(
            input_ids, attention_mask, past_key_values
        )
        # Tied embeddings
        logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        return logits, past_key_values

    @classmethod
    def from_pretrained(cls, model_path: str) -> "BitNetForCausalLM":
        """Load model from safetensors checkpoint or sgl-kernel binary.

        Args:
            model_path: Either:
                - Path to .bin file (sgl-kernel format from convert_to_sglkernel.py)
                - Path to directory with config.json + model.safetensors
        """
        model_path = Path(model_path)

        # Check if it's a .bin file (sgl-kernel format)
        if model_path.suffix == '.bin' or (model_path.is_file() and model_path.name.endswith('.bin')):
            return cls._from_sglkernel_binary(model_path)
        else:
            return cls._from_safetensors(model_path)

    @classmethod
    def _from_sglkernel_binary(cls, bin_path: Path) -> "BitNetForCausalLM":
        """Load model from sgl-kernel binary format."""
        config, tensors = load_sglkernel_binary(bin_path)
        model = cls(config)

        def get_module_by_path(root, path: str):
            """Get module by dot-separated path, handling numeric indices for ModuleList."""
            parts = path.split('.')
            current = root
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            return current

        # Map tensors to model
        loaded = 0
        scales_set = 0
        for name, param in model.named_parameters():
            key = name
            if key in tensors:
                t = tensors[key]
                param.data.copy_(t['tensor'])
                loaded += 1

                # If tensor has a scale, set it on the parent BitLinear
                if t['scale'] is not None and name.endswith('.weight'):
                    parent_path = name.rsplit('.', 1)[0]
                    try:
                        parent = get_module_by_path(model, parent_path)
                        if hasattr(parent, 'weight_scale'):
                            parent.weight_scale.data.fill_(t['scale'])
                            scales_set += 1
                    except (AttributeError, IndexError, KeyError) as e:
                        logger.warning(f"Could not set scale for {name}: {e}")
            else:
                logger.debug(f"Missing key: {key}")

        logger.info(f"Loaded {loaded} parameters, set {scales_set} scales, native kernels: {NATIVE_KERNEL}")
        return model

    @classmethod
    def _from_safetensors(cls, model_dir: Path) -> "BitNetForCausalLM":
        """Load model from safetensors checkpoint."""
        config_path = model_dir / "config.json"
        config = BitNetConfig.from_json(str(config_path))

        model = cls(config)

        # Load weights
        weights_path = model_dir / "model.safetensors"
        logger.info(f"Loading weights from {weights_path}")

        with safe_open(str(weights_path), framework="pt") as f:
            state_dict = {}
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        # Map weights to model
        model._load_weights_safetensors(state_dict)
        logger.info(f"Model loaded with native kernels: {NATIVE_KERNEL}")

        return model

    def _load_weights_safetensors(self, state_dict: dict):
        """Load weights from safetensors state dict."""
        for name, param in self.named_parameters():
            key = name
            if key in state_dict:
                param.data.copy_(state_dict[key])
            else:
                logger.warning(f"Missing key: {key}")


# Flask app
app = Flask(__name__)
model = None
tokenizer = None
model_lock = Lock()


def load_model(model_path: str, tokenizer_path: str = None):
    """Load BitNet model and tokenizer.

    Args:
        model_path: Path to model (.bin file or directory)
        tokenizer_path: Path to tokenizer (defaults to model_path for directories,
                        required for .bin files)
    """
    global model, tokenizer

    # Determine tokenizer path
    if tokenizer_path is None:
        if model_path.endswith('.bin'):
            # For .bin files, try to find tokenizer in same directory or parent
            bin_path = Path(model_path)
            # Try: models/dlm-bitnet-2b.bin -> models/dlm-bitnet-2b/
            candidate = bin_path.parent / bin_path.stem
            if candidate.exists() and (candidate / 'tokenizer.json').exists():
                tokenizer_path = str(candidate)
            else:
                raise ValueError(f"Cannot find tokenizer for {model_path}. "
                                 f"Please specify --tokenizer path.")
        else:
            tokenizer_path = model_path

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading BitNet model from {model_path}")
    model = BitNetForCausalLM.from_pretrained(model_path)
    model.eval()


@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate text completion.

    Args:
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate (default: 128)
        temperature: Sampling temperature. 0.0 = greedy (fastest, default).
                     Higher values = more random.
        top_p: Top-p (nucleus) sampling. Only used if temperature > 0.
        repetition_penalty: Penalty for repeated tokens. 1.0 = no penalty,
                            >1.0 = discourage repetition (default: 1.2)

    Returns:
        Generated text (excluding the prompt)

    Performance:
        - Greedy (temperature=0): ~29 tok/s
        - Sampling (temperature>0): ~21 tok/s
    """
    with model_lock:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        generated = []
        past_key_values = None

        for _ in range(max_new_tokens):
            logits, past_key_values = model(input_ids, past_key_values=past_key_values)

            # Sample next token
            next_logits = logits[:, -1, :]
            
            # Apply repetition penalty to already-generated tokens
            if repetition_penalty != 1.0 and len(generated) > 0:
                for token_id in set(generated):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty
            
            if temperature <= 0.0:
                # Greedy decoding (fastest)
                next_token = next_logits.argmax(dim=-1)
            else:
                # Temperature + top-p sampling
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                idx = torch.multinomial(sorted_probs[0], 1).item()
                next_token = sorted_indices[0, idx]
            
            generated.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = next_token.view(1, 1) if next_token.dim() == 0 else next_token.view(1, -1)

        return tokenizer.decode(generated, skip_special_tokens=True)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "native_kernel": NATIVE_KERNEL})


@app.route('/generate', methods=['POST'])
def generate_endpoint():
    data = request.json
    prompt = data.get('text', data.get('prompt', ''))
    max_tokens = data.get('max_new_tokens', data.get('max_tokens', 128))

    start_time = time.time()
    result = generate(prompt, max_tokens)
    latency = time.time() - start_time

    return jsonify({
        "text": result,
        "meta_info": {
            "latency": latency,
            "completion_tokens": len(tokenizer.encode(result)),
            "tok_per_sec": len(tokenizer.encode(result)) / latency,
        }
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 128)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    start_time = time.time()
    result = generate(prompt, max_tokens)
    latency = time.time() - start_time

    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "model": "bitnet-native",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result},
            "finish_reason": "stop",
        }],
        "usage": {
            "completion_tokens": len(tokenizer.encode(result)),
            "latency": latency,
            "tok_per_sec": len(tokenizer.encode(result)) / latency,
        }
    })


def main():
    parser = argparse.ArgumentParser(
        description='Native BitNet Server with sgl-kernel TL2 kernels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # From sgl-kernel binary format
    python scripts/serve_bitnet_native.py --model models/dlm-bitnet-2b.bin --tokenizer models/dlm-bitnet-2b

    # From packed safetensors directory
    python scripts/serve_bitnet_native.py --model models/bitnet-2b
        """
    )
    parser.add_argument('--model', '--model-path', type=str, default='models/bitnet-2b',
                        help='Path to model (.bin file or directory)')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to tokenizer (required for .bin files)')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    load_model(args.model, args.tokenizer)
    logger.info(f"Starting native BitNet server on {args.host}:{args.port}")
    logger.info(f"Native kernels enabled: {NATIVE_KERNEL}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
