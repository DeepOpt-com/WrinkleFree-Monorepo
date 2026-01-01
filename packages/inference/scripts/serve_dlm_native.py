#!/usr/bin/env python3
"""Native DLM server using sgl-kernel SIMD kernels with Fast-dLLM v2 block diffusion.

This combines:
- sgl-kernel's AVX512 SIMD kernels for fast inference (~27 tok/s)
- Fast-dLLM v2 block diffusion for parallel token generation

Usage:
    # Step 1: Convert DLM checkpoint to packed format
    python scripts/convert_dlm_to_sglkernel.py models/dlm-bitnet-2b models/dlm-bitnet-2b.bin

    # Step 2: Start server
    python scripts/serve_dlm_native.py \
        --model models/dlm-bitnet-2b.bin \
        --tokenizer models/dlm-bitnet-2b

    # Step 3: Test
    curl http://localhost:30000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'

Performance: ~27 tok/s on GCP c3d-standard-32 (AMD EPYC Genoa with AVX512)
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
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request, Response
from safetensors import safe_open
from transformers import AutoTokenizer, PretrainedConfig

logging.basicConfig(level=logging.INFO)
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


@dataclass
class DlmConfig:
    """DLM block diffusion configuration."""
    mask_token_id: int = 128256  # |<MASK>| token
    block_size: int = 32
    small_block_size: int = 8
    threshold: float = 0.95

    def num_small_blocks(self) -> int:
        return self.block_size // self.small_block_size


def load_sglkernel_binary(path: Path) -> Tuple[BitNetConfig, dict]:
    """Load model from sgl-kernel binary format (.bin file)."""
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
            scale_bytes = f.read(4)
            scale = struct.unpack('<f', scale_bytes)[0] if has_scale else None

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
    """Linear layer using packed 1.58-bit weights with native kernels."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

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
        orig_shape = x.shape[:-1]
        x_flat = x.view(-1, x.shape[-1])

        if NATIVE_KERNEL:
            scale = self.weight_scale.item()

            if x_flat.shape[0] == 1:
                x_int8, act_scale = quantize_activations_i8(x_flat.squeeze(0).float())
                out = bitnet_gemv(self.weight, x_int8, scale)
                out = out.unsqueeze(0)
            else:
                x_int8, act_scale = quantize_activations_i8(x_flat.float())
                out = bitnet_gemm(self.weight, x_int8, scale)

            out = (out * act_scale).to(x.dtype)
        else:
            out = self._fallback_forward(x_flat)

        return out.view(*orig_shape, -1)

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._unpack_weights_simd()
        return F.linear(x, weight * self.weight_scale.item())

    def _unpack_weights_simd(self) -> torch.Tensor:
        packed = self.weight
        M, K_packed = packed.shape
        K = K_packed * 4

        out = torch.zeros(M, K, dtype=torch.float32)
        num_blocks = K // QK_I2_S
        for block_idx in range(num_blocks):
            base_w = block_idx * QK_I2_S
            base_p = block_idx * 32

            for j in range(32):
                byte_val = packed[:, base_p + j]
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
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.attn_sub_norm(attn_output)
        output = self.o_proj(attn_output)
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
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
        self.lm_head_weight = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        hidden_states, past_key_values = self.model(
            input_ids, attention_mask, past_key_values
        )
        logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        return logits, past_key_values

    @classmethod
    def from_pretrained(cls, model_path: str) -> "BitNetForCausalLM":
        """Load model from sgl-kernel binary format."""
        model_path = Path(model_path)

        if model_path.suffix == '.bin' or model_path.name.endswith('.bin'):
            return cls._from_sglkernel_binary(model_path)
        else:
            raise ValueError("DLM server requires .bin format (sgl-kernel packed)")

    @classmethod
    def _from_sglkernel_binary(cls, bin_path: Path) -> "BitNetForCausalLM":
        config, tensors = load_sglkernel_binary(bin_path)
        model = cls(config)

        def get_module_by_path(root, path: str):
            parts = path.split('.')
            current = root
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)
            return current

        loaded = 0
        scales_set = 0
        for name, param in model.named_parameters():
            key = name
            if key in tensors:
                t = tensors[key]
                param.data.copy_(t['tensor'])
                loaded += 1

                if t['scale'] is not None and name.endswith('.weight'):
                    parent_path = name.rsplit('.', 1)[0]
                    try:
                        parent = get_module_by_path(model, parent_path)
                        if hasattr(parent, 'weight_scale'):
                            parent.weight_scale.data.fill_(t['scale'])
                            scales_set += 1
                    except (AttributeError, IndexError, KeyError):
                        pass

        logger.info(f"Loaded {loaded} parameters, set {scales_set} scales, native kernels: {NATIVE_KERNEL}")
        return model


# Flask app
app = Flask(__name__)
model = None
tokenizer = None
model_lock = Lock()
dlm_config = None


def load_model(model_path: str, tokenizer_path: str = None):
    """Load BitNet model and tokenizer."""
    global model, tokenizer, dlm_config

    if tokenizer_path is None:
        if model_path.endswith('.bin'):
            bin_path = Path(model_path)
            candidate = bin_path.parent / bin_path.stem
            if candidate.exists() and (candidate / 'tokenizer.json').exists():
                tokenizer_path = str(candidate)
            else:
                raise ValueError(f"Cannot find tokenizer for {model_path}")
        else:
            tokenizer_path = model_path

    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading BitNet model from {model_path}")
    model = BitNetForCausalLM.from_pretrained(model_path)
    model.eval()

    # Setup DLM config
    dlm_config = DlmConfig()
    logger.info(f"DLM config: mask_token_id={dlm_config.mask_token_id}, "
                f"block_size={dlm_config.block_size}, threshold={dlm_config.threshold}")


@torch.no_grad()
def generate_block_diffusion(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """Generate text using Fast-dLLM v2 block diffusion.

    This generates tokens in parallel blocks instead of one at a time,
    achieving ~2.5x speedup over autoregressive generation.
    """
    with model_lock:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        block_size = dlm_config.block_size
        small_block_size = dlm_config.small_block_size
        threshold = dlm_config.threshold
        mask_id = dlm_config.mask_token_id

        generated = []
        past_key_values = None

        # Prefill: process prompt
        logits, past_key_values = model(input_ids, past_key_values=past_key_values)

        # Generate first token (autoregressive)
        next_logits = logits[:, -1, :]
        first_token = next_logits.argmax(dim=-1).item()
        generated.append(first_token)

        if first_token == tokenizer.eos_token_id:
            return tokenizer.decode(generated, skip_special_tokens=True)

        # Update input for next step
        current_ids = torch.tensor([[first_token]])

        tokens_generated = 1
        while tokens_generated < max_new_tokens:
            # Initialize block with mask tokens
            block_tokens = [mask_id] * block_size
            block_start_pos = len(input_ids[0]) + len(generated) - 1

            # Process each small block
            num_small_blocks = block_size // small_block_size
            for small_block_idx in range(num_small_blocks):
                start_idx = small_block_idx * small_block_size
                end_idx = start_idx + small_block_size

                # Iterate until all masks in this small block are resolved
                while True:
                    # Count remaining masks in this small block
                    masks_remaining = sum(1 for i in range(start_idx, end_idx)
                                         if block_tokens[i] == mask_id)

                    if masks_remaining == 0:
                        break

                    # Build input: context + current block tokens
                    block_input = torch.tensor([block_tokens], dtype=torch.long)

                    # Full forward (no KV cache for block diffusion within blocks)
                    # Concatenate prompt + generated + block
                    full_input = torch.cat([
                        input_ids,
                        torch.tensor([generated], dtype=torch.long),
                        block_input
                    ], dim=1)

                    # Forward pass
                    logits_out, _ = model(full_input)

                    # Get logits for masked positions in this small block
                    candidates = []
                    prompt_len = input_ids.shape[1]
                    gen_len = len(generated)

                    for i in range(start_idx, end_idx):
                        if block_tokens[i] == mask_id:
                            # Position in full sequence
                            pos = prompt_len + gen_len + i
                            token_logits = logits_out[0, pos, :]

                            # Sample token
                            if temperature <= 0.0:
                                token_id = token_logits.argmax().item()
                            else:
                                probs = F.softmax(token_logits / temperature, dim=-1)
                                token_id = torch.multinomial(probs, 1).item()

                            # Compute confidence
                            probs = F.softmax(token_logits, dim=-1)
                            confidence = probs[token_id].item()
                            candidates.append((i, token_id, confidence))

                    # Unmask high-confidence tokens
                    unmasked_any = False
                    for idx, token_id, conf in candidates:
                        if conf > threshold:
                            block_tokens[idx] = token_id
                            unmasked_any = True

                    # Always unmask at least one (highest confidence)
                    if not unmasked_any and candidates:
                        best = max(candidates, key=lambda x: x[2])
                        block_tokens[best[0]] = best[1]

            # Add block tokens to generated
            for token in block_tokens:
                if token == tokenizer.eos_token_id:
                    return tokenizer.decode(generated, skip_special_tokens=True)
                generated.append(token)
                tokens_generated += 1
                if tokens_generated >= max_new_tokens:
                    break

            if tokens_generated >= max_new_tokens:
                break

        return tokenizer.decode(generated, skip_special_tokens=True)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "native_kernel": NATIVE_KERNEL,
        "mode": "dlm_block_diffusion"
    })


@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": "dlm-bitnet-native",
            "object": "model",
            "created": 1700000000,
            "owned_by": "wrinklefree",
        }]
    })


@app.route('/generate', methods=['POST'])
def generate_endpoint():
    data = request.json
    prompt = data.get('text', data.get('prompt', ''))
    max_tokens = data.get('max_new_tokens', data.get('max_tokens', 128))

    start_time = time.time()
    result = generate_block_diffusion(prompt, max_tokens)
    latency = time.time() - start_time

    result_tokens = len(tokenizer.encode(result))
    return jsonify({
        "text": result,
        "meta_info": {
            "latency": latency,
            "completion_tokens": result_tokens,
            "tok_per_sec": result_tokens / latency if latency > 0 else 0,
        }
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 128)
    temperature = data.get('temperature', 0.0)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    start_time = time.time()
    result = generate_block_diffusion(prompt, max_tokens, temperature)
    latency = time.time() - start_time

    result_tokens = len(tokenizer.encode(result))
    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "model": "dlm-bitnet-native",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result},
            "finish_reason": "stop",
        }],
        "usage": {
            "completion_tokens": result_tokens,
            "latency": latency,
            "tok_per_sec": result_tokens / latency if latency > 0 else 0,
        }
    })


def main():
    parser = argparse.ArgumentParser(
        description='Native DLM Server with sgl-kernel SIMD + block diffusion',
    )
    parser.add_argument('--model', '--model-path', type=str, required=True,
                        help='Path to model (.bin file)')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to tokenizer')
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--mask-token-id', type=int, default=128256)
    args = parser.parse_args()

    global dlm_config
    dlm_config = DlmConfig(
        mask_token_id=args.mask_token_id,
        block_size=args.block_size,
        threshold=args.threshold,
    )

    load_model(args.model, args.tokenizer)
    logger.info(f"Starting DLM server on {args.host}:{args.port}")
    logger.info(f"Native kernels: {NATIVE_KERNEL}")
    logger.info(f"Block diffusion: block_size={args.block_size}, threshold={args.threshold}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
