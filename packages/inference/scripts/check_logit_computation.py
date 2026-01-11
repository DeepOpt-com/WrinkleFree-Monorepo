#!/usr/bin/env python3
"""Check logit computation with my manual BitLinear code."""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
import struct
import mmap


def load_gguf_tensors(gguf_path):
    """Load tensors from GGUF file."""
    def read_string(mm, offset):
        length = struct.unpack_from('<Q', mm, offset)[0]
        s = mm[offset + 8:offset + 8 + length].decode('utf-8')
        return s, 8 + length

    def skip_value(mm, offset, value_type):
        type_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        if value_type == 8:
            _, consumed = read_string(mm, offset)
            return consumed
        elif value_type == 9:
            arr_type = struct.unpack_from('<I', mm, offset)[0]
            arr_len = struct.unpack_from('<Q', mm, offset + 4)[0]
            consumed = 12
            if arr_type == 8:
                for _ in range(arr_len):
                    _, c = read_string(mm, offset + consumed)
                    consumed += c
            else:
                consumed += arr_len * type_sizes.get(arr_type, 0)
            return consumed
        return type_sizes.get(value_type, 0)

    with open(gguf_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        n_tensors = struct.unpack_from('<Q', mm, 8)[0]
        n_kv = struct.unpack_from('<Q', mm, 16)[0]
        offset = 24
        alignment = 32

        for _ in range(n_kv):
            _, consumed = read_string(mm, offset)
            offset += consumed
            vtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            offset += skip_value(mm, offset, vtype)

        tensors = {}
        for _ in range(n_tensors):
            name, consumed = read_string(mm, offset)
            offset += consumed
            n_dims = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack_from('<Q', mm, offset)[0])
                offset += 8
            dtype = struct.unpack_from('<I', mm, offset)[0]
            offset += 4
            rel_offset = struct.unpack_from('<Q', mm, offset)[0]
            offset += 8
            tensors[name] = {'dims': dims, 'dtype': dtype, 'offset': rel_offset}

        padding = offset % alignment
        if padding != 0:
            offset += alignment - padding
        data_offset = offset

        result = {}
        for name, info in tensors.items():
            abs_offset = data_offset + info['offset']
            if info['dtype'] == 0:  # F32
                n_elements = 1
                for d in info['dims']:
                    n_elements *= d
                n_bytes = n_elements * 4
                data = np.frombuffer(mm[abs_offset:abs_offset + n_bytes], dtype=np.float32).copy()
                result[name] = {'data': data, 'dims': info['dims'], 'dtype': 'f32'}
            elif info['dtype'] == 1:  # F16
                n_elements = 1
                for d in info['dims']:
                    n_elements *= d
                n_bytes = n_elements * 2
                data = np.frombuffer(mm[abs_offset:abs_offset + n_bytes], dtype=np.float16).copy().astype(np.float32)
                result[name] = {'data': data, 'dims': info['dims'], 'dtype': 'f16'}

        return result, mm


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization."""
    variance = (x ** 2).mean(-1, keepdims=True)
    rms = np.sqrt(variance + eps)
    return (x / rms) * weight


def main():
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"

    print("Loading GGUF tensors...")
    tensors, mm = load_gguf_tensors(gguf_path)

    # Load output norm
    output_norm = tensors['output_norm.weight']['data']
    print(f"Output norm shape: {output_norm.shape}")
    print(f"Output norm first 8: {output_norm[:8]}")
    print(f"Output norm min/max: {output_norm.min():.4f}, {output_norm.max():.4f}")

    # Load embeddings
    emb_tensor = tensors['token_embd.weight']
    hidden_size = emb_tensor['dims'][0]
    vocab_size = emb_tensor['dims'][1]
    embeddings = emb_tensor['data'].reshape(vocab_size, hidden_size)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {emb_tensor['dtype']}")

    # Check Paris embedding
    paris_id = 12366
    paris_emb = embeddings[paris_id]
    print(f"\nParis (token {paris_id}) embedding:")
    print(f"  First 8: {paris_emb[:8]}")
    print(f"  Min: {paris_emb.min():.4f}, Max: {paris_emb.max():.4f}")
    print(f"  L2 norm: {np.linalg.norm(paris_emb):.4f}")

    # Create a synthetic "final hidden state" similar to what Rust produces
    # Use Rust's reported values: Min: -0.9056, Max: 0.4728
    np.random.seed(42)
    fake_hidden = np.random.randn(hidden_size).astype(np.float32) * 0.3
    print(f"\nSynthetic hidden (before norm):")
    print(f"  Min: {fake_hidden.min():.4f}, Max: {fake_hidden.max():.4f}")

    # Apply output norm
    hidden_normed = rms_norm(fake_hidden.reshape(1, -1), output_norm)[0]
    print(f"\nSynthetic hidden (after norm):")
    print(f"  Min: {hidden_normed.min():.4f}, Max: {hidden_normed.max():.4f}")

    # Compute logit for Paris
    paris_logit = np.dot(hidden_normed, paris_emb)
    print(f"\nParis logit (synthetic): {paris_logit:.4f}")

    # Compute all logits to get rank
    all_logits = embeddings @ hidden_normed
    paris_rank = (all_logits > paris_logit).sum() + 1
    print(f"Paris rank (synthetic): {paris_rank}")
    print(f"Logit range: {all_logits.min():.4f} to {all_logits.max():.4f}")

    # Check what the top tokens are
    top_indices = np.argsort(all_logits)[::-1][:10]
    print(f"\nTop 10 tokens (synthetic):")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Token {idx}: {all_logits[idx]:.4f}")

    # Now check with actual HuggingFace final hidden state
    print("\n" + "="*50)
    print("Loading HuggingFace model to get real hidden state...")

    model_name = "microsoft/BitNet-b1.58-2B-4T"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        output_hidden_states=True
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        final_hidden = outputs.hidden_states[-1][0, -1, :].float().numpy()

    print(f"\nHuggingFace final hidden (before norm):")
    print(f"  First 8: {final_hidden[:8]}")
    print(f"  Min: {final_hidden.min():.4f}, Max: {final_hidden.max():.4f}")
    print(f"  L2 norm: {np.linalg.norm(final_hidden):.4f}")

    # Apply output norm (using GGUF weights)
    hf_hidden_normed = rms_norm(final_hidden.reshape(1, -1), output_norm)[0]
    print(f"\nHuggingFace hidden (after my RMS norm):")
    print(f"  First 8: {hf_hidden_normed[:8]}")
    print(f"  Min: {hf_hidden_normed.min():.4f}, Max: {hf_hidden_normed.max():.4f}")

    # Compute logit for Paris
    paris_logit_hf = np.dot(hf_hidden_normed, paris_emb)
    print(f"\nParis logit (HF hidden + my norm + GGUF embeddings): {paris_logit_hf:.4f}")

    # Compare with HuggingFace's computed logit
    hf_logits = outputs.logits[0, -1, :].float().numpy()
    print(f"Paris logit (HuggingFace computed): {hf_logits[paris_id]:.4f}")

    # Compute all logits using GGUF embeddings
    my_logits = embeddings @ hf_hidden_normed
    my_paris_rank = (my_logits > my_logits[paris_id]).sum() + 1
    print(f"\nMy logit computation Paris rank: {my_paris_rank}")
    print(f"My logit range: {my_logits.min():.4f} to {my_logits.max():.4f}")

    # Check if GGUF embeddings match HuggingFace embeddings
    hf_embeddings = model.model.embed_tokens.weight.float().numpy()
    print(f"\nHuggingFace embedding shape: {hf_embeddings.shape}")
    print(f"GGUF embedding shape: {embeddings.shape}")

    # Compare Paris embeddings
    hf_paris = hf_embeddings[paris_id]
    gguf_paris = embeddings[paris_id]
    diff = np.abs(hf_paris - gguf_paris)
    print(f"\nParis embedding comparison:")
    print(f"  HF first 8: {hf_paris[:8]}")
    print(f"  GGUF first 8: {gguf_paris[:8]}")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")


def check_layer_outputs():
    """Check HuggingFace hidden state at each layer."""
    print("Loading HuggingFace model...")
    model_name = "microsoft/BitNet-b1.58-2B-4T"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        output_hidden_states=True
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    print(f"\nNumber of hidden states: {len(hidden_states)} (embedding + {len(hidden_states)-1} layers)")

    print("\nHidden state magnitudes at each layer (position 0):")
    for i, h in enumerate(hidden_states):
        h_np = h[0, 0, :].float().numpy()  # Position 0
        print(f"  Layer {i-1 if i > 0 else 'emb':>3}: min={h_np.min():.4f}, max={h_np.max():.4f}, L2={np.linalg.norm(h_np):.4f}")


def investigate_layer29():
    """Investigate what happens at layer 29."""
    print("Loading HuggingFace model...")
    model_name = "microsoft/BitNet-b1.58-2B-4T"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for precision
        device_map="cpu",
        output_hidden_states=True
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Check if there's a final norm after the layers
    print("\nModel structure:")
    print(f"  model.model.norm: {model.model.norm}")
    print(f"  model.lm_head: {model.lm_head}")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    # Compare layer 28 and layer 29
    layer28 = hidden_states[29][0, 0, :].numpy()  # After layer 28
    layer29 = hidden_states[30][0, 0, :].numpy()  # After layer 29

    print(f"\nLayer 28 output: min={layer28.min():.4f}, max={layer28.max():.4f}")
    print(f"Layer 29 output: min={layer29.min():.4f}, max={layer29.max():.4f}")

    # Check if layer 29 applies a different norm
    print("\n--- Examining layer 29 ---")
    layer29_module = model.model.layers[29]
    print(f"Layer 29 type: {type(layer29_module)}")

    # Apply model.norm manually to layer 28 output to see if it matches layer 29
    with torch.no_grad():
        layer28_tensor = hidden_states[29][0:1, 0:1, :]  # Keep batch/seq dims
        normed = model.model.norm(layer28_tensor)
        normed_np = normed[0, 0, :].numpy()
    print(f"\nLayer 28 + model.norm: min={normed_np.min():.4f}, max={normed_np.max():.4f}")

    # Check if there's a discrepancy
    diff = np.abs(layer29 - normed_np)
    print(f"Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")


def quick_model_check():
    """Quick check of model structure."""
    print("Loading model config only...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("microsoft/BitNet-b1.58-2B-4T")
    print(f"\nModel config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  rms_norm_eps: {getattr(config, 'rms_norm_eps', 'N/A')}")


def verify_norm_theory():
    """Verify the final norm theory."""
    print("Loading HuggingFace model...")
    model_name = "microsoft/BitNet-b1.58-2B-4T"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        output_hidden_states=True
    )

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    # hidden_states[29] = after layer 28, hidden_states[30] = after layer 29
    h29 = hidden_states[29][0, -1, :].float()  # Last position
    h30 = hidden_states[30][0, -1, :].float()

    print(f"\nAfter layer 28: min={h29.min():.2f}, max={h29.max():.2f}, L2={h29.norm():.2f}")
    print(f"After layer 29: min={h30.min():.4f}, max={h30.max():.4f}, L2={h30.norm():.4f}")

    # Apply model.norm to h29 (after layer 28) and check if it matches h30
    with torch.no_grad():
        h29_normed = model.model.norm(h29.unsqueeze(0).unsqueeze(0)).squeeze()

    print(f"\nAfter layer 28 + model.norm: min={h29_normed.min():.4f}, max={h29_normed.max():.4f}")

    # Check: is h30 the result of layer29(h29) or model.norm(layer29_output)?
    # If hidden_states[30] is already normalized, it should be very different from layer29(h29)

    # Also check what output.logits looks like
    logits = outputs.logits[0, -1, :]
    print(f"\nLogits: min={logits.min():.2f}, max={logits.max():.2f}")


def compare_bitlinear():
    """Compare my BitLinear with HuggingFace's BitLinear."""
    print("Loading models...")

    model_name = "microsoft/BitNet-b1.58-2B-4T"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Get the first layer's gate projection weights
    gate_proj = model.model.layers[0].mlp.gate_proj
    print(f"\nHuggingFace gate_proj type: {type(gate_proj)}")
    print(f"  weight shape: {gate_proj.weight.shape}")
    print(f"  weight dtype: {gate_proj.weight.dtype}")

    # Check if it's quantized
    if hasattr(gate_proj, 'weight_scale'):
        print(f"  weight_scale: {gate_proj.weight_scale}")

    # Get the weights
    weights = gate_proj.weight.detach()
    print(f"  weight min: {weights.min():.4f}, max: {weights.max():.4f}")
    print(f"  weight unique values: {torch.unique(weights).tolist()[:10]}")

    # Count ternary distribution
    n_minus1 = (weights == -1).sum().item()
    n_zero = (weights == 0).sum().item()
    n_plus1 = (weights == 1).sum().item()
    total = weights.numel()
    print(f"  Ternary distribution: -1: {n_minus1/total*100:.1f}%, 0: {n_zero/total*100:.1f}%, +1: {n_plus1/total*100:.1f}%")

    # Now load GGUF and compare
    print("\nLoading GGUF weights...")
    gguf_path = "/tmp/bitnet-gguf/ggml-model-i2_s.gguf"
    tensors, mm = load_gguf_tensors(gguf_path)

    # GGUF gate_proj for layer 0
    gate_gguf = tensors['blk.0.ffn_gate.weight']
    print(f"\nGGUF gate_proj:")
    print(f"  shape: {gate_gguf['dims']}")
    print(f"  scale: {gate_gguf['scale']}")
    print(f"  dtype: {gate_gguf['dtype']}")

    gguf_weights = gate_gguf['data']
    print(f"  weight min: {gguf_weights.min()}, max: {gguf_weights.max()}")
    print(f"  weight unique: {np.unique(gguf_weights).tolist()}")

    n_minus1 = (gguf_weights == -1).sum()
    n_zero = (gguf_weights == 0).sum()
    n_plus1 = (gguf_weights == 1).sum()
    total = len(gguf_weights)
    print(f"  Ternary distribution: -1: {n_minus1/total*100:.1f}%, 0: {n_zero/total*100:.1f}%, +1: {n_plus1/total*100:.1f}%")

    # Compare first 100 weights
    hf_flat = weights.flatten()[:100].numpy()
    gguf_flat = gguf_weights[:100].astype(np.float32)
    print(f"\nFirst 10 weights comparison:")
    print(f"  HF:   {hf_flat[:10].tolist()}")
    print(f"  GGUF: {gguf_flat[:10].tolist()}")

    # Check if they match
    matches = (hf_flat == gguf_flat).sum()
    print(f"  Match rate (first 100): {matches}%")

    # The key difference might be in how scaling is applied
    # HuggingFace BitLinear: output = (weights @ quant(x)) * weight_scale * activation_scale
    # My implementation: same
    # But maybe the weight_scale values are different?

    # Check if HuggingFace has a different scale
    if hasattr(gate_proj, 'weight_scale'):
        print(f"\nHuggingFace weight_scale: {gate_proj.weight_scale}")
    else:
        print("\nHuggingFace gate_proj doesn't have weight_scale attribute")

    # Check the actual forward computation
    print("\n--- Testing forward pass ---")

    # Create a test input (same as our normalized FFN input)
    test_input = torch.randn(1, 1, 2560) * 10  # Scale similar to FFN input

    # HuggingFace forward
    with torch.no_grad():
        hf_output = gate_proj(test_input.squeeze(0)).squeeze(0)

    print(f"Test input: min={test_input.min():.2f}, max={test_input.max():.2f}")
    print(f"HuggingFace output: min={hf_output.min():.2f}, max={hf_output.max():.2f}")

    # My BitLinear forward (using GGUF weights)
    input_np = test_input.squeeze().numpy()
    max_abs = np.abs(input_np).max()
    input_scale = max_abs / 127.0
    input_quant = np.round(input_np / input_scale).clip(-127, 127).astype(np.int8)

    in_features = gate_gguf['dims'][0]
    out_features = gate_gguf['dims'][1]
    W = gguf_weights.reshape(out_features, in_features)
    output_int = W.astype(np.int32) @ input_quant.astype(np.int32)
    my_output = output_int.astype(np.float32) * input_scale * gate_gguf['scale']

    print(f"My output: min={my_output.min():.2f}, max={my_output.max():.2f}")
    print(f"\nRatio (mine/HF): {my_output.max() / hf_output.max().item():.2f}x")


if __name__ == "__main__":
    compare_bitlinear()
