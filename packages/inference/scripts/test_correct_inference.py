#!/usr/bin/env python3
"""Test the correct inference pipeline with pre-quantized weights.

KEY INSIGHT:
- Pre-quantized weights should be used in STANDARD Linear layers
- NOT in BitLinear, which would re-quantize with wrong scale
- Activation quantization is done only during training, not inference

The correct inference pipeline for GGUF/C++:
1. Load pre-quantized weights into standard model
2. Run inference WITHOUT re-quantizing
3. Accept slight difference from BitLinear (no activation quantization)
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "architecture/src"))
from bitnet_arch import set_global_lambda_warmup, LambdaWarmup, convert_model_to_bitnet


def main():
    # Set lambda = 1.0 for BitLinear reference
    warmup = LambdaWarmup(warmup_steps=0, min_lambda=1.0, max_lambda=1.0)
    set_global_lambda_warmup(warmup)

    tokenizer = AutoTokenizer.from_pretrained("models/smollm2-135m-dlm-subln")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("=" * 60)
    print("CORRECT INFERENCE PIPELINE TEST")
    print("=" * 60)

    # Reference: BitLinear with on-the-fly quantization (training behavior)
    print("\n=== Reference: BitLinear (on-the-fly quant + activation quant) ===")
    model_bl = AutoModelForCausalLM.from_pretrained(
        "models/smollm2-135m-dlm-subln",
        torch_dtype=torch.float32,
    )
    model_bl = convert_model_to_bitnet(model_bl, 576, 1536, insert_subln=False)
    model_bl.eval()

    with torch.no_grad():
        out_bl = model_bl.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    print(f"Output: {tokenizer.decode(out_bl[0], skip_special_tokens=True)}")

    # Correct inference: Pre-quantized weights in standard model
    print("\n=== Correct Inference: Standard Model + Pre-quantized Weights ===")
    model_std = AutoModelForCausalLM.from_pretrained(
        "models/smollm2-135m-dlm-subln",
        torch_dtype=torch.float32,
    )

    # Load pre-quantized weights
    with safe_open("models/smollm2-135m-quantized_quantized/model.safetensors", framework="pt") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    model_std.load_state_dict(state_dict, strict=False)
    model_std.eval()

    # Verify weights are quantized
    w = model_std.model.layers[0].self_attn.q_proj.weight
    print(f"Weight check: {len(torch.unique(w))} unique values")

    with torch.no_grad():
        out_std = model_std.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    print(f"Output: {tokenizer.decode(out_std[0], skip_special_tokens=True)}")

    # Wrong approach: BitLinear with pre-quantized weights (re-quantizes!)
    print("\n=== WRONG: BitLinear + Pre-quantized (re-quantizes with wrong scale!) ===")
    model_wrong = AutoModelForCausalLM.from_pretrained(
        "models/smollm2-135m-dlm-subln",
        torch_dtype=torch.float32,
    )
    model_wrong = convert_model_to_bitnet(model_wrong, 576, 1536, insert_subln=False)
    model_wrong.load_state_dict(state_dict, strict=False)
    model_wrong.eval()

    with torch.no_grad():
        out_wrong = model_wrong.generate(
            **inputs, max_new_tokens=30, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    print(f"Output: {tokenizer.decode(out_wrong[0], skip_special_tokens=True)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nFor C++/GGUF inference:")
    print("1. Use pre-quantized weights (ternary * scale)")
    print("2. Do NOT re-apply weight quantization")
    print("3. Activation quantization is optional (slight diff OK)")
    print("\nExpected: Pre-quantized output should be similar to BitLinear,")
    print("but may differ slightly due to no activation quantization.")


if __name__ == "__main__":
    main()
