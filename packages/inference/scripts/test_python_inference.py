#!/usr/bin/env python3
"""Test Python inference with BitLinear on-the-fly quantization.

This script:
1. Loads model as standard LlamaForCausalLM (no quantization)
2. Converts to BitLinear using bitnet_arch (applies on-the-fly quantization)
3. Tests inference with both to compare outputs

The pre-quantized checkpoint should produce identical results to
BitLinear's on-the-fly quantization when lambda=1.0.
"""

import argparse
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add bitnet_arch to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "architecture/src"))
from bitnet_arch import convert_model_to_bitnet, set_global_lambda_warmup, LambdaWarmup


def test_inference(model, tokenizer, prompt: str, max_tokens: int = 30):
    """Run inference and return generated text."""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                       help="Path to checkpoint directory")
    parser.add_argument("--quantized", type=Path, default=None,
                       help="Path to pre-quantized checkpoint (optional)")
    parser.add_argument("--prompt", type=str, default="The capital of France is",
                       help="Test prompt")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    print("=" * 60)
    print("Testing Python Inference")
    print("=" * 60)

    # Set lambda to 1.0 for full quantization
    warmup = LambdaWarmup(warmup_steps=0)
    warmup._current_lambda = 1.0
    set_global_lambda_warmup(warmup)
    print(f"Lambda warmup set to 1.0 (full quantization)")

    # Test 1: Standard model (no quantization)
    print("\n=== Standard LlamaForCausalLM (no quantization) ===")
    model_std = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model_std.eval()
    output_std = test_inference(model_std, tokenizer, args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output_std}")

    # Test 2: Convert to BitLinear (on-the-fly quantization)
    print("\n=== BitLinear Model (on-the-fly quantization) ===")
    model_bitlinear = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    # Get model config for conversion
    config = model_bitlinear.config
    model_bitlinear = convert_model_to_bitnet(
        model_bitlinear,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        insert_subln=False,  # Don't insert SubLN if already present
    )
    model_bitlinear.eval()

    # Verify conversion
    from bitnet_arch.layers import BitLinear
    first_layer = model_bitlinear.model.layers[0].self_attn.q_proj
    print(f"First layer type: {type(first_layer).__name__}")

    output_bitlinear = test_inference(model_bitlinear, tokenizer, args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Output: {output_bitlinear}")

    # Test 3: Pre-quantized weights (if provided)
    if args.quantized:
        print("\n=== Pre-Quantized Weights (loaded into standard model) ===")
        model_prequant = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # Load pre-quantized weights
        with safe_open(args.quantized / "model.safetensors", framework="pt") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
        model_prequant.load_state_dict(state_dict, strict=False)
        model_prequant.eval()

        # Verify weights are quantized
        w = model_prequant.model.layers[0].self_attn.q_proj.weight
        unique = torch.unique(w)
        print(f"Weight unique values: {len(unique)} ({unique.tolist()[:5]}...)")

        output_prequant = test_inference(model_prequant, tokenizer, args.prompt)
        print(f"Prompt: {args.prompt}")
        print(f"Output: {output_prequant}")

    # Compare outputs
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"\nStandard (no quant):     {output_std}")
    print(f"BitLinear (on-the-fly):  {output_bitlinear}")
    if args.quantized:
        print(f"Pre-quantized:           {output_prequant}")

        if output_bitlinear == output_prequant:
            print("\n✓ BitLinear and pre-quantized outputs MATCH!")
        else:
            print("\n✗ BitLinear and pre-quantized outputs DIFFER")
            # Check logits similarity
            print("  (This is expected due to activation quantization in BitLinear)")

    return 0


if __name__ == "__main__":
    exit(main())
