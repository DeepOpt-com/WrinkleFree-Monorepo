#!/usr/bin/env python3
"""Test HuggingFace BitNet model to see what it produces."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_name = "microsoft/BitNet-b1.58-2B-4T"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Use CPU since we're just testing
    )

    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"Token IDs: {inputs.input_ids.tolist()}")

    # Get logits for next token
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last position

    # Find top tokens
    top_k = 10
    top_indices = torch.topk(logits, top_k).indices.tolist()
    top_values = torch.topk(logits, top_k).values.tolist()

    print(f"\nTop {top_k} predictions:")
    for i, (idx, val) in enumerate(zip(top_indices, top_values)):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. Token {idx} ({repr(token)}): {val:.4f}")

    # Check Paris specifically
    paris_id = tokenizer.encode(" Paris", add_special_tokens=False)[0]
    paris_logit = logits[paris_id].item()
    paris_rank = (logits > paris_logit).sum().item() + 1
    print(f"\n' Paris' (token {paris_id}) logit: {paris_logit:.4f}, rank: {paris_rank}")

    # Generate a few tokens
    print("\n=== Generation ===")
    generated = model.generate(
        inputs.input_ids,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Output: {tokenizer.decode(generated[0])}")


if __name__ == "__main__":
    main()
