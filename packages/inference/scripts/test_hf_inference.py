#!/usr/bin/env python3
"""Test HuggingFace inference to verify model produces correct output."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    model_path = "microsoft/bitnet-b1.58-2B-4T"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"Input tokens: {input_ids.tolist()}")
    print(f"Decoded tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Get the logit for the next token (after the prompt)
    next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

    # Find top tokens
    top_k = 10
    top_values, top_indices = torch.topk(next_token_logits, top_k)

    print(f"\nTop {top_k} next token predictions:")
    for i, (value, idx) in enumerate(zip(top_values, top_indices)):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. '{token}' (token_id={idx.item()}, logit={value.item():.4f})")

    # Check if Paris is in top
    paris_token_ids = tokenizer.encode(" Paris", add_special_tokens=False)
    print(f"\n' Paris' token IDs: {paris_token_ids}")

    for paris_id in paris_token_ids:
        logit = next_token_logits[paris_id].item()
        rank = (next_token_logits > logit).sum().item() + 1
        print(f"  Token {paris_id} ('{tokenizer.decode([paris_id])}') logit: {logit:.4f}, rank: {rank}")

    # Generate some text
    print("\n--- Generating continuation ---")
    generated = model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False,  # Greedy decoding
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
