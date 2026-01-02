#!/usr/bin/env python3
"""Debug script to isolate catastrophic training loss issue.

Mimics exactly what happens during Lightning training:
1. Load model
2. Convert to BitNet (no SubLN)
3. Set lambda warmup (lambda=0 at start)
4. Preprocess batch with DLM
5. Run forward pass
6. Calculate loss

Goal: Identify what breaks the model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bitnet_arch.conversion import auto_convert_if_needed
from bitnet_arch.quantization import LambdaWarmup, set_global_lambda_warmup, get_current_lambda


def test_step_by_step():
    """Test each step of the training pipeline to isolate the issue."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\n=== Step 1: Load model ===")
    model_name = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load WITHOUT moving to device first (like training does)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Move to device AFTER loading
    model = model.to(device)
    model.eval()

    # Create test batch
    text = "The capital of France is Paris. It is a beautiful city with many"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(device)

    # Test original model
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        print(f"Original model loss: {outputs.loss.item():.4f}")
        print(f"Original max logits: {outputs.logits.abs().max().item():.4f}")

    # Step 2: Convert to BitNet WITHOUT lambda warmup (should be lambda=1.0)
    # IMPORTANT: Convert BEFORE moving to device (like training does)
    print("\n=== Step 2: Convert to BitNet (no warmup) ===")
    model_converted = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    # Convert on CPU first (like training)
    model_converted = auto_convert_if_needed(
        model_converted,
        hidden_size=576,
        intermediate_size=1536,
        exclude_layers=["embed_tokens", "lm_head"],
        insert_subln=False,
    )
    # Then move to device
    model_converted = model_converted.to(device)
    model_converted.eval()

    print(f"Lambda (no warmup set): {get_current_lambda():.4f}")

    with torch.no_grad():
        outputs = model_converted(input_ids, labels=input_ids)
        print(f"BitNet model loss (lambda=1.0): {outputs.loss.item():.4f}")
        print(f"BitNet max logits: {outputs.logits.abs().max().item():.4f}")

    # Step 3: Set lambda warmup to simulate training start
    print("\n=== Step 3: Set lambda warmup (lambda=0) ===")
    warmup = LambdaWarmup(warmup_steps=200, schedule="linear")
    set_global_lambda_warmup(warmup)
    # Lambda starts at 0 before first step()
    print(f"Lambda after warmup init: {get_current_lambda():.4f}")

    # Reload and convert (to ensure clean state with lambda=0)
    model_warmup = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    model_warmup = auto_convert_if_needed(
        model_warmup,
        hidden_size=576,
        intermediate_size=1536,
        exclude_layers=["embed_tokens", "lm_head"],
        insert_subln=False,
    )
    model_warmup = model_warmup.to(device)
    model_warmup.eval()

    with torch.no_grad():
        outputs = model_warmup(input_ids, labels=input_ids)
        print(f"BitNet model loss (lambda=0): {outputs.loss.item():.4f}")
        print(f"BitNet max logits: {outputs.logits.abs().max().item():.4f}")

    # Step a few times and check loss
    print("\n=== Step 4: Step warmup and check loss ===")
    for step in [1, 10, 50, 100]:
        # Reset and step to specific point
        warmup = LambdaWarmup(warmup_steps=200, schedule="linear")
        set_global_lambda_warmup(warmup)
        for _ in range(step):
            warmup.step()

        # Need to reload model each time to test cleanly
        model_test = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        model_test = auto_convert_if_needed(
            model_test,
            hidden_size=576,
            intermediate_size=1536,
            exclude_layers=["embed_tokens", "lm_head"],
            insert_subln=False,
        )
        model_test = model_test.to(device)
        model_test.eval()

        with torch.no_grad():
            outputs = model_test(input_ids, labels=input_ids)
            print(f"Step {step}: lambda={get_current_lambda():.4f}, loss={outputs.loss.item():.4f}, max_logits={outputs.logits.abs().max().item():.4f}")

    # Step 5: Test with DLM preprocessing
    print("\n=== Step 5: Test with DLM preprocessing ===")
    set_global_lambda_warmup(None)  # Reset to lambda=1.0

    from wrinklefree.objectives.dlm import DLMObjective

    dlm = DLMObjective(
        mask_token_id=0,  # unk token
        mask_prob=0.5,
        use_complementary_masks=True,
    )

    # Prepare batch like training does
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }

    # Apply DLM preprocessing
    batch = dlm.preprocess_batch(batch)

    print(f"Original input_ids: {input_ids[0, :10].tolist()}")
    print(f"Masked input_ids (first half): {batch['input_ids'][0, :10].tolist()}")
    print(f"Masked input_ids (second half): {batch['input_ids'][batch['input_ids'].shape[0]//2, :10].tolist()}")
    print(f"Batch size after DLM: {batch['input_ids'].shape[0]} (doubled)")

    # Load fresh model - convert on CPU first
    model_dlm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    model_dlm = auto_convert_if_needed(
        model_dlm,
        hidden_size=576,
        intermediate_size=1536,
        exclude_layers=["embed_tokens", "lm_head"],
        insert_subln=False,
    )
    model_dlm = model_dlm.to(device)
    model_dlm.eval()

    with torch.no_grad():
        outputs = model_dlm(batch["input_ids"], labels=batch["_original_labels"])
        print(f"BitNet with DLM loss: {outputs.loss.item():.4f}")
        print(f"BitNet with DLM max logits: {outputs.logits.abs().max().item():.4f}")

    # Step 6: Test with lambda=0 AND DLM
    print("\n=== Step 6: Test with lambda=0 AND DLM ===")
    warmup = LambdaWarmup(warmup_steps=200, schedule="linear")
    set_global_lambda_warmup(warmup)
    # Lambda = 0 at start

    model_combined = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )

    model_combined = auto_convert_if_needed(
        model_combined,
        hidden_size=576,
        intermediate_size=1536,
        exclude_layers=["embed_tokens", "lm_head"],
        insert_subln=False,
    )
    model_combined = model_combined.to(device)
    model_combined.eval()

    print(f"Lambda: {get_current_lambda():.4f}")

    with torch.no_grad():
        outputs = model_combined(batch["input_ids"], labels=batch["_original_labels"])
        print(f"BitNet + DLM (lambda=0) loss: {outputs.loss.item():.4f}")
        print(f"BitNet + DLM (lambda=0) max logits: {outputs.logits.abs().max().item():.4f}")

    # Step 7: Test training mode vs eval mode
    print("\n=== Step 7: Test training mode vs eval mode ===")
    model_combined.train()

    with torch.no_grad():
        outputs = model_combined(batch["input_ids"], labels=batch["_original_labels"])
        print(f"TRAIN MODE - loss: {outputs.loss.item():.4f}, max_logits: {outputs.logits.abs().max().item():.4f}")

    model_combined.eval()

    with torch.no_grad():
        outputs = model_combined(batch["input_ids"], labels=batch["_original_labels"])
        print(f"EVAL MODE - loss: {outputs.loss.item():.4f}, max_logits: {outputs.logits.abs().max().item():.4f}")

    # Step 8: Test ObjectiveManager with curriculum (simulates training)
    print("\n=== Step 8: Test ObjectiveManager with curriculum (weight=0 should skip DLM) ===")
    from wrinklefree.objectives import create_objective_manager
    from wrinklefree.objectives.manager import CurriculumScheduler, CurriculumPhase
    from omegaconf import DictConfig, OmegaConf

    # Simulate warmup phase where DLM weight is 0
    # Note: create_objective_manager expects the config at the training level
    cfg = OmegaConf.create({
        "objectives": {
            "continue_pretrain": {"enabled": True, "weight": 1.0},
            "dlm": {
                "enabled": True,
                "weight": 0.5,
                "mask_prob": 0.5,
                "mask_token_id": 0,
                "use_complementary_masks": True,
            },
        },
        "curriculum": {
            "enabled": True,
            "interpolation": "linear",
            "phases": [
                {"name": "warmup", "end_ratio": 0.2, "objectives": {"continue_pretrain": 1.0, "dlm": 0.0}},
                {"name": "main", "end_ratio": 1.0, "objectives": {"continue_pretrain": 1.0, "dlm": 0.5}},
            ],
        },
    })

    manager = create_objective_manager(cfg, total_steps=100)

    # Reset lambda warmup
    set_global_lambda_warmup(None)

    # Create fresh batch
    fresh_batch = {
        "input_ids": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }

    # Test at step 0 (warmup phase, DLM weight = 0)
    print(f"Current weights at step 0: {manager.get_current_weights()}")
    processed_batch = manager.preprocess_batch(fresh_batch.copy())

    # Check if input was masked
    if "input_ids" in processed_batch:
        mask_count = (processed_batch["input_ids"] == 0).sum().item()
        print(f"Masked tokens at step 0: {mask_count} (should be 0 if DLM weight=0)")
        print(f"Batch size after preprocess: {processed_batch['input_ids'].shape[0]} (should be 1)")

    # Load fresh model
    model_manager = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model_manager = auto_convert_if_needed(
        model_manager,
        hidden_size=576,
        intermediate_size=1536,
        exclude_layers=["embed_tokens", "lm_head"],
        insert_subln=False,
    )
    model_manager = model_manager.to(device)
    model_manager.eval()

    with torch.no_grad():
        outputs = model_manager(processed_batch["input_ids"], labels=processed_batch.get("labels", processed_batch["input_ids"]))
        print(f"Loss at step 0 (warmup, DLM weight=0): {outputs.loss.item():.4f}")
        print(f"Max logits: {outputs.logits.abs().max().item():.4f}")

    # Now simulate step 50 (in main phase, DLM weight > 0)
    for _ in range(50):
        manager.step_curriculum()

    print(f"\nCurrent weights at step 50: {manager.get_current_weights()}")
    fresh_batch2 = {
        "input_ids": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }
    processed_batch2 = manager.preprocess_batch(fresh_batch2)

    if "input_ids" in processed_batch2:
        mask_count = (processed_batch2["input_ids"] == 0).sum().item()
        print(f"Masked tokens at step 50: {mask_count} (should be > 0 if DLM active)")
        print(f"Batch size after preprocess: {processed_batch2['input_ids'].shape[0]} (should be 2 if complementary)")

    print("\n=== Summary ===")
    print("If loss at step 0 is ~2.2 (no DLM preprocessing), the fix works!")
    print("If loss at step 0 is ~9-10 (DLM preprocessing applied), the bug persists.")


if __name__ == "__main__":
    test_step_by_step()
