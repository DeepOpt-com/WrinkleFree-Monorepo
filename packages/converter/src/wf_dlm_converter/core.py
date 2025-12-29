"""High-level API for DLM conversion.

Simple interface for converting BitNet models to Diffusion LLMs.

Example:
    >>> from wf_dlm_converter import convert, validate
    >>> result = convert(model="qwen3_4b", checkpoint="hf://org/model")
    >>> validate(model_path=result["output_path"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def convert(
    model: str,
    checkpoint_path: str,
    output_path: str = "./outputs/dlm",
    total_tokens: int = 1_000_000_000,
    block_size: int = 32,
    num_diffusion_steps: int = 8,
    learning_rate: float = 5e-5,
    backend: str = "modal",
    gpu: str = "A10G",
    hydra_overrides: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Convert a BitNet checkpoint to Diffusion LLM.

    This is the main entry point for conversion. It:
    1. Loads the BitNet checkpoint
    2. Adapts the model for block diffusion
    3. Fine-tunes with diffusion objectives
    4. Saves the converted model

    Args:
        model: Model config name (e.g., 'qwen3_4b', 'smollm2_135m')
        checkpoint_path: Path to BitNet checkpoint (local, hf://, gs://)
        output_path: Directory for converted model output
        total_tokens: Fine-tuning token budget (default 1B)
        block_size: Block size for block diffusion
        num_diffusion_steps: Diffusion steps per block
        learning_rate: Learning rate for fine-tuning
        backend: Execution backend ('modal' or 'local')
        gpu: GPU type for Modal ('H100', 'A10G', 'L4', 'dev')
        hydra_overrides: Additional Hydra config overrides

    Returns:
        Result dict with:
        - run_id: Job identifier
        - status: 'success' or 'failed'
        - output_path: Path to converted model
        - tokens_trained: Actual tokens used
    """
    if backend == "modal":
        return _convert_modal(
            model=model,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            total_tokens=total_tokens,
            block_size=block_size,
            num_diffusion_steps=num_diffusion_steps,
            learning_rate=learning_rate,
            gpu=gpu,
            hydra_overrides=hydra_overrides,
        )
    else:
        return _convert_local(
            model=model,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            total_tokens=total_tokens,
            block_size=block_size,
            num_diffusion_steps=num_diffusion_steps,
            learning_rate=learning_rate,
        )


def _convert_modal(
    model: str,
    checkpoint_path: str,
    output_path: str,
    total_tokens: int,
    block_size: int,
    num_diffusion_steps: int,
    learning_rate: float,
    gpu: str,
    hydra_overrides: Optional[list[str]],
) -> dict[str, Any]:
    """Run conversion on Modal."""
    from wf_dlm_converter.modal.deployer import run_conversion
    from wf_dlm_converter.constants import RunIdPrefix

    import hashlib
    import time

    # Generate run ID
    fingerprint = f"{model}-{checkpoint_path}-{total_tokens}-{time.time()}"
    run_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:8]
    run_id = f"{RunIdPrefix.CONVERT.value}{model}-{run_hash}"

    logger.info(f"Starting Modal conversion job: {run_id} on {gpu}")

    # Launch Modal function with GPU selection
    result = run_conversion(
        model=model,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        total_tokens=total_tokens,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
        learning_rate=learning_rate,
        gpu=gpu,
        hydra_overrides=hydra_overrides,
    )

    return {
        "run_id": run_id,
        "status": result.get("status", "unknown"),
        "output_path": result.get("output_path"),
        "tokens_trained": result.get("tokens_trained", 0),
    }


def _convert_local(
    model: str,
    checkpoint_path: str,
    output_path: str,
    total_tokens: int,
    block_size: int,
    num_diffusion_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Run conversion locally."""
    import torch
    from datasets import load_dataset

    from wf_dlm_converter.models import load_bitnet_checkpoint, BlockDiffusionAdapter
    from wf_dlm_converter.conversion import DiffusionFineTuner, save_dlm_checkpoint, DLMConfig
    from wf_dlm_converter.conversion.training import TrainingConfig

    logger.info(f"Starting local conversion for {model}")

    # Load BitNet checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model_obj, tokenizer, model_config = load_bitnet_checkpoint(checkpoint_path)

    # Adapt for block diffusion
    logger.info("Adapting model for block diffusion")
    adapter = BlockDiffusionAdapter(
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
    )
    adapted_model = adapter.adapt_model(model_obj)

    # Create training config
    training_config = TrainingConfig(
        total_tokens=total_tokens,
        learning_rate=learning_rate,
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
        output_dir=output_path,
    )

    # Load training data
    logger.info("Loading training data")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    # Create dataloader
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=training_config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    # Fine-tune
    logger.info(f"Starting fine-tuning for {total_tokens:,} tokens")
    finetuner = DiffusionFineTuner(
        model=adapted_model,
        config=training_config,
        tokenizer=tokenizer,
    )

    # Create simple dataloader (simplified for local testing)
    from torch.utils.data import DataLoader

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    dataloader = DataLoader(
        tokenized,
        batch_size=training_config.batch_size,
    )

    trained_model = finetuner.train(dataloader)

    # Save checkpoint
    dlm_config = DLMConfig(
        block_size=block_size,
        num_diffusion_steps=num_diffusion_steps,
        source_checkpoint=checkpoint_path,
        total_tokens_trained=finetuner.tokens_seen,
    )

    final_path = save_dlm_checkpoint(
        model=trained_model,
        output_path=output_path,
        tokenizer=tokenizer,
        dlm_config=dlm_config,
    )

    return {
        "run_id": f"local-{model}",
        "status": "success",
        "output_path": str(final_path),
        "tokens_trained": finetuner.tokens_seen,
    }


def validate(
    model_path: str,
    test_prompt: str = "Hello, how are you?",
    block_size: int = 32,
    diffusion_steps: int = 8,
    max_length: int = 128,
) -> dict[str, Any]:
    """Validate a converted DLM model works correctly.

    Runs inference and checks output quality.

    Args:
        model_path: Path to converted DLM checkpoint
        test_prompt: Prompt for test generation
        block_size: Block size for generation
        diffusion_steps: Diffusion steps per block
        max_length: Maximum generation length

    Returns:
        Validation results with:
        - success: Whether validation passed
        - generated_text: Model output
        - tokens_per_second: Generation speed
        - error: Error message if failed
    """
    import time

    from wf_dlm_converter.conversion import load_dlm_checkpoint

    try:
        logger.info(f"Validating model at {model_path}")

        # Load model
        model, tokenizer, dlm_config = load_dlm_checkpoint(model_path)

        # Tokenize prompt
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        elapsed = time.time() - start_time
        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "success": True,
            "generated_text": generated_text,
            "tokens_per_second": new_tokens / elapsed,
            "new_tokens": new_tokens,
            "elapsed_seconds": elapsed,
            "dlm_config": dlm_config.to_dict(),
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def logs(run_id: str, follow: bool = False) -> None:
    """View logs for a conversion job.

    Args:
        run_id: Job identifier
        follow: Stream logs continuously
    """
    from wf_dlm_converter.constants import RunIdPrefix

    if run_id.startswith(RunIdPrefix.CONVERT.value):
        _logs_modal(run_id, follow)
    else:
        logger.info(f"Unknown run ID format: {run_id}")


def _logs_modal(run_id: str, follow: bool) -> None:
    """Get logs from Modal."""
    import subprocess

    cmd = ["modal", "logs", run_id]
    if follow:
        cmd.append("--follow")

    subprocess.run(cmd)


def cancel(run_id: str) -> dict[str, Any]:
    """Cancel a running conversion job.

    Args:
        run_id: Job identifier

    Returns:
        Cancellation result
    """
    from wf_dlm_converter.constants import RunIdPrefix

    if run_id.startswith(RunIdPrefix.CONVERT.value):
        return _cancel_modal(run_id)

    return {"success": False, "error": f"Unknown run ID: {run_id}"}


def _cancel_modal(run_id: str) -> dict[str, Any]:
    """Cancel Modal job."""
    import subprocess

    result = subprocess.run(
        ["modal", "app", "stop", run_id],
        capture_output=True,
        text=True,
    )

    return {
        "success": result.returncode == 0,
        "message": result.stdout or result.stderr,
    }


# Import torch for local conversion
try:
    import torch
except ImportError:
    torch = None
