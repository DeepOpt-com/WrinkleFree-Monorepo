#!/usr/bin/env python
"""Fast-dLLM v2 SFT Training Recipe.

Implements the training recipe from the Fast-dLLM v2 paper (arXiv:2509.26328):
- SFT on conversations with loss ONLY on assistant responses (prompts masked)
- Adds mask token |<MASK>| to tokenizer vocabulary
- Sets bd_size (block diffusion size) in model config
- Pads response sequences to multiples of bd_size with mask token
- Block diffusion (complementary masks, token shift) happens at INFERENCE, not training

Dataset: nvidia/Llama-Nemotron-Post-Training-Dataset (CC-BY-4.0)

Usage:
    # Basic (uses Hydra configs from configs/)
    uv run python scripts/train_dlm.py model=qwen3_4b source.path=hf://org/checkpoint

    # With overrides
    uv run python scripts/train_dlm.py model=smollm2_135m conversion.total_tokens=100000000

    # With W&B logging
    WANDB_API_KEY=xxx uv run python scripts/train_dlm.py model=qwen3_4b source.path=...
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

from cheapertraining.training import PlateauEarlyStopping

# Try to import wrinklefree for quantization control (BitNet integration)
try:
    from wrinklefree.quantization.lambda_warmup import LambdaWarmup, set_global_lambda_warmup
    HAS_WRINKLEFREE = True
except ImportError:
    HAS_WRINKLEFREE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

if not HAS_WRINKLEFREE:
    logger.warning("Could not import wrinklefree. Quantization warmup will be disabled (lambda=1.0).")

# Fast-dLLM v2 constants
MASK_TOKEN = "|<MASK>|"
DEFAULT_SEQ_LENGTH = 512  # Official script uses 512
DEFAULT_WARMUP_RATIO = 0.03  # Official: 3% warmup
GCS_UPLOAD_INTERVAL = 1000  # Upload to GCS every N steps
BATCH_PROBE_REDUCTION = 0.8  # Reduce batch by 20% on OOM


class ZClip:
    """Adaptive gradient clipping using z-score anomaly detection.

    Dynamically adjusts clipping threshold based on gradient norm statistics.
    Detects and clips gradient spikes that exceed z_threshold standard deviations
    from the running mean.

    Reference: arXiv:2504.02507 (ZClip: Adaptive Spike Mitigation for LLM Pre-Training)
    """

    def __init__(self, z_threshold: float = 3.0, ema_decay: float = 0.99):
        self.z_threshold = z_threshold
        self.ema_decay = ema_decay
        self.ema_mean = None
        self.ema_var = None

    def clip(self, model) -> tuple[float, float]:
        """Clip gradients and return (raw_norm, clipped_norm).

        Args:
            model: PyTorch model with gradients computed

        Returns:
            Tuple of (raw gradient norm, clipped gradient norm)
        """
        # Compute raw gradient norm
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        if not params_with_grad:
            return 0.0, 0.0

        raw_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, float("inf"))
        raw_norm_val = raw_norm.item() if hasattr(raw_norm, "item") else raw_norm

        # Initialize EMA on first call
        if self.ema_mean is None:
            self.ema_mean = raw_norm_val
            self.ema_var = 0.0
            return raw_norm_val, raw_norm_val

        # Update EMA statistics
        self.ema_mean = self.ema_decay * self.ema_mean + (1 - self.ema_decay) * raw_norm_val
        self.ema_var = self.ema_decay * self.ema_var + (1 - self.ema_decay) * (raw_norm_val - self.ema_mean) ** 2

        # Z-score anomaly detection
        std = (self.ema_var + 1e-8) ** 0.5
        z_score = (raw_norm_val - self.ema_mean) / std

        if z_score > self.z_threshold:
            # Clip to threshold
            clip_val = self.ema_mean + self.z_threshold * std
            torch.nn.utils.clip_grad_norm_(params_with_grad, clip_val)
            return raw_norm_val, clip_val

        return raw_norm_val, raw_norm_val


def set_seed(seed: int):
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set global random seed to {seed}")


def get_gpu_memory_gb() -> float:
    """Get GPU VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


def probe_batch_size(
    model,
    tokenizer,
    starting_batch_size: int,
    seq_length: int,
    min_batch_size: int = 1,
) -> int:
    """Find maximum batch size that fits in GPU memory.

    Runs a few probe forward/backward passes, reducing batch size on OOM.
    Returns a safe batch size (95% of max to leave headroom).
    """
    import gc

    batch_size = starting_batch_size
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    logger.info(f"Probing batch size (starting={starting_batch_size}, seq_len={seq_length})")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({get_gpu_memory_gb():.1f}GB)")

    while batch_size >= min_batch_size:
        try:
            gc.collect()
            torch.cuda.empty_cache()

            # Create dummy batch
            dummy_ids = torch.full((batch_size, seq_length), pad_id, dtype=torch.long, device="cuda")
            dummy_mask = torch.ones_like(dummy_ids)
            dummy_labels = dummy_ids.clone()

            # Forward + backward pass
            outputs = model(input_ids=dummy_ids, attention_mask=dummy_mask, labels=dummy_labels)
            outputs.loss.backward()
            torch.cuda.synchronize()

            # Clean up
            del dummy_ids, dummy_mask, dummy_labels, outputs
            gc.collect()
            torch.cuda.empty_cache()

            # Leave 5% headroom
            safe_batch = max(min_batch_size, int(batch_size * 0.95))
            logger.info(f"✓ batch_size={batch_size} works, using {safe_batch} with headroom")
            return safe_batch

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                new_batch = max(min_batch_size, int(batch_size * BATCH_PROBE_REDUCTION))
                if new_batch == batch_size:
                    new_batch = batch_size - 1
                logger.warning(f"OOM at batch_size={batch_size}, trying {new_batch}")
                batch_size = new_batch

                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise

    raise RuntimeError(f"Even batch_size={min_batch_size} causes OOM!")


def upload_to_gcs(local_path: Path, model_name: str, checkpoint_name: str = "checkpoint-latest") -> bool:
    """Upload checkpoint to GCS if bucket is configured.

    Returns True if upload succeeded, False otherwise.
    """
    gcs_bucket = os.environ.get("GCS_BUCKET")
    if not gcs_bucket:
        return False

    gcs_path = f"gs://{gcs_bucket}/dlm/{model_name}/{checkpoint_name}/"
    logger.info(f"Uploading checkpoint to {gcs_path}")

    try:
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{local_path}/*", gcs_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        if result.returncode == 0:
            logger.info(f"GCS upload complete: {gcs_path}")
            return True
        else:
            logger.warning(f"gsutil failed: {result.stderr}")
            # Try gcloud storage as fallback
            result = subprocess.run(
                ["gcloud", "storage", "cp", "-r", f"{local_path}/*", gcs_path],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"GCS upload complete (gcloud): {gcs_path}")
                return True
            logger.warning(f"gcloud storage also failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.warning("gsutil/gcloud not found, skipping GCS upload")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("GCS upload timed out")
        return False


def find_latest_gcs_checkpoint(model_name: str) -> str | None:
    """Find the latest checkpoint on GCS. Returns GCS URI or None."""
    gcs_bucket = os.environ.get("GCS_BUCKET")
    if not gcs_bucket:
        return None

    gcs_prefix = f"gs://{gcs_bucket}/dlm/{model_name}/"
    try:
        result = subprocess.run(
            ["gsutil", "ls", "-d", f"{gcs_prefix}checkpoint-step-*"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["gcloud", "storage", "ls", f"{gcs_prefix}checkpoint-step-*"],
                capture_output=True,
                text=True,
                timeout=30,
            )

        if result.returncode == 0 and result.stdout:
            checkpoints = result.stdout.strip().split("\n")
            steps = []
            for cp in checkpoints:
                try:
                    step_str = cp.rstrip("/").split("-")[-1]
                    steps.append((int(step_str), cp.rstrip("/")))
                except ValueError:
                    continue

            if steps:
                latest = sorted(steps, key=lambda x: x[0])[-1]
                logger.info(f"Found GCS checkpoint at step {latest[0]}: {latest[1]}")
                return latest[1]

    except (subprocess.SubprocessError, ValueError) as e:
        logger.warning(f"Failed to list GCS checkpoints: {e}")

    return None


def download_gcs_checkpoint(gcs_uri: str, local_dir: Path) -> bool:
    """Download checkpoint from GCS to local directory."""
    logger.info(f"Downloading checkpoint from {gcs_uri} to {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{gcs_uri}/*", str(local_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["gcloud", "storage", "cp", "-r", f"{gcs_uri}/*", str(local_dir)],
                capture_output=True,
                text=True,
                timeout=600,
            )

        if result.returncode == 0:
            logger.info("Checkpoint download successful")
            return True
        else:
            logger.error(f"Download failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Checkpoint download timed out")
        return False


class ConversationDataset(IterableDataset):
    """SFT dataset with response-only loss for Fast-dLLM v2.

    Only computes loss on assistant responses (prompts get labels=-100).
    Response tokens are padded to block_size multiples with mask token.
    Tracks raw_samples_seen for efficient resume via dataset.skip().
    """

    def __init__(self, hf_dataset, tokenizer, max_length, block_size, mask_id, pad_id):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.raw_samples_seen = 0

    def __iter__(self):
        for example in self.dataset:
            self.raw_samples_seen += 1
            # NVIDIA dataset format: input (messages), output (response), system_prompt
            messages = example.get("input", [])
            output = example.get("output", "")
            system = example.get("system_prompt", "")

            if not messages or not output:
                continue

            # Build full conversation
            if system:
                messages = [{"role": "system", "content": system}] + list(messages)
            messages = list(messages) + [{"role": "assistant", "content": output}]

            # Apply chat template to get full text
            try:
                full_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                # Get prompt-only text (without assistant response)
                prompt_messages = messages[:-1]
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                continue  # Skip if chat template fails

            # Tokenize both
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            prompt_len = len(prompt_ids)
            response_len = len(full_ids) - prompt_len

            if response_len <= 0:
                continue  # Skip invalid samples

            # Truncate to max_length first, leaving room for block padding
            max_content = self.max_length - self.block_size
            if len(full_ids) > max_content:
                full_ids = full_ids[:max_content]
                # Update prompt_len if prompt was truncated too
                prompt_len = min(prompt_len, len(full_ids))
                response_len = len(full_ids) - prompt_len

            # Skip if response was fully truncated
            if response_len <= 0:
                continue

            # Pad response to block_size multiple with mask_id (Fast-dLLM v2 requirement)
            pad_len = (self.block_size - response_len % self.block_size) % self.block_size
            if pad_len > 0:
                full_ids = full_ids + [self.mask_id] * pad_len

            # Ensure we don't exceed max_length after padding
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]

            # Create labels: -100 for prompt, actual tokens for response
            seq_len = len(full_ids)
            labels = [-100] * prompt_len + full_ids[prompt_len:seq_len]

            # Pad to max_length with pad token (no loss on these)
            final_pad = self.max_length - seq_len
            if final_pad > 0:
                full_ids = full_ids + [self.pad_id] * final_pad
                labels = labels + [-100] * final_pad

            attention_mask = [1] * seq_len + [0] * final_pad

            # Sanity check: all tensors must have max_length size
            assert len(full_ids) == self.max_length, f"input_ids size {len(full_ids)} != {self.max_length}"
            assert len(labels) == self.max_length, f"labels size {len(labels)} != {self.max_length}"
            assert len(attention_mask) == self.max_length, f"attention_mask size {len(attention_mask)} != {self.max_length}"

            yield {
                "input_ids": torch.tensor(full_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
            }


class TextDataset(IterableDataset):
    """Simple text dataset for pretraining-style loss (loss on all tokens).

    Used for cleaner signal with datasets like FineWeb-Edu that don't have
    conversation structure. Loss is computed on all tokens (no masking).
    """

    def __init__(self, hf_dataset, tokenizer, max_length, block_size, mask_id, pad_id):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.raw_samples_seen = 0

    def __iter__(self):
        for example in self.dataset:
            self.raw_samples_seen += 1
            # FineWeb-Edu format: just "text" field
            text = example.get("text", "")
            if not text or len(text) < 50:  # Skip very short texts
                continue

            # Tokenize
            tokens = self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]

            if len(tokens) < 10:
                continue

            # Truncate to max_length, leaving room for block padding
            max_content = self.max_length - self.block_size
            if len(tokens) > max_content:
                tokens = tokens[:max_content]

            # Pad to block_size multiple with mask_id (Fast-dLLM v2 requirement)
            content_len = len(tokens)
            pad_len = (self.block_size - content_len % self.block_size) % self.block_size
            if pad_len > 0:
                tokens = tokens + [self.mask_id] * pad_len

            # Ensure we don't exceed max_length after padding
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]

            seq_len = len(tokens)

            # Labels: loss on all tokens (shifted by 1 in model forward)
            labels = tokens.copy()

            # Pad to max_length with pad token (no loss on padding)
            final_pad = self.max_length - seq_len
            if final_pad > 0:
                tokens = tokens + [self.pad_id] * final_pad
                labels = labels + [-100] * final_pad

            attention_mask = [1] * seq_len + [0] * final_pad

            assert len(tokens) == self.max_length
            assert len(labels) == self.max_length
            assert len(attention_mask) == self.max_length

            yield {
                "input_ids": torch.tensor(tokens),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
            }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def train(
    model_path: str,
    output_dir: str,
    total_tokens: int = 1_000_000_000,  # Official: ~1B tokens
    block_size: int = 32,
    learning_rate: float = 2e-5,  # Official: 2e-5
    max_seq_length: int = DEFAULT_SEQ_LENGTH,  # Official: 512
    batch_size: int = 8,
    gradient_accumulation_steps: int = 16,  # Effective batch ~128
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,  # Official: 3%
    scheduler_type: str = "constant",  # Official: constant_with_warmup
    auto_batch_size: bool = True,  # Enable dynamic batch probing
    resume: bool = True,  # Auto-resume from GCS checkpoint if available
    seed: int = 42,
    early_stopping_cfg: dict | None = None,  # Early stopping config from Hydra
    quantization_warmup_steps: int = 0,  # Lambda warmup for BitNet (0 = disabled)
):
    """Run Fast-dLLM v2 SFT training."""
    set_seed(seed)

    model_name = model_path.split("/")[-1]
    final_output = Path(output_dir)
    final_output.mkdir(parents=True, exist_ok=True)

    # === CHECK FOR RESUME CHECKPOINT ===
    resume_checkpoint = None
    resume_step = 0
    resume_tokens = 0
    resume_raw_samples = 0
    resume_state = None

    if resume:
        # Check GCS for latest checkpoint
        gcs_checkpoint = find_latest_gcs_checkpoint(model_name)
        if gcs_checkpoint:
            local_resume_dir = final_output / "checkpoint-resume"
            if download_gcs_checkpoint(gcs_checkpoint, local_resume_dir):
                resume_checkpoint = local_resume_dir
                # Load trainer state if available
                state_path = local_resume_dir / "trainer_state.pt"
                if state_path.exists():
                    resume_state = torch.load(state_path, weights_only=False)
                    resume_step = resume_state.get("step", 0)
                    resume_tokens = resume_state.get("tokens_seen", 0)
                    resume_raw_samples = resume_state.get("raw_samples_seen", 0)
                    logger.info(f"Will resume from step {resume_step}, tokens {resume_tokens:,}, raw samples {resume_raw_samples:,}")

                    # Restore RNG state if available
                    if "rng_state" in resume_state:
                        rng = resume_state["rng_state"]
                        random.setstate(rng["python"])
                        np.random.set_state(rng["numpy"])
                        torch.set_rng_state(rng["torch"])
                        if torch.cuda.is_available() and rng.get("cuda") is not None:
                            torch.cuda.set_rng_state_all(rng["cuda"])
                        logger.info("Restored RNG state from checkpoint")
                else:
                    # Extract step from checkpoint name
                    try:
                        resume_step = int(gcs_checkpoint.rstrip("/").split("-")[-1])
                        logger.info(f"Will resume from step {resume_step} (no trainer_state.pt)")
                    except ValueError:
                        pass

    # Initialize wandb if API key is available
    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    if use_wandb:
        import wandb

        run_name = f"dlm-v2-{model_name}-{total_tokens // 1_000_000_000}B"
        if resume_step > 0:
            run_name += f"-resume-{resume_step}"

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "wrinklefree-dlm"),
            name=run_name,
            config={
                "model": model_path,
                "total_tokens": total_tokens,
                "block_size": block_size,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_seq_length": max_seq_length,
                "training_method": "fast-dllm-v2-sft",
                "resume_step": resume_step,
            },
        )
        logger.info(f"W&B run initialized: {wandb.run.url}")
    else:
        logger.info("W&B disabled (no WANDB_API_KEY)")

    logger.info("=== Fast-dLLM v2 SFT Training ===")
    logger.info(f"Model: {model_path}")
    logger.info(f"Tokens: {total_tokens:,}")
    logger.info(f"Block size (bd_size): {block_size}")
    logger.info(f"Batch size: {batch_size}, Grad accum: {gradient_accumulation_steps}")
    if resume_step > 0:
        logger.info(f"Resuming from step {resume_step}")

    # Load model and tokenizer (from checkpoint if resuming)
    load_path = str(resume_checkpoint) if resume_checkpoint else model_path
    logger.info(f"Loading checkpoint from {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
    ).cuda()

    # === FAST-DLLM V2 SETUP ===

    # 1. Add mask token to tokenizer
    if MASK_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Added mask token {MASK_TOKEN} to vocabulary")
    mask_id = tokenizer.encode(MASK_TOKEN, add_special_tokens=False)[0]
    logger.info(f"Mask token ID: {mask_id}")

    # === SMART INITIALIZATION (Fix for BitNet Instability) ===
    # Initialize mask token embedding to mean of existing vocabulary.
    # Random initialization creates activation outliers that break BitNet's
    # per-token quantization (scale = 127 / max(|x|)), causing loss spikes.
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and input_embeddings.weight.shape[0] > 1:
        with torch.no_grad():
            if mask_id < input_embeddings.weight.shape[0]:
                # Calculate mean of all tokens except the new one (if it's last)
                limit_idx = input_embeddings.weight.shape[0] - 1 if mask_id == input_embeddings.weight.shape[0] - 1 else None
                mean_embedding = input_embeddings.weight[:limit_idx].mean(dim=0)
                input_embeddings.weight[mask_id] = mean_embedding
                logger.info(f"Initialized {MASK_TOKEN} embedding to mean of vocab (norm={mean_embedding.norm().item():.2f})")
            else:
                logger.warning(f"Mask token ID {mask_id} out of embedding range, skipping smart initialization")
    else:
        logger.warning("Could not access input embeddings for smart initialization")

    # 2. Set bd_size in model config
    model.config.bd_size = block_size
    logger.info(f"Set bd_size = {block_size} in model config")

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Auto-probe batch size to maximize GPU utilization
    if auto_batch_size:
        # Start with aggressive batch size based on VRAM
        vram_gb = get_gpu_memory_gb()
        # Heuristic: 1 batch element ~ 100MB for 2B model at seq_len=512
        starting_batch = max(4, int(vram_gb * 0.6))  # Use 60% of VRAM heuristic
        logger.info(f"Auto-probing batch size (VRAM={vram_gb:.1f}GB, starting={starting_batch})")

        model.zero_grad()  # Clear any stale gradients
        optimal_batch = probe_batch_size(model, tokenizer, starting_batch, max_seq_length)

        if optimal_batch > batch_size:
            # Scale up batch, reduce grad_accum to keep effective batch constant
            effective_batch = batch_size * gradient_accumulation_steps
            new_grad_accum = max(1, effective_batch // optimal_batch)
            logger.info(f"Scaling up: batch_size {batch_size}->{optimal_batch}, grad_accum {gradient_accumulation_steps}->{new_grad_accum}")
            batch_size = optimal_batch
            gradient_accumulation_steps = new_grad_accum
        else:
            logger.info(f"Keeping original batch_size={batch_size} (optimal={optimal_batch})")

    # Calculate training steps
    tokens_per_step = batch_size * gradient_accumulation_steps * max_seq_length
    total_steps = total_tokens // tokens_per_step

    # Calculate warmup steps from ratio (Official: 3%)
    # Skip warmup if resuming from checkpoint
    if resume_step > 0:
        actual_warmup_steps = 0  # No warmup when resuming
        logger.info("Skipping warmup (resuming from checkpoint)")
    else:
        actual_warmup_steps = int(total_steps * warmup_ratio)

    logger.info(f"Training for {total_steps} steps ({tokens_per_step:,} tokens/step)")
    logger.info(f"Warmup steps: {actual_warmup_steps} ({warmup_ratio*100:.0f}% of total, scheduler: {scheduler_type})")

    # Load training data - NVIDIA Llama-Nemotron (per Fast-dLLM v2 paper)
    # Uses SFT with response-only loss, MASK padding only on response tokens
    logger.info("Loading training data (Llama-Nemotron-Post-Training-Dataset)")
    splits = ["code", "math", "science", "chat", "safety"]
    datasets_list = [
        load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=s, streaming=True)
        for s in splits
    ]
    dataset = interleave_datasets(datasets_list, seed=seed)

    # Deterministic shuffle for reproducibility
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    # NOTE: We don't skip samples on resume because:
    # 1. interleave_datasets + shuffle + skip has unreliable behavior
    # 2. The shuffle buffer (10K) makes exact position restoration impossible
    # 3. Training on some duplicate data is acceptable - model weights are restored
    # The training loop will start from resume_step, so progress is maintained.
    if resume_raw_samples > 0:
        logger.info(f"Resuming from step {resume_step} (not skipping samples - shuffle buffer makes exact restore impossible)")

    tokenized_dataset = ConversationDataset(
        dataset, tokenizer, max_seq_length, block_size, mask_id, pad_id
    )
    # Track cumulative samples across runs (for checkpoint metadata)
    tokenized_dataset.raw_samples_seen = resume_raw_samples

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )
    data_iter = iter(dataloader)

    # Setup optimizer (AdamW per Fast-dLLM v2 paper)
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    logger.info(f"Using AdamW optimizer (lr={learning_rate})")

    # Create scheduler based on config
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=total_steps
        )
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=actual_warmup_steps)

    # === LOAD OPTIMIZER/SCHEDULER STATE IF RESUMING ===
    if resume_state and "optimizer" in resume_state:
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])

            # Reset LR to config value (checkpoint may have different LR)
            old_lr = optimizer.param_groups[0]["lr"]
            if old_lr != learning_rate:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate
                    if "initial_lr" in param_group:
                        param_group["initial_lr"] = learning_rate
                logger.info(f"Reset LR from checkpoint value {old_lr} to config value {learning_rate}")

            # Update scheduler's base_lrs to match config LR
            if hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [learning_rate] * len(scheduler.base_lrs)

            logger.info("Loaded optimizer and scheduler state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")

    # Early stopping setup
    early_stop_cfg = early_stopping_cfg or {}
    early_stopper = PlateauEarlyStopping(
        patience=early_stop_cfg.get("patience", 5),
        min_delta=early_stop_cfg.get("min_delta", 0.01),
        mode="min",
        min_evals=early_stop_cfg.get("min_evals", 10),
        enabled=early_stop_cfg.get("enabled", False),
        rank=0,
    )
    if early_stopper.enabled:
        logger.info(
            f"Early stopping enabled: patience={early_stopper.patience}, "
            f"min_delta={early_stopper.min_delta}"
        )

    # === BITNET STABILITY SETUP ===
    # Lambda warmup for gradual quantization
    lambda_warmup_scheduler = None
    if HAS_WRINKLEFREE and quantization_warmup_steps > 0:
        logger.info(f"Initializing quantization warmup: {quantization_warmup_steps} steps")
        lambda_warmup_scheduler = LambdaWarmup(
            warmup_steps=quantization_warmup_steps,
            min_lambda=0.0,
            max_lambda=1.0,
            schedule="linear",
        )
        set_global_lambda_warmup(lambda_warmup_scheduler)

        # If resuming, fast-forward warmup to match step
        if resume_step > 0:
            for _ in range(min(resume_step, quantization_warmup_steps)):
                lambda_warmup_scheduler.step()
            logger.info(f"Fast-forwarded lambda warmup to step {resume_step} (lambda={lambda_warmup_scheduler.lambda_val:.3f})")
    elif HAS_WRINKLEFREE:
        # Ensure full quantization if no warmup requested
        set_global_lambda_warmup(None)
        logger.info("Quantization warmup disabled (lambda=1.0)")

    # ZClip for adaptive gradient clipping
    zclip = ZClip(z_threshold=3.0, ema_decay=0.99)
    logger.info("ZClip adaptive gradient clipping enabled (z_threshold=3.0)")

    # Training loop
    model.train()
    tokens_seen = resume_tokens
    step = resume_step
    running_loss = 0.0
    best_loss = resume_state.get("best_loss", float("inf")) if resume_state else float("inf")

    # Stability tracking
    loss_ema = None
    loss_ema_alpha = 0.99  # EMA decay factor
    prev_loss = None
    step_start_time = None

    logger.info("Starting training loop")
    pbar = tqdm(total=total_steps, initial=step, desc="Training (SFT)")

    optimizer.zero_grad()
    accum_count = 0

    while step < total_steps:
        if step_start_time is None:
            step_start_time = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        running_loss += loss.item()
        tokens_seen += attention_mask.sum().item()
        accum_count += 1

        if accum_count >= gradient_accumulation_steps:
            # Adaptive gradient clipping with ZClip (replaces fixed clip_grad_norm_)
            raw_grad_norm, clipped_grad_norm = zclip.clip(model)
            grad_norm = clipped_grad_norm  # For logging compatibility

            optimizer.step()
            scheduler.step()

            # Step lambda warmup for gradual quantization
            if lambda_warmup_scheduler is not None:
                lambda_warmup_scheduler.step()

            optimizer.zero_grad()

            step += 1
            accum_count = 0

            avg_loss = running_loss
            if avg_loss < best_loss:
                best_loss = avg_loss

            # Calculate step time and throughput
            step_time = time.time() - step_start_time
            tokens_this_step = batch_size * gradient_accumulation_steps * max_seq_length
            tokens_per_sec = tokens_this_step / step_time if step_time > 0 else 0
            step_start_time = time.time()

            # Update loss EMA for stability tracking
            if loss_ema is None:
                loss_ema = avg_loss
            else:
                loss_ema = loss_ema_alpha * loss_ema + (1 - loss_ema_alpha) * avg_loss

            # Detect loss spikes (>2x previous loss)
            loss_spike = 1 if (prev_loss is not None and avg_loss > 2.0 * prev_loss) else 0
            prev_loss = avg_loss

            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "tokens": f"{tokens_seen:,}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if use_wandb:
                # Get current lambda value for quantization tracking
                current_lambda = lambda_warmup_scheduler.lambda_val if lambda_warmup_scheduler else 1.0

                wandb.log({
                    # Core metrics
                    "train/loss": avg_loss,
                    "train/tokens": tokens_seen,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": step,
                    # Stability metrics (ZClip + quantization)
                    "train/grad_norm_raw": raw_grad_norm,
                    "train/grad_norm_clipped": clipped_grad_norm,
                    "train/grad_norm": grad_norm,  # For backward compatibility
                    "train/loss_ema": loss_ema,
                    "train/loss_spike": loss_spike,
                    "train/lambda": current_lambda,  # Quantization lambda
                    # Throughput metrics
                    "train/tokens_per_second": tokens_per_sec,
                    "train/step_time": step_time,
                    # System metrics
                    "system/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "system/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
                })

            running_loss = 0.0

            # Early stopping check (using smoothed loss)
            if early_stopper.check(loss_ema, step):
                logger.warning("Stopping training early due to loss plateau.")
                early_stopper.save_json(final_output)
                # Save final checkpoint before exiting
                checkpoint_dir = final_output / "checkpoint-early-stop"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                break

            # Save checkpoint every 1000 steps
            if step % GCS_UPLOAD_INTERVAL == 0:
                logger.info(f"Step {step}: Saving checkpoint")
                checkpoint_dir = final_output / "checkpoint-latest"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

                # Save trainer state for resume (including RNG for reproducibility)
                torch.save({
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "raw_samples_seen": tokenized_dataset.raw_samples_seen,
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                }, checkpoint_dir / "trainer_state.pt")

                # Upload to GCS periodically
                upload_to_gcs(checkpoint_dir, model_name, f"checkpoint-step-{step}")

    pbar.close()

    # Save final model
    logger.info("Saving final model")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    # Save DLM config
    dlm_config = {
        "bd_size": block_size,
        "mask_token": MASK_TOKEN,
        "mask_token_id": mask_id,
        "num_diffusion_steps": 8,
        "source_checkpoint": model_path,
        "total_tokens_trained": tokens_seen,
        "training_loss": best_loss,
        "training_method": "fast-dllm-v2-sft",
        "max_seq_length": max_seq_length,
    }
    with open(final_output / "dlm_config.json", "w") as f:
        json.dump(dlm_config, f, indent=2)

    logger.info(f"Model saved to {final_output}")
    logger.info(f"Fast-dLLM v2 config: bd_size={block_size}, mask_id={mask_id}")

    # Upload final model to GCS
    upload_to_gcs(final_output, model_name, "final")

    if use_wandb:
        wandb.finish()

    return {
        "output_path": str(final_output),
        "tokens_trained": tokens_seen,
        "final_loss": best_loss,
        "bd_size": block_size,
        "mask_token_id": mask_id,
    }


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for Fast-dLLM v2 training."""
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Resolve model path from config (fail loudly if missing)
    model_path = cfg.model.get("bitnet_checkpoint") or cfg.source.get("path")
    if not model_path:
        raise ValueError(
            "No model path! Set via CLI:\n"
            "  uv run python scripts/train_dlm.py source.path=hf://org/model\n"
            "  uv run python scripts/train_dlm.py model.bitnet_checkpoint=/path/to/ckpt"
        )

    # Resolve output directory
    output_dir = Path(cfg.output_dir) / cfg.model.name
    output_dir.mkdir(parents=True, exist_ok=True)

    result = train(
        model_path=model_path,
        output_dir=str(output_dir),
        total_tokens=cfg.conversion.total_tokens,
        block_size=cfg.model.block_size,
        learning_rate=cfg.conversion.optimizer.lr,
        max_seq_length=cfg.conversion.get("max_seq_length", DEFAULT_SEQ_LENGTH),
        batch_size=cfg.conversion.batch_size,
        gradient_accumulation_steps=cfg.conversion.gradient_accumulation_steps,
        warmup_ratio=cfg.conversion.scheduler.get("warmup_ratio", DEFAULT_WARMUP_RATIO),
        scheduler_type=cfg.conversion.scheduler.get("type", "constant"),
        seed=cfg.get("seed", 42),
        early_stopping_cfg=OmegaConf.to_container(cfg.conversion.get("early_stopping", {})),
        quantization_warmup_steps=cfg.conversion.get("quantization_warmup_steps", 0),
    )

    logger.info(f"Training complete!")
    logger.info(f"Output: {result['output_path']}")
    logger.info(f"Tokens trained: {result['tokens_trained']:,}")
    logger.info(f"Final loss: {result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
