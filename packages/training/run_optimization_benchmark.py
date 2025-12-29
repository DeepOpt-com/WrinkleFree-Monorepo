#!/usr/bin/env python3
"""Modal script to run optimization benchmarks on A10G.

Usage:
    modal run run_optimization_benchmark.py --stage all
    modal run run_optimization_benchmark.py --stage 1.9
"""

import modal
import os

app = modal.App("wrinklefree-optimization-benchmark")

# Use the same image as the training deployer
benchmark_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        # Core ML
        "torch>=2.5.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "safetensors>=0.4.0",
        # Training
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "bitsandbytes>=0.43.0",
        # Logging
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
        # HTTP client for vLLM
        "httpx>=0.27.0",
        # Fast HF downloads
        "hf_transfer>=0.1.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)


@app.function(
    image=benchmark_image,
    gpu="A10G",
    timeout=2 * 60 * 60,  # 2 hours
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def run_benchmark(
    stage: str = "all",
    model: str = "smollm2_135m",
    num_steps: int = 50,
    warmup_steps: int = 3,
):
    """Run optimization benchmarks for specified stage(s)."""
    import subprocess
    import json
    import time
    import gc
    from dataclasses import dataclass, field, asdict
    from typing import Optional

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader, TensorDataset

    @dataclass
    class BenchmarkResult:
        name: str
        stage: str
        steps_per_second: float = 0.0
        tokens_per_second: float = 0.0
        peak_memory_gb: float = 0.0
        time_to_first_step_ms: float = 0.0
        avg_step_time_ms: float = 0.0
        equivalence_cosine: float = 0.0
        speedup: float = 1.0
        config: dict = field(default_factory=dict)

    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_flat = a.detach().float().flatten()
        b_flat = b.detach().float().flatten()
        if a_flat.norm() < 1e-8 or b_flat.norm() < 1e-8:
            return 1.0 if (a_flat.norm() < 1e-8 and b_flat.norm() < 1e-8) else 0.0
        return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)))

    def get_optimization_configs(stage: str) -> list[dict]:
        if stage == "1.9":
            return [
                {"name": "Baseline", "is_baseline": True},
                {"name": "torch.compile (default)", "torch_compile": True, "compile_mode": "default"},
                {"name": "torch.compile (reduce-overhead)", "torch_compile": True, "compile_mode": "reduce-overhead"},
                {"name": "torch.compile (max-autotune)", "torch_compile": True, "compile_mode": "max-autotune"},
                {"name": "Gradient Checkpointing", "gradient_checkpointing": True},
                {"name": "compile + grad_ckpt", "torch_compile": True, "compile_mode": "reduce-overhead", "gradient_checkpointing": True},
                {"name": "compile (max) + grad_ckpt", "torch_compile": True, "compile_mode": "max-autotune", "gradient_checkpointing": True},
                {"name": "All opts (reduce)", "torch_compile": True, "compile_mode": "reduce-overhead", "gradient_checkpointing": True, "fullgraph": False},
                {"name": "All opts (max)", "torch_compile": True, "compile_mode": "max-autotune", "gradient_checkpointing": True, "fullgraph": False},
                {"name": "Max + fullgraph", "torch_compile": True, "compile_mode": "max-autotune", "fullgraph": True},
            ]
        elif stage == "2":
            return [
                {"name": "Baseline", "is_baseline": True},
                {"name": "torch.compile (default)", "torch_compile": True, "compile_mode": "default"},
                {"name": "torch.compile (reduce-overhead)", "torch_compile": True, "compile_mode": "reduce-overhead"},
                {"name": "torch.compile (max-autotune)", "torch_compile": True, "compile_mode": "max-autotune"},
                {"name": "max-autotune + fullgraph", "torch_compile": True, "compile_mode": "max-autotune", "fullgraph": True},
                {"name": "reduce-overhead + fullgraph", "torch_compile": True, "compile_mode": "reduce-overhead", "fullgraph": True},
                {"name": "Gradient Checkpointing", "gradient_checkpointing": True},
                {"name": "compile + grad_ckpt", "torch_compile": True, "compile_mode": "reduce-overhead", "gradient_checkpointing": True},
                {"name": "max + grad_ckpt", "torch_compile": True, "compile_mode": "max-autotune", "gradient_checkpointing": True},
                {"name": "All Stage 2 opts", "torch_compile": True, "compile_mode": "max-autotune", "fullgraph": True, "gradient_checkpointing": True},
            ]
        elif stage == "3":
            return [
                {"name": "Baseline", "is_baseline": True},
                {"name": "torch.compile (default)", "torch_compile": True, "compile_mode": "default"},
                {"name": "torch.compile (reduce-overhead)", "torch_compile": True, "compile_mode": "reduce-overhead"},
                {"name": "torch.compile (max-autotune)", "torch_compile": True, "compile_mode": "max-autotune"},
                {"name": "Gradient Checkpointing", "gradient_checkpointing": True},
                {"name": "compile + grad_ckpt", "torch_compile": True, "compile_mode": "reduce-overhead", "gradient_checkpointing": True},
                {"name": "max + grad_ckpt", "torch_compile": True, "compile_mode": "max-autotune", "gradient_checkpointing": True},
                {"name": "max + fullgraph", "torch_compile": True, "compile_mode": "max-autotune", "fullgraph": True},
                {"name": "reduce + fullgraph + ckpt", "torch_compile": True, "compile_mode": "reduce-overhead", "fullgraph": True, "gradient_checkpointing": True},
                {"name": "All Stage 3 opts", "torch_compile": True, "compile_mode": "max-autotune", "fullgraph": True, "gradient_checkpointing": True},
            ]
        return []

    def run_single_benchmark(
        stage: str,
        opt_config: dict,
        model_name: str,
        num_steps: int,
        warmup_steps: int,
    ) -> BenchmarkResult:
        """Run a single benchmark iteration."""
        print(f"\n{'='*60}")
        print(f"Running: {opt_config['name']}")
        print(f"Config: {opt_config}")
        print(f"{'='*60}")

        # Load model
        hf_model_name = "HuggingFaceTB/SmolLM2-135M"
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.cuda()

        # Enable gradient checkpointing if requested
        if opt_config.get("gradient_checkpointing"):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                print("  Gradient checkpointing enabled")

        # Apply torch.compile if requested
        if opt_config.get("torch_compile"):
            mode = opt_config.get("compile_mode", "reduce-overhead")
            fullgraph = opt_config.get("fullgraph", False)
            print(f"  Applying torch.compile(mode={mode}, fullgraph={fullgraph})")
            model = torch.compile(model, mode=mode, fullgraph=fullgraph)

        # Create synthetic data
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = 4
        seq_len = 512
        num_samples = 100

        input_ids = torch.randint(0, tokenizer.vocab_size, (num_samples, seq_len))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: {
                "input_ids": torch.stack([s[0] for s in x]).cuda(),
                "attention_mask": torch.stack([s[1] for s in x]).cuda(),
                "labels": torch.stack([s[2] for s in x]).cuda(),
            },
        )

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Warmup (for torch.compile)
        print(f"  Warming up ({warmup_steps} steps)...")
        warmup_start = time.perf_counter()
        model.train()
        data_iter = iter(dataloader)
        for _ in range(warmup_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )
            loss.backward()
            model.zero_grad()

        warmup_time = time.perf_counter() - warmup_start
        time_to_first_step = (warmup_time / warmup_steps) * 1000  # ms

        # Timed benchmark
        print(f"  Running benchmark ({num_steps} steps)...")
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        step_times = []
        total_tokens = 0

        data_iter = iter(dataloader)
        for step in range(num_steps):
            step_start = time.perf_counter()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            total_tokens += batch["input_ids"].numel()

            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )
            loss.backward()
            model.zero_grad()

            torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        steps_per_second = num_steps / total_time
        tokens_per_second = total_tokens / total_time
        avg_step_time = (sum(step_times) / len(step_times)) * 1000  # ms
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB

        # Equivalence check (determinism)
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(dataloader))
            out1 = model(input_ids=test_batch["input_ids"], attention_mask=test_batch["attention_mask"])
            out2 = model(input_ids=test_batch["input_ids"], attention_mask=test_batch["attention_mask"])
            logits1 = out1.logits if hasattr(out1, "logits") else out1["logits"]
            logits2 = out2.logits if hasattr(out2, "logits") else out2["logits"]
            equiv_cosine = cosine_similarity(logits1, logits2)

        result = BenchmarkResult(
            name=opt_config["name"],
            stage=stage,
            steps_per_second=steps_per_second,
            tokens_per_second=tokens_per_second,
            peak_memory_gb=peak_memory,
            time_to_first_step_ms=time_to_first_step,
            avg_step_time_ms=avg_step_time,
            equivalence_cosine=equiv_cosine,
            config=opt_config,
        )

        print(f"  Results: {steps_per_second:.2f} steps/s, {peak_memory:.2f} GB, cosine={equiv_cosine:.4f}")

        # Cleanup
        del model, dataloader, dataset
        gc.collect()
        torch.cuda.empty_cache()

        return result

    # Main benchmark loop
    print("=" * 70)
    print("WRINKLEFREE OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model: {model}")
    print(f"Steps per iteration: {num_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print("=" * 70)

    stages = ["1.9", "2", "3"] if stage == "all" else [stage]
    all_results = []

    for current_stage in stages:
        print(f"\n{'#'*70}")
        print(f"# STAGE {current_stage} BENCHMARKS")
        print(f"{'#'*70}")

        configs = get_optimization_configs(current_stage)
        baseline_speed = None
        stage_results = []

        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] {config['name']}")

            try:
                result = run_single_benchmark(
                    stage=current_stage,
                    opt_config=config,
                    model_name=model,
                    num_steps=num_steps,
                    warmup_steps=warmup_steps,
                )

                if config.get("is_baseline"):
                    baseline_speed = result.steps_per_second

                if baseline_speed:
                    result.speedup = result.steps_per_second / baseline_speed

                stage_results.append(result)
                all_results.append(result)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Print stage summary
        print(f"\n{'='*70}")
        print(f"STAGE {current_stage} SUMMARY")
        print(f"{'='*70}")
        print(f"{'Optimization':<35} {'Steps/s':>10} {'Memory':>10} {'Speedup':>10}")
        print("-" * 70)
        for r in stage_results:
            speedup_str = f"{r.speedup:.2f}x" if r.speedup != 1.0 else "baseline"
            print(f"{r.name:<35} {r.steps_per_second:>10.2f} {r.peak_memory_gb:>9.2f}G {speedup_str:>10}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - ALL STAGES")
    print(f"{'='*70}")

    for current_stage in stages:
        stage_results = [r for r in all_results if r.stage == current_stage]
        if stage_results:
            best = max(stage_results, key=lambda x: x.steps_per_second)
            baseline = next((r for r in stage_results if r.config.get("is_baseline")), stage_results[0])
            print(f"\nStage {current_stage}:")
            print(f"  Baseline: {baseline.steps_per_second:.2f} steps/s, {baseline.peak_memory_gb:.2f} GB")
            print(f"  Best: {best.name} - {best.steps_per_second:.2f} steps/s ({best.speedup:.2f}x), {best.peak_memory_gb:.2f} GB")

    return [asdict(r) for r in all_results]


@app.local_entrypoint()
def main(
    stage: str = "all",
    model: str = "smollm2_135m",
    num_steps: int = 50,
    warmup_steps: int = 3,
):
    """Run benchmarks from CLI."""
    print(f"Launching optimization benchmark for stage={stage}")
    results = run_benchmark.remote(
        stage=stage,
        model=model,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    import json
    print(json.dumps(results, indent=2))
