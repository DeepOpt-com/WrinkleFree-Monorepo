#!/usr/bin/env python3
"""
Benchmark different GGUF quantization formats for BitNet models.

This script:
1. Converts a checkpoint to multiple GGUF formats
2. Benchmarks inference speed (tok/s) for each
3. Validates output quality against Python baseline
4. Reports size, speed, and quality metrics

Usage:
    python benchmark_quantization.py --checkpoint /path/to/checkpoint --output-dir ./benchmark_results
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Quantization formats to benchmark
QUANT_FORMATS = {
    "TQ1_0": {"bpw": 1.69, "description": "Ternary quantization (smallest)"},
    "TQ2_0": {"bpw": 2.06, "description": "Ternary quantization"},
    "IQ2_S": {"bpw": 2.5, "description": "2-bit integer quantization"},
    "IQ2_XS": {"bpw": 2.31, "description": "2-bit integer (extra small)"},
    "Q2_K": {"bpw": 2.96, "description": "2-bit k-quant"},
}

# Test prompts for benchmarking
TEST_PROMPTS = [
    "The capital of France is",
    "Write a haiku about computers:",
    "2 + 2 =",
    "def fibonacci(n):",
    "Once upon a time in a land far away,",
]


@dataclass
class BenchmarkResult:
    format: str
    file_size_mb: float
    prompt_tok_s: float
    gen_tok_s: float
    output_sample: str
    bpw: float


def run_llama_quantize(input_gguf: Path, output_gguf: Path, quant_type: str, llama_cpp_dir: Path) -> bool:
    """Run llama-quantize to convert to a specific format."""
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print(f"Error: llama-quantize not found at {quantize_bin}")
        return False

    cmd = [str(quantize_bin), str(input_gguf), str(output_gguf), quant_type]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def benchmark_format(
    gguf_path: Path,
    llama_cpp_dir: Path,
    n_predict: int = 50,
    n_runs: int = 3,
) -> Optional[BenchmarkResult]:
    """Benchmark a single GGUF file."""
    server_bin = llama_cpp_dir / "build" / "bin" / "llama-cli"
    if not server_bin.exists():
        print(f"Error: llama-cli not found at {server_bin}")
        return None

    file_size_mb = gguf_path.stat().st_size / (1024 * 1024)

    prompt_speeds = []
    gen_speeds = []
    output_sample = ""

    for i, prompt in enumerate(TEST_PROMPTS[:n_runs]):
        cmd = [
            str(server_bin),
            "-m", str(gguf_path),
            "-p", prompt,
            "-n", str(n_predict),
            "--no-display-prompt",
            "-t", "8",
        ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"  Run {i+1} failed: {result.stderr[:200]}")
            continue

        # Parse timing from stderr
        for line in result.stderr.split("\n"):
            if "prompt eval time" in line.lower():
                # Extract tok/s from line like "prompt eval time = 42.38 ms / 5 tokens (8.48 ms per token, 117.97 tokens per second)"
                if "tokens per second" in line:
                    try:
                        tok_s = float(line.split("tokens per second")[0].split(",")[-1].strip())
                        prompt_speeds.append(tok_s)
                    except:
                        pass
            elif "eval time" in line.lower() and "prompt" not in line.lower():
                if "tokens per second" in line:
                    try:
                        tok_s = float(line.split("tokens per second")[0].split(",")[-1].strip())
                        gen_speeds.append(tok_s)
                    except:
                        pass

        if i == 0:
            output_sample = result.stdout[:200]

    if not prompt_speeds or not gen_speeds:
        return None

    # Extract format name from filename
    format_name = gguf_path.stem.split("-")[-1].upper()

    return BenchmarkResult(
        format=format_name,
        file_size_mb=file_size_mb,
        prompt_tok_s=np.mean(prompt_speeds),
        gen_tok_s=np.mean(gen_speeds),
        output_sample=output_sample,
        bpw=QUANT_FORMATS.get(format_name, {}).get("bpw", 0),
    )


def compare_outputs(outputs: dict[str, str], reference_format: str = "F16") -> dict[str, float]:
    """Compare outputs across formats using simple token overlap."""
    if reference_format not in outputs:
        return {}

    ref_tokens = set(outputs[reference_format].split())
    similarities = {}

    for fmt, output in outputs.items():
        if fmt == reference_format:
            similarities[fmt] = 1.0
        else:
            fmt_tokens = set(output.split())
            if ref_tokens:
                overlap = len(ref_tokens & fmt_tokens) / len(ref_tokens | fmt_tokens)
                similarities[fmt] = overlap
            else:
                similarities[fmt] = 0.0

    return similarities


def main():
    parser = argparse.ArgumentParser(description="Benchmark GGUF quantization formats")
    parser.add_argument("--f16-gguf", type=Path, required=True, help="Path to F16 GGUF file")
    parser.add_argument("--llama-cpp", type=Path, default=Path.home() / "llama.cpp", help="llama.cpp directory")
    parser.add_argument("--output-dir", type=Path, default=Path("./benchmark_results"), help="Output directory")
    parser.add_argument("--formats", nargs="+", default=list(QUANT_FORMATS.keys()), help="Formats to benchmark")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of benchmark runs per format")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    outputs = {}

    # Benchmark F16 as baseline
    print("Benchmarking F16 (baseline)...")
    f16_result = benchmark_format(args.f16_gguf, args.llama_cpp, n_runs=args.n_runs)
    if f16_result:
        f16_result.format = "F16"
        f16_result.bpw = 16.0
        results.append(f16_result)
        outputs["F16"] = f16_result.output_sample

    # Quantize and benchmark each format
    for fmt in args.formats:
        print(f"\nQuantizing to {fmt}...")
        output_gguf = args.output_dir / f"model-{fmt.lower()}.gguf"

        if not output_gguf.exists():
            success = run_llama_quantize(args.f16_gguf, output_gguf, fmt, args.llama_cpp)
            if not success:
                print(f"  Failed to quantize to {fmt}")
                continue

        print(f"Benchmarking {fmt}...")
        result = benchmark_format(output_gguf, args.llama_cpp, n_runs=args.n_runs)
        if result:
            result.format = fmt
            result.bpw = QUANT_FORMATS[fmt]["bpw"]
            results.append(result)
            outputs[fmt] = result.output_sample

    # Compare outputs
    similarities = compare_outputs(outputs)

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Format':<10} {'Size (MB)':<12} {'BPW':<6} {'Prompt (tok/s)':<15} {'Gen (tok/s)':<12} {'Similarity':<10}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x.gen_tok_s, reverse=True):
        sim = similarities.get(r.format, 0)
        print(f"{r.format:<10} {r.file_size_mb:<12.1f} {r.bpw:<6.2f} {r.prompt_tok_s:<15.1f} {r.gen_tok_s:<12.1f} {sim:<10.2%}")

    # Save results to JSON
    results_json = args.output_dir / "benchmark_results.json"
    with open(results_json, "w") as f:
        json.dump(
            {
                "results": [
                    {
                        "format": r.format,
                        "file_size_mb": r.file_size_mb,
                        "bpw": r.bpw,
                        "prompt_tok_s": r.prompt_tok_s,
                        "gen_tok_s": r.gen_tok_s,
                        "similarity": similarities.get(r.format, 0),
                    }
                    for r in results
                ],
                "outputs": outputs,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_json}")


if __name__ == "__main__":
    main()
