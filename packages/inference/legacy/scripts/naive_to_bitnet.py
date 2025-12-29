#!/usr/bin/env python3
"""
Naive ternary conversion script for BitNet benchmarking.

Converts FP16/BF16 models to naive ternary (-1, 0, 1) format.
WARNING: This produces LOW QUALITY outputs - only for cost/speed benchmarking.

Usage:
    # Convert a model
    python scripts/naive_to_bitnet.py --model-id meta-llama/Llama-3.1-8B --output-dir models/naive

    # With GPU acceleration
    python scripts/naive_to_bitnet.py --model-id meta-llama/Llama-3.1-70B --output-dir models/naive --use-gpu

    # Estimate memory requirements first
    python scripts/naive_to_bitnet.py --model-id meta-llama/Llama-3.1-70B --estimate-only
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.naive_converter import NaiveConverter, ConversionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert models to naive ternary format for BitNet benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WARNING: Naive ternary conversion produces very low quality outputs.
This is only suitable for benchmarking inference speed and cost, NOT for production use.

The conversion process:
1. Loads the model weights in BF16
2. For each tensor: scale = mean(|weights|)
3. Quantizes: round(weights / scale), clamp to [-1, 0, 1]
4. Saves in safetensors format

For actual 1.58-bit inference, use models trained natively as BitNet
(e.g., microsoft/BitNet-b1.58-2B-4T).
""",
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/naive"),
        help="Output directory for converted model (default: models/naive)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for faster conversion (recommended for large models)",
    )
    parser.add_argument(
        "--architecture",
        choices=["llama", "moe", "auto"],
        default="auto",
        help="Model architecture (default: auto-detect)",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate memory requirements, don't convert",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def progress_callback(message: str, progress: float):
    """Display progress bar."""
    bar_length = 40
    filled = int(bar_length * progress / 100)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"\r[{bar}] {progress:.1f}% - {message}", end="", flush=True)
    if progress >= 100:
        print()  # Newline at end


def main():
    args = parse_args()

    print("=" * 60)
    print("NAIVE TERNARY CONVERSION FOR BITNET BENCHMARKING")
    print("=" * 60)
    print()
    print("WARNING: This produces LOW QUALITY outputs!")
    print("Only use for speed/cost benchmarking, NOT production.")
    print()

    config = ConversionConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        architecture=args.architecture if args.architecture != "auto" else "llama",
        use_gpu=args.use_gpu,
        verbose=args.verbose,
    )

    converter = NaiveConverter(config)

    # Estimate memory first
    print(f"Model: {args.model_id}")
    print()

    estimates = converter.estimate_memory_requirements()
    if "error" in estimates:
        logger.warning(f"Could not estimate memory: {estimates['error']}")
    else:
        print("Memory Requirements:")
        print(f"  Model (BF16): {estimates['model_bf16_gb']:.1f} GB")
        print(f"  Working Memory: {estimates['working_memory_gb']:.1f} GB")
        print(f"  Total Recommended: {estimates['total_recommended_gb']:.1f} GB")
        print(f"  Estimated Parameters: {estimates['estimated_params']:,}")
        print()

    if args.estimate_only:
        print("Estimate only mode - exiting without conversion.")
        return 0

    # Set up progress callback
    if not args.verbose:
        converter.set_progress_callback(progress_callback)

    # Run conversion
    print("Starting conversion...")
    print()

    result = converter.convert()

    if result.success:
        print()
        print("=" * 60)
        print("CONVERSION COMPLETE")
        print("=" * 60)
        print(f"Output: {result.output_path}")
        print(f"Original Size: {result.original_size_gb:.2f} GB")
        print(f"Converted Size: {result.converted_size_gb:.2f} GB (theoretical)")
        print(f"Compression Ratio: {result.compression_ratio:.1f}x")
        print(f"Layers Processed: {result.num_layers}")
        print()
        print("Next steps:")
        print("1. Convert to GGUF format using BitNet's convert utilities")
        print("2. Run benchmark: python scripts/benchmark_cost.py --url <server> --hardware <hw>")
        return 0
    else:
        print()
        logger.error(f"Conversion failed: {result.error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
