#!/usr/bin/env python3
"""Script to convert HuggingFace models to GGUF format."""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wrinklefree_inference.converter.hf_to_gguf import ConversionConfig, HFToGGUFConverter


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to GGUF")
    parser.add_argument(
        "--hf-repo",
        default="microsoft/BitNet-b1.58-2B-4T",
        help="HuggingFace repository ID (default: microsoft/BitNet-b1.58-2B-4T)",
    )
    parser.add_argument(
        "--quant-type", "-q",
        choices=["i2_s", "tl1", "tl2"],
        default="i2_s",
        help="Quantization type (default: i2_s for CPU)",
    )
    parser.add_argument(
        "--quant-embd",
        action="store_true",
        help="Quantize embeddings to f16",
    )
    parser.add_argument(
        "--bitnet-path",
        type=Path,
        default=None,
        help="Path to BitNet installation",
    )

    args = parser.parse_args()

    print(f"Converting {args.hf_repo}")
    print(f"  Quantization type: {args.quant_type}")

    try:
        converter = HFToGGUFConverter(args.bitnet_path)
        config = ConversionConfig(
            hf_repo=args.hf_repo,
            quant_type=args.quant_type,
            quant_embd=args.quant_embd,
        )

        def progress(msg: str):
            print(f"  {msg}")

        gguf_path = converter.convert(config, progress_callback=progress)
        print(f"\nSuccess! Model saved to: {gguf_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
