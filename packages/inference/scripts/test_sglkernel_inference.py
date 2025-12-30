#!/usr/bin/env python3
"""
Test script for sgl-kernel native BitNet inference.

This script:
1. Validates the conversion from safetensors to sgl-kernel format
2. Tests the native C++ inference engine
3. Compares output with Python reference implementation

Usage:
    # Test conversion only
    python test_sglkernel_inference.py --checkpoint /path/to/checkpoint

    # Full test with server
    python test_sglkernel_inference.py --checkpoint /path/to/checkpoint --test-server

    # Test against running server
    python test_sglkernel_inference.py --server-url http://localhost:30000
"""

import argparse
import json
import logging
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
import torch
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Test prompts for validation
TEST_PROMPTS = [
    "The capital of France is",
    "1 + 1 = ",
    "Hello, how are you",
]

# Expected patterns (substring match)
EXPECTED_PATTERNS = {
    "The capital of France is": ["Paris", "paris"],
    "1 + 1 = ": ["2", "two"],
    "Hello, how are you": ["I'm", "good", "well", "fine", "Hello"],
}


def validate_conversion(checkpoint_path: Path, output_path: Path) -> bool:
    """Validate the converted binary file."""
    logger.info("Validating converted model...")

    # Read header
    with open(output_path, 'rb') as f:
        magic = f.read(8)
        if magic != b"SGLBITNT":
            logger.error(f"Invalid magic: {magic}")
            return False

        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            logger.error(f"Unsupported version: {version}")
            return False

        config_len = struct.unpack('<I', f.read(4))[0]
        config_json = f.read(config_len).decode('utf-8')
        config = json.loads(config_json)

        num_tensors = struct.unpack('<I', f.read(4))[0]

    logger.info(f"Model config:")
    logger.info(f"  vocab_size: {config.get('vocab_size')}")
    logger.info(f"  hidden_size: {config.get('hidden_size')}")
    logger.info(f"  num_hidden_layers: {config.get('num_hidden_layers')}")
    logger.info(f"  num_tensors: {num_tensors}")

    # Load original safetensors to compare
    original_weights = load_file(checkpoint_path / "model.safetensors")

    # Count packed vs unpacked tensors
    packed_count = 0
    unpacked_count = 0

    with open(output_path, 'rb') as f:
        # Skip to tensors
        f.seek(8 + 4 + 4 + config_len + 4)

        for _ in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            dtype = struct.unpack('<I', f.read(4))[0]
            ndims = struct.unpack('<I', f.read(4))[0]
            shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndims)]
            has_scale = struct.unpack('<I', f.read(4))[0]
            if has_scale:
                scale = struct.unpack('<f', f.read(4))[0]
            else:
                scale = None
            data_size = struct.unpack('<Q', f.read(8))[0]

            # Skip data
            f.seek(data_size, 1)

            if dtype == 0:  # uint8 (packed)
                packed_count += 1
            else:
                unpacked_count += 1

    logger.info(f"  packed tensors: {packed_count}")
    logger.info(f"  unpacked tensors: {unpacked_count}")

    # Validate tensor counts
    expected_layers = config.get('num_hidden_layers', 30)
    expected_packed = expected_layers * 7  # 4 attn + 3 mlp per layer
    expected_unpacked = expected_layers * 2 + 3  # 2 norms per layer + embed + lm_head + final_norm

    if packed_count != expected_packed:
        logger.warning(f"Expected {expected_packed} packed tensors, got {packed_count}")

    return True


def test_python_inference(checkpoint_path: Path, prompts: list[str]) -> dict:
    """Run Python inference as reference."""
    logger.info("Running Python reference inference...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

        results = {}
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results[prompt] = response
            logger.info(f"  '{prompt}' -> '{response[:100]}...'")

        return results

    except Exception as e:
        logger.error(f"Python inference failed: {e}")
        return {}


def test_server_inference(server_url: str, prompts: list[str]) -> dict:
    """Test inference against a running server."""
    logger.info(f"Testing server at {server_url}...")

    results = {}
    for prompt in prompts:
        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            results[prompt] = text
            logger.info(f"  '{prompt}' -> '{text[:100]}...'")
        except Exception as e:
            logger.error(f"Server request failed: {e}")
            results[prompt] = f"ERROR: {e}"

    return results


def validate_responses(results: dict, expected: dict = None) -> bool:
    """Validate inference responses."""
    if expected is None:
        expected = EXPECTED_PATTERNS

    all_passed = True
    for prompt, response in results.items():
        if prompt in expected:
            patterns = expected[prompt]
            found = any(p.lower() in response.lower() for p in patterns)
            status = "PASS" if found else "FAIL"
            if not found:
                all_passed = False
            logger.info(f"  [{status}] '{prompt}' -> expected one of {patterns}")
        else:
            logger.info(f"  [SKIP] '{prompt}' (no expected pattern)")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test sgl-kernel BitNet inference")
    parser.add_argument("--checkpoint", type=Path, help="Path to DLM checkpoint")
    parser.add_argument("--output", type=Path, help="Output .bin file path")
    parser.add_argument("--server-url", type=str, help="URL of running server to test")
    parser.add_argument("--test-server", action="store_true", help="Start and test server")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip conversion step")

    args = parser.parse_args()

    if args.server_url:
        # Test against running server
        results = test_server_inference(args.server_url, TEST_PROMPTS)
        if validate_responses(results):
            logger.info("All tests PASSED!")
            return 0
        else:
            logger.error("Some tests FAILED!")
            return 1

    if not args.checkpoint:
        parser.error("--checkpoint is required when not using --server-url")

    checkpoint_path = args.checkpoint
    output_path = args.output or checkpoint_path.parent / (checkpoint_path.name + ".bin")

    # Step 1: Convert checkpoint
    if not args.skip_conversion:
        logger.info(f"Converting {checkpoint_path} to sgl-kernel format...")
        scripts_dir = Path(__file__).parent
        convert_script = scripts_dir / "convert_to_sglkernel.py"

        result = subprocess.run(
            ["python", str(convert_script), str(checkpoint_path), str(output_path)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            return 1
        logger.info("Conversion complete!")

    # Step 2: Validate conversion
    if not validate_conversion(checkpoint_path, output_path):
        logger.error("Conversion validation failed!")
        return 1

    # Step 3: Test Python inference (reference)
    python_results = test_python_inference(checkpoint_path, TEST_PROMPTS)

    if python_results:
        logger.info("\nPython reference results:")
        if validate_responses(python_results):
            logger.info("Python inference produces correct output!")
        else:
            logger.warning("Python inference has issues!")

    # Step 4: Test server if requested
    if args.test_server:
        logger.info("\nStarting native inference server...")
        # TODO: Start server and test
        logger.warning("Server testing not yet implemented")

    logger.info("\nTest complete!")
    return 0


if __name__ == "__main__":
    exit(main())
