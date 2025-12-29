#!/usr/bin/env python3
"""Debug the repetition bug in sglang-bitnet.

Tests:
1. Native SIMD kernels vs Python fallback
2. Different sequence lengths to find where repetition starts
3. Position embedding verification
"""

import os
import sys
import requests
import json
import argparse
from typing import Optional

SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:30000")


def generate(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
) -> tuple[str, dict]:
    """Generate response from sglang server."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "stream": False,
    }

    resp = requests.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    data = resp.json()

    if "error" in data:
        return f"ERROR: {data['error']}", {}

    choices = data.get("choices", [])
    content = choices[0]["message"]["content"] if choices else ""
    usage = data.get("usage", {})

    return content, usage


def detect_repetition(text: str, min_pattern_len: int = 3, min_repeats: int = 3) -> Optional[tuple[str, int]]:
    """Detect repeating patterns in text.

    Returns:
        Tuple of (repeating_pattern, position_where_starts) or None
    """
    words = text.split()

    # Look for repeating word sequences
    for pattern_len in range(min_pattern_len, len(words) // 2):
        for start in range(len(words) - pattern_len * min_repeats):
            pattern = words[start:start + pattern_len]
            pattern_str = " ".join(pattern)

            # Count how many times this pattern repeats consecutively
            repeats = 1
            pos = start + pattern_len
            while pos + pattern_len <= len(words):
                if words[pos:pos + pattern_len] == pattern:
                    repeats += 1
                    pos += pattern_len
                else:
                    break

            if repeats >= min_repeats:
                return pattern_str, start

    return None


def test_different_lengths():
    """Test generation at different max_tokens to find where repetition starts."""
    print("\n=== Testing Different Sequence Lengths ===\n")

    prompt = "Write a detailed story about a brave knight"

    for max_tokens in [20, 40, 60, 80, 100, 150, 200]:
        output, usage = generate(prompt, max_tokens=max_tokens, temperature=0.0)

        repetition = detect_repetition(output)
        status = "REPEATS" if repetition else "OK"

        print(f"max_tokens={max_tokens:3d}: {status}")
        if repetition:
            pattern, pos = repetition
            print(f"  Pattern: '{pattern[:50]}...' at word {pos}")
        print(f"  Output: {output[:100]}...")
        print()


def test_multiple_prompts():
    """Test different prompts to see if repetition is consistent."""
    print("\n=== Testing Multiple Prompts ===\n")

    prompts = [
        "Hello, how are you today?",
        "Write a short poem about the ocean",
        "Explain quantum computing in simple terms",
        "Give me a long poem about Genghis Khan",
        "Write a 200-word story about a cat",
    ]

    for prompt in prompts:
        output, usage = generate(prompt, max_tokens=150, temperature=0.0)
        repetition = detect_repetition(output)

        status = "REPEATS" if repetition else "OK"
        print(f"Prompt: '{prompt[:40]}...'")
        print(f"Status: {status}")
        if repetition:
            pattern, pos = repetition
            print(f"Pattern: '{pattern[:40]}...' at word {pos}")
        print(f"Output: {output[:80]}...")
        print()


def test_with_penalty():
    """Test if repetition penalty helps."""
    print("\n=== Testing Repetition Penalty ===\n")

    prompt = "Write a detailed story about a brave knight"

    for penalty in [1.0, 1.1, 1.2, 1.5, 2.0]:
        output, usage = generate(
            prompt,
            max_tokens=150,
            temperature=0.0,
            repetition_penalty=penalty,
        )

        repetition = detect_repetition(output)
        status = "REPEATS" if repetition else "OK"

        print(f"penalty={penalty:.1f}: {status}")
        if repetition:
            pattern, pos = repetition
            print(f"  Pattern at word {pos}")
        print(f"  Output: {output[:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Debug repetition bug")
    parser.add_argument("--test", choices=["lengths", "prompts", "penalty", "all"],
                       default="all", help="Which test to run")
    args = parser.parse_args()

    # Check server is running
    try:
        resp = requests.get(f"{SGLANG_URL}/v1/models", timeout=5)
        if resp.status_code != 200:
            print(f"Error: Server not responding properly at {SGLANG_URL}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to {SGLANG_URL}")
        sys.exit(1)

    print(f"Connected to SGLang server at {SGLANG_URL}")

    if args.test in ["lengths", "all"]:
        test_different_lengths()

    if args.test in ["prompts", "all"]:
        test_multiple_prompts()

    if args.test in ["penalty", "all"]:
        test_with_penalty()

    print("\n=== Summary ===")
    print("If repetition happens with temp=0 and penalty=2.0, it's a model/kernel bug.")
    print("If repetition only happens without penalty, it's a sampling issue.")


if __name__ == "__main__":
    main()
