#!/usr/bin/env python3
"""Correctness tests for sglang-bitnet.

Tests:
1. No repetition in output (the bug we fixed)
2. Deterministic output with temp=0
3. Coherent output for various prompts
4. Output length matches requested max_tokens

Run with server active:
    SGLANG_URL=http://127.0.0.1:30000 pytest tests/test_sglang_correctness.py -v
"""

import os
import re
import pytest
import requests
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
        raise RuntimeError(f"API error: {data['error']}")

    choices = data.get("choices", [])
    content = choices[0]["message"]["content"] if choices else ""
    usage = data.get("usage", {})

    return content, usage


def detect_repetition(
    text: str, min_pattern_len: int = 3, min_repeats: int = 3
) -> Optional[tuple[str, int]]:
    """Detect repeating word patterns in text.

    Returns:
        Tuple of (repeating_pattern, position_where_starts) or None
    """
    words = text.split()
    if len(words) < min_pattern_len * min_repeats:
        return None

    for pattern_len in range(min_pattern_len, min(20, len(words) // 2)):
        for start in range(len(words) - pattern_len * min_repeats):
            pattern = words[start : start + pattern_len]
            pattern_str = " ".join(pattern)

            repeats = 1
            pos = start + pattern_len
            while pos + pattern_len <= len(words):
                if words[pos : pos + pattern_len] == pattern:
                    repeats += 1
                    pos += pattern_len
                else:
                    break

            if repeats >= min_repeats:
                return pattern_str, start

    return None


def detect_char_repetition(text: str, min_len: int = 10, min_repeats: int = 3) -> bool:
    """Detect character-level repetition (e.g., 'aaaa' or 'abcabc')."""
    for pattern_len in range(min_len, min(50, len(text) // min_repeats)):
        for start in range(len(text) - pattern_len * min_repeats):
            pattern = text[start : start + pattern_len]
            repeats = 1
            pos = start + pattern_len
            while pos + pattern_len <= len(text):
                if text[pos : pos + pattern_len] == pattern:
                    repeats += 1
                    pos += pattern_len
                else:
                    break
            if repeats >= min_repeats:
                return True
    return False


@pytest.fixture(scope="module")
def server_available():
    """Check if server is available."""
    try:
        resp = requests.get(f"{SGLANG_URL}/v1/models", timeout=5)
        if resp.status_code != 200:
            pytest.skip(f"Server not responding at {SGLANG_URL}")
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Cannot connect to {SGLANG_URL}")


class TestNoRepetition:
    """Test that output doesn't degenerate into repetition loops."""

    @pytest.mark.parametrize("max_tokens", [50, 100, 150, 200])
    def test_no_repetition_various_lengths(self, server_available, max_tokens):
        """Test no repetition at various output lengths."""
        prompt = "Write a detailed story about a brave knight"
        output, _ = generate(prompt, max_tokens=max_tokens, temperature=0.0)

        repetition = detect_repetition(output)
        assert repetition is None, (
            f"Repetition detected at max_tokens={max_tokens}: "
            f"pattern='{repetition[0][:50]}...' at word {repetition[1]}"
        )

    @pytest.mark.parametrize("prompt", [
        "Hello, how are you today?",
        "Write a short poem about the ocean",
        "Explain quantum computing in simple terms",
        "Give me a long poem about Genghis Khan",
        "Write a 200-word story about a cat",
        "List the first 10 prime numbers and explain why they are prime",
        "Describe the process of photosynthesis",
    ])
    def test_no_repetition_various_prompts(self, server_available, prompt):
        """Test no repetition with various prompts."""
        output, _ = generate(prompt, max_tokens=150, temperature=0.0)

        word_rep = detect_repetition(output)
        assert word_rep is None, (
            f"Word repetition for '{prompt[:30]}...': "
            f"pattern='{word_rep[0][:40]}...' at word {word_rep[1]}"
        )

        char_rep = detect_char_repetition(output)
        assert not char_rep, f"Character repetition detected for '{prompt[:30]}...'"


class TestDeterminism:
    """Test deterministic output with temperature=0."""

    def test_deterministic_output_temp0(self, server_available):
        """Same prompt with temp=0 should produce identical output."""
        prompt = "What is the capital of France?"

        output1, _ = generate(prompt, max_tokens=50, temperature=0.0)
        output2, _ = generate(prompt, max_tokens=50, temperature=0.0)
        output3, _ = generate(prompt, max_tokens=50, temperature=0.0)

        assert output1 == output2, "Output 1 != Output 2 with temp=0"
        assert output2 == output3, "Output 2 != Output 3 with temp=0"

    def test_deterministic_longer_output(self, server_available):
        """Determinism should hold for longer outputs."""
        prompt = "Write a story about a robot"

        output1, _ = generate(prompt, max_tokens=100, temperature=0.0)
        output2, _ = generate(prompt, max_tokens=100, temperature=0.0)

        assert output1 == output2, "Long outputs not deterministic with temp=0"


class TestCoherence:
    """Test that outputs are coherent and sensible."""

    def test_output_not_empty(self, server_available):
        """Output should not be empty."""
        output, _ = generate("Hello", max_tokens=20, temperature=0.0)
        assert len(output.strip()) > 0, "Output is empty"

    def test_output_contains_words(self, server_available):
        """Output should contain actual words."""
        output, _ = generate("Tell me about cats", max_tokens=50, temperature=0.0)
        words = output.split()
        assert len(words) >= 5, f"Output too short: {len(words)} words"

    def test_math_question(self, server_available):
        """Simple math should be correct."""
        output, _ = generate("What is 2 + 2?", max_tokens=20, temperature=0.0)
        assert "4" in output, f"Math answer doesn't contain '4': {output}"

    def test_factual_question(self, server_available):
        """Basic factual questions should be correct."""
        output, _ = generate(
            "What is the capital of France? Answer in one word.",
            max_tokens=20,
            temperature=0.0,
        )
        assert "paris" in output.lower(), f"Answer doesn't contain 'Paris': {output}"


class TestTokenCount:
    """Test that output respects token limits."""

    def test_respects_max_tokens(self, server_available):
        """Output should not significantly exceed max_tokens."""
        for max_tokens in [20, 50, 100]:
            output, usage = generate(
                "Write as much as you can",
                max_tokens=max_tokens,
                temperature=0.0,
            )
            completion_tokens = usage.get("completion_tokens", len(output.split()))
            # Allow some tolerance for tokenization differences
            assert completion_tokens <= max_tokens + 5, (
                f"Output exceeded max_tokens={max_tokens}: got {completion_tokens}"
            )


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_very_short_prompt(self, server_available):
        """Single character prompt should work."""
        output, _ = generate("Hi", max_tokens=20, temperature=0.0)
        assert len(output) > 0

    def test_long_prompt(self, server_available):
        """Long prompt should work."""
        long_prompt = "Please help me. " * 50
        output, _ = generate(long_prompt, max_tokens=50, temperature=0.0)
        assert len(output) > 0

    def test_special_characters(self, server_available):
        """Prompts with special characters should work."""
        output, _ = generate(
            "What does the symbol @ mean in email addresses?",
            max_tokens=50,
            temperature=0.0,
        )
        assert len(output) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
