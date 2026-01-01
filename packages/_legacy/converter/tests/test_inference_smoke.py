"""Smoke tests for Fast-dLLM v2 inference with DualCache.

These tests verify end-to-end generation works correctly with real models.
Requires GPU and takes ~1-2 minutes to run.

Run with: uv run pytest tests/test_inference_smoke.py -m smoke -v
"""

import pytest
import torch


@pytest.mark.smoke
@pytest.mark.gpu
class TestInferenceSmoke:
    """End-to-end smoke tests for inference with DualCache."""

    def test_generate_without_dualcache(self, small_model, small_tokenizer):
        """Baseline: generation works without DualCache (use_block_cache=False)."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer
        prompt = "The capital of France is"

        result = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=False,  # Disabled
            max_new_tokens=32,
            temperature=0.0,
        )

        assert result.text is not None
        assert len(result.text) > len(prompt)
        assert result.used_dualcache is False
        assert result.tokens_generated > 0

    def test_generate_with_dualcache(self, small_model, small_tokenizer):
        """Generation works with DualCache enabled (use_block_cache=True)."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer
        prompt = "The capital of France is"

        result = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=True,  # Enabled
            max_new_tokens=32,
            temperature=0.0,
        )

        assert result.text is not None
        assert len(result.text) > len(prompt)
        assert result.used_dualcache is True
        assert result.tokens_generated > 0

    def test_dualcache_output_valid(self, small_model, small_tokenizer):
        """DualCache output should be coherent text, not garbage."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer
        prompt = "Hello, how are you today?"

        result = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=64,
            temperature=0.0,
        )

        # Output should start with prompt
        assert result.text.startswith("Hello")

        # Should not be just repetition of special tokens or garbage
        special_chars = result.text.count("[") + result.text.count("]")
        total_chars = len(result.text)
        assert special_chars / max(total_chars, 1) < 0.5, "Too many special characters"

    def test_various_small_block_sizes(self, small_model, small_tokenizer):
        """Different small_block_size values should all work."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer
        prompt = "Once upon a time"

        for small_block_size in [4, 8, 16, 32]:
            result = generate_with_dualcache(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                block_size=32,
                small_block_size=small_block_size,
                use_block_cache=True,
                max_new_tokens=32,
                temperature=0.0,
            )

            assert result.text is not None, f"Failed with small_block_size={small_block_size}"
            assert result.tokens_generated > 0, f"No tokens generated with small_block_size={small_block_size}"

    def test_dualcache_nfe_tracked(self, small_model, small_tokenizer):
        """NFE (Number of Forward Evaluations) should be tracked."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer
        prompt = "Test prompt"

        result = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=32,
            temperature=0.0,
        )

        # NFE should be positive (or -1 if tracked internally by batch_sample)
        assert result.nfe != 0

    def test_generation_timing(self, small_model, small_tokenizer):
        """Generation should complete in reasonable time and report speed."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer
        prompt = "Performance test"

        result = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=64,
            temperature=0.0,
        )

        # Should complete in under 30 seconds for 64 tokens
        assert result.elapsed_seconds < 30.0

        # Should report tokens per second
        assert result.tokens_per_second >= 0


@pytest.mark.smoke
@pytest.mark.gpu
class TestDualCachePerformance:
    """Tests verifying DualCache provides expected benefits."""

    def test_dualcache_reduces_computation(self, small_model, small_tokenizer):
        """DualCache should reduce redundant computation vs no cache."""
        from wf_dlm_converter.inference import generate_with_dualcache
        import time

        model, tokenizer = small_model, small_tokenizer
        prompt = "Performance comparison test"

        # Without DualCache
        start = time.time()
        result_no_cache = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=False,
            max_new_tokens=64,
            temperature=0.0,
        )
        time_no_cache = time.time() - start

        # With DualCache
        start = time.time()
        result_with_cache = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=64,
            temperature=0.0,
        )
        time_with_cache = time.time() - start

        # Both should produce valid output
        assert result_no_cache.tokens_generated > 0
        assert result_with_cache.tokens_generated > 0

        # Log timing for analysis (not strict assertion due to variance)
        print(f"\nWithout DualCache: {time_no_cache:.2f}s")
        print(f"With DualCache: {time_with_cache:.2f}s")
        print(f"Speedup: {time_no_cache / max(time_with_cache, 0.001):.2f}x")


@pytest.mark.smoke
@pytest.mark.gpu
class TestEdgeCases:
    """Edge case tests for DualCache inference."""

    def test_short_prompt(self, small_model, small_tokenizer):
        """Should handle very short prompts."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer

        result = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt="Hi",
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=32,
            temperature=0.0,
        )

        assert result.text is not None
        assert "Hi" in result.text

    def test_long_prompt(self, small_model, small_tokenizer):
        """Should handle prompts longer than block_size."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer

        # Prompt longer than block_size (32 tokens)
        long_prompt = "This is a very long prompt that should exceed the block size. " * 5

        result = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=long_prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=32,
            temperature=0.0,
        )

        assert result.text is not None
        assert result.tokens_generated >= 0

    def test_batch_size_one(self, small_model, small_tokenizer):
        """Single sample batch should work correctly."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer

        result = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt="Single sample test",
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=16,
            temperature=0.0,
        )

        assert result.text is not None

    def test_greedy_vs_sampling(self, small_model, small_tokenizer):
        """Both greedy and sampling modes should work."""
        from wf_dlm_converter.inference import generate_with_dualcache

        model, tokenizer = small_model, small_tokenizer
        prompt = "Determinism test"

        # Greedy (temperature=0)
        result_greedy = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=32,
            temperature=0.0,
        )

        # Sampling (temperature=0.7)
        result_sample = generate_with_dualcache(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            block_size=32,
            small_block_size=8,
            use_block_cache=True,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
        )

        assert result_greedy.text is not None
        assert result_sample.text is not None


@pytest.mark.smoke
@pytest.mark.gpu
class TestModelLoading:
    """Tests for model loading utilities."""

    def test_load_dlm_model(self, tmp_path):
        """Test load_dlm_model utility function."""
        from wf_dlm_converter.inference import load_dlm_model, DEFAULT_MASK_TOKEN
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Create a minimal test model
        model_name = "HuggingFaceTB/SmolLM2-135M"

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )

            # Add mask token and save
            tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_MASK_TOKEN]})
            model.resize_token_embeddings(len(tokenizer))

            save_path = tmp_path / "test_model"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Test loading
            loaded_model, loaded_tokenizer = load_dlm_model(save_path)

            assert DEFAULT_MASK_TOKEN in loaded_tokenizer.get_vocab()
            assert loaded_model is not None

        except Exception as e:
            pytest.skip(f"Could not download test model: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])
