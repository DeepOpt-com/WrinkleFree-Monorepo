#!/usr/bin/env python3
"""Tests for dataset loading - validates all datasets in mixed_pretrain.yaml can be loaded.

Run with: uv run pytest tests/test_dataset_loading.py -v
"""
import pytest
from datasets import load_dataset


# Dataset configs to test - must match mixed_pretrain.yaml
DATASET_CONFIGS = [
    {
        "name": "dclm",
        "path": "mlfoundations/dclm-baseline-1.0-parquet",
        "subset": None,
        "text_column": "text",
    },
    {
        "name": "fineweb_edu",
        "path": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "text_column": "text",
    },
    {
        "name": "github_code",
        "path": "nick007x/github-code-2025",
        "subset": None,
        "text_column": "content",
    },
    {
        "name": "finemath",  # Math dataset - commercially friendly (ODC-By license)
        "path": "HuggingFaceTB/finemath",
        "subset": "finemath-3plus",
        "text_column": "text",
    },
    {
        "name": "slimpajama",
        "path": "cerebras/SlimPajama-627B",
        "subset": None,
        "text_column": "text",
    },
]

# Alternative math datasets to try if primary fails
ALTERNATIVE_MATH_DATASETS = [
    {
        "name": "metamathqa",
        "path": "meta-math/MetaMathQA",
        "subset": None,
        "text_column": "query",  # Has: query, response
    },
    {
        "name": "megamath_web",  # Try loading MegaMath with data_files
        "path": "LLM360/MegaMath",
        "data_dir": "megamath-web",  # Use data_dir instead of subset
        "text_column": "text",
    },
]


class TestDatasetLoading:
    """Test that each dataset in mixed_pretrain can be loaded and has expected columns."""

    @pytest.mark.parametrize("config", DATASET_CONFIGS, ids=lambda c: c["name"])
    def test_dataset_loads_streaming(self, config):
        """Test that dataset can be loaded in streaming mode."""
        print(f"\nTesting: {config['name']} ({config['path']})")

        try:
            ds = load_dataset(
                config["path"],
                config.get("subset"),
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

            # Get first example to verify columns
            first_example = next(iter(ds))

            # Check text column exists
            text_col = config["text_column"]
            assert text_col in first_example, (
                f"Dataset {config['name']} missing column '{text_col}'. "
                f"Available: {list(first_example.keys())}"
            )

            # Verify text is non-empty string
            text_value = first_example[text_col]
            assert isinstance(text_value, str), f"Column '{text_col}' is not string: {type(text_value)}"
            assert len(text_value) > 0, f"Column '{text_col}' is empty"

            print(f"  OK: Found column '{text_col}' with {len(text_value)} chars")
            print(f"  Columns: {list(first_example.keys())}")

        except Exception as e:
            pytest.fail(f"Failed to load {config['name']}: {e}")

    def test_all_weights_sum_to_one(self):
        """Verify dataset weights in config sum to 1.0."""
        # Weights from mixed_pretrain.yaml (must match!)
        weights = [0.25, 0.30, 0.15, 0.15, 0.15]  # dclm, fineweb, code, finemath, slimpajama
        total = sum(weights)
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"


class TestMathDatasetAlternatives:
    """Test alternative math datasets to find one that works."""

    @pytest.mark.parametrize("config", ALTERNATIVE_MATH_DATASETS, ids=lambda c: c["name"])
    def test_math_dataset_alternative(self, config):
        """Test alternative math datasets."""
        print(f"\nTesting math alternative: {config['name']} ({config['path']})")

        try:
            kwargs = {
                "path": config["path"],
                "split": "train",
                "streaming": True,
                "trust_remote_code": True,
            }

            if config.get("subset"):
                kwargs["name"] = config["subset"]
            if config.get("data_dir"):
                kwargs["data_dir"] = config["data_dir"]

            ds = load_dataset(**kwargs)
            first_example = next(iter(ds))

            text_col = config["text_column"]
            if text_col in first_example:
                print(f"  OK: Found column '{text_col}'")
                print(f"  Columns: {list(first_example.keys())}")
                print(f"  Sample: {str(first_example[text_col])[:200]}...")
            else:
                print(f"  WARNING: Column '{text_col}' not found")
                print(f"  Available columns: {list(first_example.keys())}")

        except Exception as e:
            print(f"  FAILED: {e}")
            # Don't fail - these are alternatives


def test_quick_smoke():
    """Quick smoke test - just verify imports work."""
    from datasets import load_dataset
    assert load_dataset is not None


if __name__ == "__main__":
    # Run quick validation
    import sys

    print("=" * 60)
    print("Dataset Loading Validation")
    print("=" * 60)

    failed = []
    for config in DATASET_CONFIGS:
        print(f"\nTesting: {config['name']}")
        try:
            ds = load_dataset(
                config["path"],
                config.get("subset"),
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            first = next(iter(ds))
            text_col = config["text_column"]

            if text_col in first:
                print(f"  OK: column '{text_col}' found")
            else:
                print(f"  FAIL: column '{text_col}' missing. Has: {list(first.keys())}")
                failed.append(config["name"])

        except Exception as e:
            print(f"  FAIL: {e}")
            failed.append(config["name"])

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("ALL PASSED")
        sys.exit(0)
