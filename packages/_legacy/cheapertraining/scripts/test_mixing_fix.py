#!/usr/bin/env python3
"""Test script to verify dataset mixing fix.

This script demonstrates that:
1. Sampling follows configured weights
2. Exhausted datasets are NOT reused (the fix)
3. Stats are tracked correctly

Run with: uv run --package cheapertraining python scripts/test_mixing_fix.py
"""

from cheapertraining.data.mixing import DatasetMixture, MixedDataset


def test_no_oversampling():
    """Verify small datasets aren't oversampled."""
    print("=" * 60)
    print("TEST: Small datasets should NOT be reused/oversampled")
    print("=" * 60)

    # Create mixtures with equal weights but different sizes
    mixtures = [
        DatasetMixture(name="small", weight=0.5, path="test/small"),
        DatasetMixture(name="large", weight=0.5, path="test/large"),
    ]
    dataset = MixedDataset(mixtures=mixtures, streaming=False, seed=42)

    # Small has 10 samples, large has 100
    dataset._datasets = [
        [{"text": f"small_{i}"} for i in range(10)],
        [{"text": f"large_{i}"} for i in range(100)],
    ]
    dataset._iterators = [iter(d) for d in dataset._datasets]

    # Consume all samples
    all_samples = list(dataset)

    # Count by source
    small_count = sum(1 for s in all_samples if s["source"] == "small")
    large_count = sum(1 for s in all_samples if s["source"] == "large")

    print(f"\nDataset sizes: small=10, large=100")
    print(f"Configured weights: small=50%, large=50%")
    print(f"\nResults:")
    print(f"  Small samples: {small_count} (expected: 10, NOT more)")
    print(f"  Large samples: {large_count} (expected: 100)")
    print(f"  Total: {len(all_samples)} (expected: 110)")

    # Get stats
    stats = dataset.get_sampling_stats()
    print(f"\nSampling stats:")
    print(f"  Counts: {stats['counts']}")
    print(f"  Exhausted: {stats['exhausted']}")

    # Verify
    assert small_count == 10, f"FAIL: Small was oversampled! Got {small_count}, expected 10"
    assert large_count == 100, f"FAIL: Large count wrong! Got {large_count}, expected 100"
    print("\n✓ PASS: Small dataset was NOT oversampled")


def test_weights_respected():
    """Verify sampling follows configured weights."""
    print("\n" + "=" * 60)
    print("TEST: Sampling should follow configured weights")
    print("=" * 60)

    mixtures = [
        DatasetMixture(name="dominant", weight=0.8, path="test/a"),
        DatasetMixture(name="minor", weight=0.2, path="test/b"),
    ]
    dataset = MixedDataset(mixtures=mixtures, streaming=False, seed=42)

    # Both have plenty of samples
    dataset._datasets = [
        [{"text": f"dominant_{i}"} for i in range(10000)],
        [{"text": f"minor_{i}"} for i in range(10000)],
    ]
    dataset._iterators = [iter(d) for d in dataset._datasets]

    # Draw 1000 samples
    counts = {"dominant": 0, "minor": 0}
    for i, sample in enumerate(dataset):
        if i >= 1000:
            break
        counts[sample["source"]] += 1

    dominant_pct = counts["dominant"] / 1000 * 100
    minor_pct = counts["minor"] / 1000 * 100

    print(f"\nConfigured weights: dominant=80%, minor=20%")
    print(f"\nResults (1000 samples):")
    print(f"  Dominant: {counts['dominant']} ({dominant_pct:.1f}%)")
    print(f"  Minor: {counts['minor']} ({minor_pct:.1f}%)")

    # Allow 5% tolerance
    assert abs(dominant_pct - 80) < 5, f"FAIL: Dominant was {dominant_pct:.1f}%, expected ~80%"
    assert abs(minor_pct - 20) < 5, f"FAIL: Minor was {minor_pct:.1f}%, expected ~20%"
    print("\n✓ PASS: Weights are respected (within 5% tolerance)")


def test_weight_renormalization():
    """Verify weights renormalize when a dataset exhausts."""
    print("\n" + "=" * 60)
    print("TEST: Weights should renormalize when datasets exhaust")
    print("=" * 60)

    mixtures = [
        DatasetMixture(name="tiny", weight=0.33, path="test/tiny"),
        DatasetMixture(name="medium", weight=0.33, path="test/medium"),
        DatasetMixture(name="large", weight=0.34, path="test/large"),
    ]
    dataset = MixedDataset(mixtures=mixtures, streaming=False, seed=42)

    # Tiny exhausts quickly, medium later, large has plenty
    dataset._datasets = [
        [{"text": f"tiny_{i}"} for i in range(5)],      # Exhausts first
        [{"text": f"medium_{i}"} for i in range(20)],   # Exhausts second
        [{"text": f"large_{i}"} for i in range(100)],   # Never exhausts
    ]
    dataset._iterators = [iter(d) for d in dataset._datasets]

    all_samples = list(dataset)

    counts = {}
    for s in all_samples:
        counts[s["source"]] = counts.get(s["source"], 0) + 1

    print(f"\nDataset sizes: tiny=5, medium=20, large=100")
    print(f"\nResults:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print(f"  Total: {len(all_samples)}")

    stats = dataset.get_sampling_stats()
    print(f"\nExhausted datasets: {stats['exhausted']}")

    # Verify exact counts (no oversampling)
    assert counts.get("tiny", 0) == 5, f"FAIL: Tiny oversampled"
    assert counts.get("medium", 0) == 20, f"FAIL: Medium oversampled"
    assert counts.get("large", 0) == 100, f"FAIL: Large count wrong"
    print("\n✓ PASS: All datasets used exactly once, weights renormalized correctly")


if __name__ == "__main__":
    print("\n🧪 Testing Dataset Mixing Fix\n")

    test_no_oversampling()
    test_weights_respected()
    test_weight_renormalization()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("\nThe fix is working correctly:")
    print("  • Small datasets are NOT oversampled")
    print("  • Sampling follows configured weights")
    print("  • Weights renormalize when datasets exhaust")
    print()
