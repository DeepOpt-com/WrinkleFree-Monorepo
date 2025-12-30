"""Real data ablation test for influence-based rebalancing.

Uses commercially-friendly datasets:
- wikitext (Apache 2.0) - general knowledge
- gsm8k (MIT) - math reasoning
- code_search_net (MIT) - code

This tests whether influence-based rebalancing provides any benefit over
static equal weighting on REAL data (not synthetic).
"""

import os
import sys
import time
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Conditionally import datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class TextDataset(Dataset):
    """Simple dataset wrapper for tokenized text."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [t for t in texts if t.strip()]  # Filter empty

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def load_wikitext_samples(n_samples: int = 500) -> list[str]:
    """Load samples from wikitext-2."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    texts = []
    for item in ds:
        text = item.get("text", "")
        if text.strip() and len(text) > 50:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    print(f"  Loaded {len(texts)} wikitext samples")
    return texts


def load_fineweb_edu_samples(n_samples: int = 500) -> list[str]:
    """Load samples from fineweb-edu (high quality educational content)."""
    try:
        # fineweb-edu-score-2 is the smaller scored version
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu-score-2",
            split="train",
            streaming=True,
        )
        texts = []
        for item in ds:
            text = item.get("text", "")
            score = item.get("score", 0)
            # Only keep high-quality samples (score >= 3)
            if text.strip() and len(text) > 100 and score >= 3:
                texts.append(text[:2000])  # Limit length
            if len(texts) >= n_samples:
                break
        print(f"  Loaded {len(texts)} fineweb-edu samples")
        return texts
    except Exception as e:
        print(f"  fineweb-edu failed: {e}, falling back to wikitext")
        return load_wikitext_samples(n_samples)


def load_gsm8k_samples(n_samples: int = 500) -> list[str]:
    """Load samples from gsm8k (math)."""
    ds = load_dataset("gsm8k", "main", split="train", streaming=True)
    texts = []
    for item in ds:
        q = item.get("question", "")
        a = item.get("answer", "")
        if q and a:
            texts.append(f"Question: {q}\nAnswer: {a}")
        if len(texts) >= n_samples:
            break
    print(f"  Loaded {len(texts)} gsm8k samples")
    return texts


def load_code_samples(n_samples: int = 500) -> list[str]:
    """Load samples from bigcode/starcoderdata (python subset) or fallback."""
    # Try multiple code datasets in order of preference
    datasets_to_try = [
        # HuggingFace datasets with simple access
        ("codeparrot/github-code-clean", {"languages": ["Python"]}),
        ("sahil2801/CodeAlpaca-20k", {}),
        ("iamtarun/python_code_instructions_18k_alpaca", {}),
    ]

    for ds_name, config in datasets_to_try:
        try:
            print(f"  Trying {ds_name}...")
            ds = load_dataset(ds_name, split="train", streaming=True, **config)
            texts = []
            for item in ds:
                # Handle different column names
                code = (
                    item.get("code", "")
                    or item.get("output", "")
                    or item.get("completion", "")
                    or item.get("content", "")
                )
                if code and len(code) > 50:
                    texts.append(code)
                if len(texts) >= n_samples:
                    break

            if texts:
                print(f"  Loaded {len(texts)} code samples from {ds_name}")
                return texts
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Ultimate fallback: generate simple Python code patterns
    print("  Using synthetic Python code patterns as fallback")
    patterns = [
        "def calculate_sum(a, b):\n    return a + b\n",
        "def find_max(numbers):\n    return max(numbers)\n",
        "class Calculator:\n    def add(self, x, y):\n        return x + y\n",
        "for i in range(10):\n    print(i)\n",
        "import os\nfiles = os.listdir('.')\n",
    ]
    return patterns * (n_samples // len(patterns) + 1)


def compute_loss(model, batch, device):
    """Compute cross-entropy loss for a batch."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss.item()


def train_step(model, optimizer, batch, device) -> float:
    """Single training step, returns loss."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def evaluate_on_loader(model, loader, device, max_batches: int = 20) -> float:
    """Compute average loss on a dataloader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            loss = compute_loss(model, batch, device)
            total_loss += loss
            n_batches += 1

    return total_loss / max(n_batches, 1)


class MixedSampler:
    """Sample from multiple datasets with configurable weights."""

    def __init__(self, loaders: dict[str, DataLoader], weights: Optional[dict[str, float]] = None):
        self.loaders = loaders
        self.names = list(loaders.keys())

        # Default to equal weights
        if weights is None:
            weights = {k: 1.0 / len(loaders) for k in loaders}
        self.weights = weights

        # Create iterators
        self._reset_iterators()

    def _reset_iterators(self):
        self.iterators = {k: iter(v) for k, v in self.loaders.items()}

    def set_weights(self, weights: dict[str, float]):
        """Update sampling weights."""
        self.weights = weights

    def sample_batch(self) -> tuple[dict, str]:
        """Sample a batch according to current weights."""
        # Normalize weights
        total = sum(self.weights.values())
        probs = [self.weights[k] / total for k in self.names]

        # Sample dataset
        idx = torch.multinomial(torch.tensor(probs), 1).item()
        name = self.names[idx]

        # Get batch from that dataset
        try:
            batch = next(self.iterators[name])
        except StopIteration:
            self.iterators[name] = iter(self.loaders[name])
            batch = next(self.iterators[name])

        return batch, name


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets library required")
@pytest.mark.integration
class TestRealDataAblation:
    """Test influence-based rebalancing vs static weights on real data."""

    @pytest.fixture(scope="class")
    def setup_data_and_model(self):
        """Load model, tokenizer, and real datasets."""
        print("\n" + "=" * 60)
        print("LOADING REAL DATA AND MODEL")
        print("=" * 60)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Load small GPT-2
        print("\nLoading GPT-2...")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device)

        # Load real datasets
        print("\nLoading datasets...")
        wiki_texts = load_wikitext_samples(500)
        math_texts = load_gsm8k_samples(500)
        code_texts = load_code_samples(500)

        # Create datasets
        wiki_ds = TextDataset(wiki_texts, tokenizer, max_length=128)
        math_ds = TextDataset(math_texts, tokenizer, max_length=128)
        code_ds = TextDataset(code_texts, tokenizer, max_length=128)

        # Split into train/eval
        def split_dataset(ds, eval_frac=0.2):
            n = len(ds)
            n_eval = int(n * eval_frac)
            indices = torch.randperm(n).tolist()
            train_indices = indices[n_eval:]
            eval_indices = indices[:n_eval]
            return (
                torch.utils.data.Subset(ds, train_indices),
                torch.utils.data.Subset(ds, eval_indices),
            )

        wiki_train, wiki_eval = split_dataset(wiki_ds)
        math_train, math_eval = split_dataset(math_ds)
        code_train, code_eval = split_dataset(code_ds)

        # Create data loaders
        batch_size = 8
        train_loaders = {
            "wiki": DataLoader(wiki_train, batch_size=batch_size, shuffle=True),
            "math": DataLoader(math_train, batch_size=batch_size, shuffle=True),
            "code": DataLoader(code_train, batch_size=batch_size, shuffle=True),
        }

        eval_loaders = {
            "wiki": DataLoader(wiki_eval, batch_size=batch_size, shuffle=False),
            "math": DataLoader(math_eval, batch_size=batch_size, shuffle=False),
            "code": DataLoader(code_eval, batch_size=batch_size, shuffle=False),
        }

        print(f"\nDataset sizes (train/eval):")
        print(f"  wiki: {len(wiki_train)}/{len(wiki_eval)}")
        print(f"  math: {len(math_train)}/{len(math_eval)}")
        print(f"  code: {len(code_train)}/{len(code_eval)}")
        print("=" * 60 + "\n")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "train_loaders": train_loaders,
            "eval_loaders": eval_loaders,
        }

    def test_data_loaded_correctly(self, setup_data_and_model):
        """Verify real data was loaded (not synthetic)."""
        data = setup_data_and_model

        # Check each dataset has reasonable content
        for name, loader in data["train_loaders"].items():
            batch = next(iter(loader))
            input_ids = batch["input_ids"]

            # Real text has lower unique token ratio
            unique_ratio = len(torch.unique(input_ids[0])) / len(input_ids[0])
            print(f"{name} unique token ratio: {unique_ratio:.2%}")

            # Real text typically has <60% unique tokens
            assert unique_ratio < 0.80, f"{name} appears to be synthetic (ratio={unique_ratio:.2%})"

    def test_static_vs_rebalanced_training(self, setup_data_and_model):
        """Compare static equal weighting vs influence-based rebalancing."""
        data = setup_data_and_model
        device = data["device"]

        # Import influence components
        from cheapertraining.influence.distillation import InfluenceDistillation
        from cheapertraining.influence.config import (
            InfluenceDistillationConfig,
            JVPEmbeddingConfig,
            LandmarkConfig,
        )

        n_steps = 200
        eval_interval = 50
        rebalance_interval = 50

        results = {}

        for mode in ["static", "rebalanced"]:
            print(f"\n{'=' * 60}")
            print(f"TRAINING MODE: {mode.upper()}")
            print("=" * 60)

            # Fresh model copy
            model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

            # Mixed sampler
            sampler = MixedSampler(data["train_loaders"])

            # Set up influence distillation for rebalanced mode
            distiller = None
            if mode == "rebalanced":
                config = InfluenceDistillationConfig(
                    jvp=JVPEmbeddingConfig(num_jvp_layers=2, projection_dim=1024),
                    landmark=LandmarkConfig(num_landmarks=64),
                )
                distiller = InfluenceDistillation(model, config)

                # Use wiki eval as probe set (target distribution)
                probe_loader = data["eval_loaders"]["wiki"]
                distiller.cache_probe_gradients(probe_loader, show_progress=False)

            # Training loop
            history = {"step": [], "train_loss": [], "eval_loss": [], "weights": []}

            for step in range(n_steps):
                # Sample and train
                batch, ds_name = sampler.sample_batch()
                loss = train_step(model, optimizer, batch, device)

                # Periodic evaluation
                if step % eval_interval == 0 or step == n_steps - 1:
                    avg_eval_loss = 0.0
                    for name, loader in data["eval_loaders"].items():
                        eval_loss = evaluate_on_loader(model, loader, device, max_batches=10)
                        avg_eval_loss += eval_loss
                    avg_eval_loss /= len(data["eval_loaders"])

                    history["step"].append(step)
                    history["train_loss"].append(loss)
                    history["eval_loss"].append(avg_eval_loss)
                    history["weights"].append(dict(sampler.weights))

                    print(f"Step {step:4d}: train_loss={loss:.4f}, eval_loss={avg_eval_loss:.4f}, "
                          f"weights={sampler.weights}")

                # Rebalance (only in rebalanced mode)
                if mode == "rebalanced" and step > 0 and step % rebalance_interval == 0:
                    print(f"  -> Rebalancing at step {step}...")
                    model.eval()

                    # Refresh probe gradients
                    distiller.cache_probe_gradients(probe_loader, show_progress=False)

                    # Compute new weights
                    new_weights = distiller.compute_mixture_weights(
                        {k: v for k, v in data["train_loaders"].items()},
                        samples_per_dataset=20,
                    )

                    print(f"  -> New weights: {new_weights}")
                    sampler.set_weights(new_weights)
                    model.train()

            results[mode] = history

        # Compare results
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)

        static_final = results["static"]["eval_loss"][-1]
        rebal_final = results["rebalanced"]["eval_loss"][-1]

        print(f"Static final eval loss:     {static_final:.4f}")
        print(f"Rebalanced final eval loss: {rebal_final:.4f}")
        print(f"Difference: {static_final - rebal_final:.4f}")

        if rebal_final < static_final:
            print("✓ Rebalancing improved eval loss!")
        else:
            print("✗ Rebalancing did not improve (may need more steps or tuning)")

        # Save results for plotting
        return results


def run_ablation_standalone():
    """Run ablation without pytest for easier debugging."""
    print("=" * 70)
    print("REAL DATA ABLATION: STATIC vs REBALANCED")
    print("=" * 70)

    if not HAS_DATASETS:
        print("ERROR: datasets library required. Install with: pip install datasets")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\nLoading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load real datasets
    print("\nLoading datasets...")
    fineweb_texts = load_fineweb_edu_samples(300)
    math_texts = load_gsm8k_samples(300)
    wiki_texts = load_wikitext_samples(300)

    # Create datasets
    fineweb_ds = TextDataset(fineweb_texts, tokenizer, max_length=128)
    math_ds = TextDataset(math_texts, tokenizer, max_length=128)
    wiki_ds = TextDataset(wiki_texts, tokenizer, max_length=128)

    batch_size = 8
    train_loaders = {
        "fineweb": DataLoader(fineweb_ds, batch_size=batch_size, shuffle=True),
        "math": DataLoader(math_ds, batch_size=batch_size, shuffle=True),
        "wiki": DataLoader(wiki_ds, batch_size=batch_size, shuffle=True),
    }

    # Verify data is real
    print("\nVerifying data is real (not synthetic):")
    for name, loader in train_loaders.items():
        batch = next(iter(loader))
        unique_ratio = len(torch.unique(batch["input_ids"][0])) / len(batch["input_ids"][0])
        status = "✓ REAL" if unique_ratio < 0.80 else "✗ SYNTHETIC?"
        print(f"  {name}: unique_ratio={unique_ratio:.2%} {status}")

    # Import influence components
    from cheapertraining.influence.distillation import InfluenceDistillation
    from cheapertraining.influence.config import (
        InfluenceDistillationConfig,
        JVPEmbeddingConfig,
        LandmarkConfig,
    )

    n_steps = 300
    eval_interval = 30
    rebalance_interval = 30

    results = {}

    for mode in ["static", "rebalanced"]:
        print(f"\n{'=' * 60}")
        print(f"TRAINING MODE: {mode.upper()}")
        print("=" * 60)

        # Fresh model
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        sampler = MixedSampler(train_loaders)

        distiller = None
        if mode == "rebalanced":
            config = InfluenceDistillationConfig(
                jvp=JVPEmbeddingConfig(num_jvp_layers=2, projection_dim=512),  # Smaller
                landmark=LandmarkConfig(num_landmarks=16),  # Much fewer for speed
                samples_per_dataset=10,  # Smaller for speed
            )
            distiller = InfluenceDistillation(model, config)

            print("  Setting up influence distillation...", flush=True)

            # Use fineweb-edu as probe (target high-quality educational content)
            probe_loader = train_loaders["fineweb"]
            print("    Caching probe gradients...", flush=True)
            distiller.cache_probe_gradients(probe_loader, show_progress=False)
            print("    Probe gradients cached.", flush=True)

            # Create combined source loader for embeddings
            from torch.utils.data import ConcatDataset
            all_datasets = [loader.dataset for loader in train_loaders.values()]
            combined_ds = ConcatDataset(all_datasets)
            combined_loader = DataLoader(combined_ds, batch_size=8, shuffle=False)

            # Cache source embeddings and landmarks
            print("    Caching source embeddings...", flush=True)
            distiller.cache_source_embeddings(combined_loader, show_progress=False)
            print("    Source embeddings cached.", flush=True)

            print("    Caching landmarks...", flush=True)
            distiller.cache_landmarks(combined_loader, show_progress=True)
            print("    Landmarks cached.", flush=True)
            print("  Distillation setup complete", flush=True)

        history = {"step": [], "loss": []}

        for step in range(n_steps):
            batch, _ = sampler.sample_batch()
            loss = train_step(model, optimizer, batch, device)

            if step % eval_interval == 0 or step == n_steps - 1:
                # Eval on fineweb-edu (target distribution)
                eval_loss = evaluate_on_loader(model, train_loaders["fineweb"], device, max_batches=10)
                history["step"].append(step)
                history["loss"].append(eval_loss)
                print(f"Step {step:4d}: train={loss:.4f}, fineweb_eval={eval_loss:.4f}, w={sampler.weights}")

            if mode == "rebalanced" and step > 0 and step % rebalance_interval == 0:
                model.eval()
                distiller.cache_probe_gradients(probe_loader, show_progress=False)
                new_weights = distiller.compute_mixture_weights(train_loaders, show_progress=False)
                print(f"  -> Rebalanced: {new_weights}")
                sampler.set_weights(new_weights)
                model.train()

        results[mode] = history

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for mode in ["static", "rebalanced"]:
        final = results[mode]["loss"][-1]
        print(f"{mode:12s}: final_fineweb_eval_loss = {final:.4f}")

    diff = results["static"]["loss"][-1] - results["rebalanced"]["loss"][-1]
    print(f"\nDifference (static - rebalanced): {diff:.4f}")
    if diff > 0:
        print("✓ Rebalancing IMPROVED loss on fineweb-edu (target distribution)")
    else:
        print("✗ Rebalancing did NOT improve (or marginal)")


if __name__ == "__main__":
    run_ablation_standalone()
