"""Probe set creation and management.

Implements MobileLLM-R1 Phase I: Data Curation
1. Quality filtering (FineWeb-Edu classifier)
2. Ask-LLM scoring (using model self-evaluation)
3. Semantic deduplication (MinHash LSH)

Reference: MobileLLM-R1 paper (arXiv:2509.24945)
"""

import random
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset

from cheapertraining.influence.config import ProbeSetConfig


class ProbeSetCreator:
    """Creates representative probe sets for influence calculation.

    Implements the three-step data curation process from MobileLLM-R1:
    1. Quality filtering (FineWeb-Edu classifier, score >= 4)
    2. Ask-LLM scoring (keep top 10%)
    3. Semantic deduplication (MinHash LSH)

    The resulting probe set is used as the "target" for influence calculations.
    """

    def __init__(
        self,
        config: Optional[ProbeSetConfig] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        """Initialize probe set creator.

        Args:
            config: Probe set configuration
            tokenizer: Tokenizer for text processing
            model: Model for Ask-LLM scoring (optional, uses quality as proxy if None)
        """
        self.config = config or ProbeSetConfig()
        self.tokenizer = tokenizer
        self.model = model

        # Quality classifier (lazy loaded)
        self._quality_classifier = None
        self._quality_tokenizer = None

    def load_quality_classifier(self):
        """Load FineWeb-Edu quality classifier.

        Uses HuggingFace's fineweb-edu-classifier for quality scoring.
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._quality_classifier = AutoModelForSequenceClassification.from_pretrained(
                "HuggingFaceFW/fineweb-edu-classifier"
            )
            self._quality_tokenizer = AutoTokenizer.from_pretrained(
                "HuggingFaceFW/fineweb-edu-classifier"
            )
            self._quality_classifier.eval()
        except Exception as e:
            print(f"Warning: Could not load quality classifier: {e}")
            print("Will use length-based quality proxy instead.")
            self._quality_classifier = None

    def score_quality(self, text: str) -> float:
        """Score text quality using FineWeb-Edu classifier.

        Args:
            text: Input text to score

        Returns:
            Quality score (0-5 scale, higher is better)
        """
        if self._quality_classifier is None:
            # Fallback: use text length as proxy
            # Longer, non-empty text is assumed to be higher quality
            word_count = len(text.split())
            if word_count < 10:
                return 1.0
            elif word_count < 50:
                return 2.0
            elif word_count < 200:
                return 3.0
            elif word_count < 500:
                return 4.0
            else:
                return 5.0

        # Use classifier
        inputs = self._quality_tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._quality_classifier(**inputs)
            # The classifier outputs logits; convert to probability
            probs = outputs.logits.softmax(dim=-1)
            # Use highest probability class or weighted average
            score = probs[0].argmax().item()  # Class index as score

        return float(score)

    def score_ask_llm(self, text: str) -> float:
        """Score text using Ask-LLM method.

        Asks the model: "Does this look like good reasoning data?"
        Returns probability of "yes" response.

        If no model is provided, uses quality score as proxy.

        Args:
            text: Input text to score

        Returns:
            Score (0-1, higher means better reasoning data)
        """
        if self.model is None:
            # Use quality score as proxy
            return self.score_quality(text) / 5.0

        if self.tokenizer is None:
            raise ValueError("Tokenizer required for Ask-LLM scoring")

        # Create a simple prompt asking about data quality
        prompt = f"Is the following text high-quality educational or reasoning content? Text: {text[:500]}... Answer yes or no:"

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs["logits"]

                # Get logits for "yes" and "no" tokens
                # This is simplified - in practice you'd want actual token IDs
                # For now, use perplexity as proxy
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    inputs["input_ids"][:, 1:].contiguous().view(-1),
                    reduction="mean",
                )
                # Lower perplexity = better fit = higher quality
                score = 1.0 / (1.0 + loss.item())

            return score

        except Exception as e:
            print(f"Warning: Ask-LLM scoring failed: {e}")
            return self.score_quality(text) / 5.0

    def filter_by_quality(
        self,
        samples: Iterator[dict],
        min_score: Optional[float] = None,
        max_samples: Optional[int] = None,
    ) -> Iterator[dict]:
        """Filter samples by quality score.

        Args:
            samples: Iterator of sample dictionaries with 'text' key
            min_score: Minimum quality score (default: from config)
            max_samples: Maximum samples to process

        Yields:
            Samples that pass quality threshold with added 'quality_score' field
        """
        min_score = min_score or self.config.fineweb_edu_min_score
        count = 0

        for sample in samples:
            if max_samples and count >= max_samples:
                break

            text = sample.get("text", "")
            if not text:
                continue

            score = self.score_quality(text)
            if score >= min_score:
                sample["quality_score"] = score
                yield sample
                count += 1

    def deduplicate_minhash(
        self,
        samples: List[dict],
        threshold: Optional[float] = None,
    ) -> List[dict]:
        """Remove semantically similar samples using MinHash LSH.

        Args:
            samples: List of sample dictionaries
            threshold: Similarity threshold (default: from config)

        Returns:
            Deduplicated list of samples
        """
        threshold = threshold or self.config.dedup_similarity_threshold

        try:
            from datasketch import MinHash, MinHashLSH

            lsh = MinHashLSH(
                threshold=threshold,
                num_perm=self.config.minhash_num_perm,
            )
            unique_samples = []

            for i, sample in enumerate(samples):
                text = sample.get("text", "")

                # Create MinHash signature
                m = MinHash(num_perm=self.config.minhash_num_perm)
                for word in text.lower().split():
                    m.update(word.encode("utf8"))

                # Check for duplicates
                result = lsh.query(m)
                if len(result) == 0:
                    lsh.insert(f"sample_{i}", m)
                    unique_samples.append(sample)

            return unique_samples

        except ImportError:
            print("Warning: datasketch not installed. Skipping deduplication.")
            return samples

    def create_domain_probe_set(
        self,
        domain: str,
        source_samples: Iterator[dict],
        num_samples: Optional[int] = None,
    ) -> List[dict]:
        """Create probe set for a specific domain.

        Applies the full Phase I pipeline:
        1. Quality filtering
        2. Ask-LLM scoring (select top fraction)
        3. Deduplication

        Args:
            domain: Domain name (e.g., "code", "math", "knowledge")
            source_samples: Iterator of samples from this domain
            num_samples: Target number of samples (default: from config)

        Returns:
            List of curated samples for this domain
        """
        num_samples = num_samples or self.config.samples_per_domain

        # Step 1: Quality filtering
        # Process more samples than needed to account for filtering
        quality_filtered = list(self.filter_by_quality(
            source_samples,
            self.config.fineweb_edu_min_score,
            max_samples=num_samples * 20,  # Process 20x to have enough after filtering
        ))

        if not quality_filtered:
            print(f"Warning: No samples passed quality filter for domain {domain}")
            return []

        # Step 2: Ask-LLM scoring (or quality proxy)
        # Score all samples
        for sample in quality_filtered:
            sample["ask_llm_score"] = self.score_ask_llm(sample["text"])

        # Sort by score and keep top fraction
        sorted_samples = sorted(
            quality_filtered,
            key=lambda x: x.get("ask_llm_score", 0),
            reverse=True,
        )
        top_k = max(1, int(len(sorted_samples) * self.config.ask_llm_top_fraction))
        top_samples = sorted_samples[:top_k]

        # Step 3: Semantic deduplication
        deduped = self.deduplicate_minhash(
            top_samples,
            self.config.dedup_similarity_threshold,
        )

        # Step 4: Sample to target size
        if len(deduped) > num_samples:
            random.seed(self.config.seed)
            deduped = random.sample(deduped, num_samples)

        # Add domain label
        for sample in deduped:
            sample["domain"] = domain

        return deduped

    def create_full_probe_set(
        self,
        domain_datasets: Dict[str, IterableDataset],
    ) -> "ProbeDataset":
        """Create complete probe set across all domains.

        Args:
            domain_datasets: Dictionary mapping domain names to datasets

        Returns:
            ProbeDataset containing curated samples from all domains
        """
        all_samples = []

        for domain in self.config.domains:
            if domain not in domain_datasets:
                print(f"Warning: Dataset for domain '{domain}' not provided")
                continue

            print(f"Creating probe set for domain: {domain}")
            samples = self.create_domain_probe_set(
                domain,
                iter(domain_datasets[domain]),
                self.config.samples_per_domain,
            )
            all_samples.extend(samples)
            print(f"  Added {len(samples)} samples")

        print(f"Total probe set size: {len(all_samples)}")
        return ProbeDataset(all_samples, self.tokenizer, self.config.max_length)


class ProbeDataset(Dataset):
    """Dataset wrapper for probe set samples.

    Provides tokenized samples ready for influence calculation.
    """

    def __init__(
        self,
        samples: List[dict],
        tokenizer: Any,
        max_length: int = 2048,
    ):
        """Initialize probe dataset.

        Args:
            samples: List of sample dictionaries with 'text' field
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = sample.get("text", "")

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Pad to max_length
        attention_mask = [1] * len(tokens)
        padding_length = self.max_length - len(tokens)
        if padding_length > 0:
            tokens = tokens + [self.tokenizer.pad_token_id or 0] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(tokens, dtype=torch.long),
            "domain": sample.get("domain", "unknown"),
            "quality_score": sample.get("quality_score", 0.0),
        }

    def get_domains(self) -> List[str]:
        """Get list of unique domains in the probe set."""
        return list(set(s.get("domain", "unknown") for s in self.samples))

    def get_samples_by_domain(self, domain: str) -> List[dict]:
        """Get samples for a specific domain."""
        return [s for s in self.samples if s.get("domain") == domain]


def create_probe_dataloader(
    probe_dataset: ProbeDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for probe set.

    Args:
        probe_dataset: ProbeDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Configured DataLoader
    """
    return torch.utils.data.DataLoader(
        probe_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
