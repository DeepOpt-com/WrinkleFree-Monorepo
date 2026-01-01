"""Influence Distillation for efficient data selection.

Implements landmark-based influence approximation where:
1. JVP embeddings are computed for all source samples (cheap)
2. Accurate gradients are computed only for landmark samples
3. Influence is propagated from landmarks to all samples via KRR

Reference: "Efficient Data Selection at Scale via Influence Distillation"
(arXiv:2505.19051)
"""

import itertools
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_handler.influence.base import DataSelector, InfluenceCalculator
from data_handler.influence.config import InfluenceDistillationConfig
from data_handler.influence.gradient import DiscriminativeGradientExtractor
from data_handler.influence.jvp_embedding import JVPEmbeddingExtractor
from data_handler.influence.landmark import LandmarkSelector

logger = logging.getLogger(__name__)


def _warn_if_synthetic_data(dataloader: DataLoader, context: str) -> None:
    """Check if dataloader appears to contain synthetic/random data and warn loudly."""
    try:
        batch = next(iter(dataloader))
        input_ids = batch.get("input_ids", batch.get("input_ids"))
        if input_ids is None:
            return

        # Check for signs of synthetic data:
        # 1. Very uniform distribution (random integers)
        # 2. No repeated tokens (real text has repetition)
        # 3. TensorDataset source

        sample = input_ids[0].float()
        unique_ratio = len(torch.unique(sample)) / len(sample)

        # Real text typically has <50% unique tokens due to common words
        # Random data has ~95%+ unique tokens
        if unique_ratio > 0.85:
            print("\n" + "=" * 80)
            print("⚠️  WARNING: DATA APPEARS TO BE SYNTHETIC/RANDOM! ⚠️")
            print("=" * 80)
            print(f"Context: {context}")
            print(f"Unique token ratio: {unique_ratio:.1%} (real text is typically <50%)")
            print()
            print("Influence-based rebalancing WILL NOT WORK with random data!")
            print("All datasets look identical → influence scores are noise.")
            print()
            print("Use REAL data from configs/data/mixed_pretrain.yaml")
            print("=" * 80 + "\n")
            logger.warning(f"Synthetic data detected in {context}! Unique ratio: {unique_ratio:.1%}")
    except Exception:
        pass  # Don't fail on check


class InfluenceDistillation(InfluenceCalculator, DataSelector):
    """Landmark-based influence distillation.

    This class implements the Influence Distillation algorithm for efficient
    data selection and continuous rebalancing. The key insight is that
    influence computation can be decomposed into:

    1. A cheap JVP embedding that captures gradient geometry
    2. Accurate gradients computed only for a small set of landmarks
    3. KRR propagation to estimate all samples' influences

    Algorithm:
        1. Compute JVP embeddings E_S for all source samples
        2. Select landmarks L, get E_L
        3. Solve KRR: C = E_S @ E_L.T @ (E_L @ E_L.T + λI)^{-1}
        4. Compute accurate gradients G_L for landmarks only
        5. Cache target gradient g_T from probe set
        6. Propagate: p = C @ (G_L @ g_T)

    Example:
        >>> config = InfluenceDistillationConfig()
        >>> distiller = InfluenceDistillation(model, config)
        >>> distiller.cache_probe_gradients(probe_loader)
        >>> scores = distiller.compute_influence_scores(source_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: InfluenceDistillationConfig | None = None,
    ):
        """Initialize the influence distillation system.

        Args:
            model: Transformer model (HuggingFace or custom)
            config: Influence distillation configuration
        """
        self.model = model
        self.config = config or InfluenceDistillationConfig()

        # Initialize extractors
        self.jvp_extractor = JVPEmbeddingExtractor(model, self.config.jvp)
        self.grad_extractor = DiscriminativeGradientExtractor(model)
        self.landmark_selector = LandmarkSelector(self.config.landmark)

        # Cached state
        self._source_embeddings: Optional[Tensor] = None
        self._landmark_indices: Optional[Tensor] = None
        self._landmark_embeddings: Optional[Tensor] = None
        self._coefficients: Optional[Tensor] = None
        self._landmark_gradients: Optional[Tensor] = None
        self._target_gradient: Optional[Tensor] = None
        self._cholesky_factor: Optional[Tensor] = None

    # =========================================================================
    # InfluenceCalculator interface
    # =========================================================================

    @property
    def is_cached(self) -> bool:
        """Check if probe gradients are cached."""
        return self._target_gradient is not None

    def clear_cache(self) -> None:
        """Clear all cached state."""
        self._source_embeddings = None
        self._landmark_indices = None
        self._landmark_embeddings = None
        self._coefficients = None
        self._landmark_gradients = None
        self._target_gradient = None
        self._cholesky_factor = None

    def cache_probe_gradients(
        self,
        probe_dataloader: DataLoader,
        show_progress: bool = True,
    ) -> None:
        """Compute and cache the target gradient from probe set.

        The target gradient is the mean gradient over the probe set,
        representing "what gradient direction helps the target distribution".

        Args:
            probe_dataloader: DataLoader for probe set
            show_progress: Whether to show progress bar
        """
        _warn_if_synthetic_data(probe_dataloader, "probe set (cache_probe_gradients)")
        device = next(self.model.parameters()).device
        all_grads = []

        iterator = probe_dataloader
        if show_progress:
            iterator = tqdm(probe_dataloader, desc="Caching probe gradients")

        for batch in iterator:
            batch = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            grad = self.grad_extractor.compute_aggregated_gradient(batch)
            all_grads.append(grad.cpu())

        # Mean gradient over probe set
        self._target_gradient = torch.stack(all_grads).mean(dim=0)
        logger.info(f"Cached target gradient with dim {self._target_gradient.size()}")

    def cache_source_embeddings(
        self,
        source_loader: DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> None:
        """Compute and cache JVP embeddings for source samples.

        Args:
            source_loader: DataLoader for source samples
            max_samples: Maximum samples to process
            show_progress: Whether to show progress bar
        """
        _warn_if_synthetic_data(source_loader, "source data (cache_source_embeddings)")
        self._source_embeddings = self.jvp_extractor.compute_embeddings(
            source_loader,
            max_samples=max_samples,
            show_progress=show_progress,
        )
        logger.info(
            f"Cached {self._source_embeddings.size(0)} source embeddings "
            f"with dim {self._source_embeddings.size(1)}"
        )

    def cache_landmarks(
        self,
        source_loader: Optional[DataLoader] = None,
        show_progress: bool = True,
    ) -> None:
        """Select landmarks and compute KRR coefficients.

        Must be called after cache_source_embeddings().

        Args:
            source_loader: Optional loader to compute landmark gradients
            show_progress: Whether to show progress bar
        """
        if self._source_embeddings is None:
            raise RuntimeError(
                "Source embeddings not cached. Call cache_source_embeddings first."
            )

        # Select landmark indices
        self._landmark_indices = self.landmark_selector.select(
            self._source_embeddings
        )
        self._landmark_embeddings = self._source_embeddings[self._landmark_indices]

        logger.info(f"Selected {len(self._landmark_indices)} landmarks")

        # Compute KRR coefficients
        self._compute_krr_coefficients()

        # Compute landmark gradients if source_loader provided
        if source_loader is not None:
            self._compute_landmark_gradients(source_loader, show_progress)

    def _compute_krr_coefficients(self) -> None:
        """Solve KRR to get propagation coefficients C.

        C = E_S @ E_L.T @ (E_L @ E_L.T + λI)^{-1}

        Uses Cholesky decomposition for numerical stability.
        """
        E_S = self._source_embeddings
        E_L = self._landmark_embeddings
        lambda_reg = self.config.krr.lambda_reg

        # Move to GPU for computation
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        E_L = E_L.to(device)

        # Compute kernel matrix K = E_L @ E_L.T
        K = E_L @ E_L.T  # [L, L]
        K = K + lambda_reg * torch.eye(K.size(0), device=device)

        # Cholesky decomposition for stable solve
        try:
            L_chol = torch.linalg.cholesky(K)
            self._cholesky_factor = L_chol.cpu()
        except RuntimeError as e:
            if "not positive definite" in str(e):
                logger.warning("Cholesky failed, increasing lambda 10x")
                K = K + 9 * lambda_reg * torch.eye(K.size(0), device=device)
                L_chol = torch.linalg.cholesky(K)
                self._cholesky_factor = L_chol.cpu()
            else:
                raise

        # Solve for coefficients in chunks to handle large N
        chunk_size = self.config.krr.chunk_size
        C_chunks = []

        for i in range(0, E_S.size(0), chunk_size):
            chunk = E_S[i:i + chunk_size].to(device)
            rhs = chunk @ E_L.T  # [chunk, L]
            C_chunk = torch.cholesky_solve(rhs.T, L_chol).T  # [chunk, L]
            C_chunks.append(C_chunk.cpu())

        self._coefficients = torch.cat(C_chunks, dim=0)
        logger.info(f"Computed KRR coefficients with shape {self._coefficients.size()}")

    def _compute_landmark_gradients(
        self,
        source_loader: DataLoader,
        show_progress: bool = True,
    ) -> None:
        """Compute accurate gradients for landmark samples.

        Args:
            source_loader: DataLoader for source samples
            show_progress: Whether to show progress bar
        """
        if self._landmark_indices is None:
            raise RuntimeError("Landmarks not selected. Call cache_landmarks first.")

        device = next(self.model.parameters()).device
        landmark_set = set(self._landmark_indices.tolist())
        landmark_grads = {}

        iterator = source_loader
        if show_progress:
            iterator = tqdm(source_loader, desc="Computing landmark gradients")

        sample_idx = 0
        for batch in iterator:
            batch_size = batch["input_ids"].size(0)

            for i in range(batch_size):
                if sample_idx in landmark_set:
                    # Extract single sample
                    single_batch = {
                        k: v[i:i + 1].to(device) if isinstance(v, Tensor) else v
                        for k, v in batch.items()
                    }
                    grad = self.grad_extractor.compute_aggregated_gradient(single_batch)
                    landmark_grads[sample_idx] = grad.cpu()

                sample_idx += 1

                if len(landmark_grads) == len(self._landmark_indices):
                    break

            if len(landmark_grads) == len(self._landmark_indices):
                break

        # Stack in correct order
        grads_list = [
            landmark_grads[idx.item()]
            for idx in self._landmark_indices
            if idx.item() in landmark_grads
        ]

        if len(grads_list) < len(self._landmark_indices):
            logger.warning(
                f"Only found gradients for {len(grads_list)} of "
                f"{len(self._landmark_indices)} landmarks"
            )

        self._landmark_gradients = torch.stack(grads_list)
        logger.info(
            f"Computed {self._landmark_gradients.size(0)} landmark gradients "
            f"with dim {self._landmark_gradients.size(1)}"
        )

    def compute_influence_scores(
        self,
        source_loader: Optional[DataLoader] = None,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tensor:
        """Compute influence scores for all cached source samples.

        Uses the distillation formula:
            p = C @ (G_L @ g_T)

        Where:
            - C: KRR coefficients [N, L]
            - G_L: Landmark gradients [L, D]
            - g_T: Target gradient [D]

        Args:
            source_loader: Ignored (uses cached embeddings)
            max_samples: Maximum samples to return
            show_progress: Ignored

        Returns:
            Influence scores [N_source]
        """
        if self._coefficients is None:
            raise RuntimeError(
                "Coefficients not computed. Call cache_landmarks first."
            )
        if self._landmark_gradients is None:
            raise RuntimeError(
                "Landmark gradients not computed. "
                "Call cache_landmarks with source_loader."
            )
        if self._target_gradient is None:
            raise RuntimeError(
                "Target gradient not cached. Call cache_probe_gradients first."
            )

        # Propagate influence: p = C @ (G_L @ g_T)
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        G_L = self._landmark_gradients.to(device)
        g_T = self._target_gradient.to(device)
        C = self._coefficients.to(device)

        # Landmark influences: [L]
        p_L = G_L @ g_T

        # Propagate to all samples: [N]
        p_all = C @ p_L

        result = p_all.cpu()
        if max_samples:
            result = result[:max_samples]

        return result

    def compute_batch_influence_aggregated(
        self,
        batch: dict,
    ) -> Tensor:
        """Compute influence for a new batch (online mode).

        For continuous rebalancing, we compute JVP embeddings for the batch,
        then use the cached Cholesky factor to compute coefficients on-the-fly.

        Args:
            batch: Dict with input_ids, attention_mask, etc.

        Returns:
            Influence scores [batch_size]
        """
        if self._cholesky_factor is None:
            raise RuntimeError("KRR not computed. Call cache_landmarks first.")
        if self._landmark_gradients is None:
            raise RuntimeError("Landmark gradients not computed.")
        if self._target_gradient is None:
            raise RuntimeError("Target gradient not cached.")

        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        # Compute JVP embeddings for batch
        batch_embeds = self.jvp_extractor.compute_batch_embeddings(batch)

        # Compute coefficients for this batch
        E_batch = batch_embeds.to(device)
        E_L = self._landmark_embeddings.to(device)
        L_chol = self._cholesky_factor.to(device)

        rhs = E_batch @ E_L.T  # [B, L]
        C_batch = torch.cholesky_solve(rhs.T, L_chol).T  # [B, L]

        # Propagate influence
        G_L = self._landmark_gradients.to(device)
        g_T = self._target_gradient.to(device)
        p_L = G_L @ g_T  # [L]
        p_batch = C_batch @ p_L  # [B]

        return p_batch.cpu()

    # =========================================================================
    # DataSelector interface
    # =========================================================================

    def select(
        self,
        source_loader: DataLoader,
        budget_k: int,
        target_loader: Optional[DataLoader] = None,
        show_progress: bool = True,
    ) -> list[int]:
        """Select top-k most influential samples.

        Full pipeline:
        1. Cache probe gradients (if target_loader provided)
        2. Compute source embeddings
        3. Select landmarks and compute gradients
        4. Compute influence scores
        5. Return top-k indices

        Args:
            source_loader: DataLoader for candidate samples
            budget_k: Number of samples to select
            target_loader: Optional probe/target DataLoader
            show_progress: Whether to show progress bar

        Returns:
            List of selected sample indices
        """
        # Cache probe gradients if needed
        if target_loader is not None:
            self.cache_probe_gradients(target_loader, show_progress)
        elif self._target_gradient is None:
            raise ValueError(
                "target_loader must be provided or cache_probe_gradients "
                "must be called first"
            )

        # Compute source embeddings
        self.cache_source_embeddings(source_loader, show_progress=show_progress)

        # Select landmarks and compute gradients
        self.cache_landmarks(source_loader, show_progress)

        # Compute influence scores
        scores = self.compute_influence_scores()

        # Select top-k
        _, top_indices = torch.topk(scores, min(budget_k, len(scores)))
        return top_indices.tolist()

    # =========================================================================
    # Continuous rebalancing support
    # =========================================================================

    def refresh_for_rebalancing(
        self,
        probe_dataloader: DataLoader,
        source_loader: Optional[DataLoader] = None,
        show_progress: bool = True,
    ) -> None:
        """Refresh caches for continuous rebalancing.

        When model weights change during training:
        - Recompute target gradient (probe gradients change)
        - Recompute landmark gradients
        - Keep JVP embeddings and KRR coefficients (approximately stable)

        Args:
            probe_dataloader: Probe set DataLoader
            source_loader: Source DataLoader (for landmark gradients)
            show_progress: Whether to show progress bar
        """
        # Refresh target gradient
        self.cache_probe_gradients(probe_dataloader, show_progress)

        # Refresh landmark gradients if source_loader provided
        if source_loader is not None and self._landmark_indices is not None:
            self._compute_landmark_gradients(source_loader, show_progress)

    def compute_mixture_weights(
        self,
        dataset_loaders: dict[str, DataLoader],
        show_progress: bool = True,
    ) -> dict[str, float]:
        """Compute optimal mixture weights for multiple datasets.

        For each dataset, computes average influence on the probe set
        and converts to normalized weights.

        Args:
            dataset_loaders: Dict mapping dataset names to DataLoaders
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping dataset names to weights (sum to 1)
        """
        if self._target_gradient is None:
            raise RuntimeError("Target gradient not cached.")

        # Warn if any dataset appears to be synthetic
        for name, loader in dataset_loaders.items():
            _warn_if_synthetic_data(loader, f"dataset '{name}' (compute_mixture_weights)")

        influences = {}
        for name, loader in dataset_loaders.items():
            total_influence = 0.0
            n_samples = 0

            iterator = itertools.islice(loader, self.config.samples_per_dataset)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    desc=f"Computing influence for {name}",
                    total=self.config.samples_per_dataset,
                )

            for batch in iterator:
                batch_influence = self.compute_batch_influence_aggregated(batch)
                total_influence += batch_influence.sum().item()
                n_samples += batch_influence.numel()

            influences[name] = total_influence / max(n_samples, 1)

        return self._influences_to_weights(influences)

    def _influences_to_weights(
        self,
        influences: dict[str, float],
    ) -> dict[str, float]:
        """Convert influences to normalized weights.

        Args:
            influences: Dict mapping names to influence scores

        Returns:
            Dict mapping names to weights (sum to 1)
        """
        # Shift to make all positive
        min_inf = min(influences.values())
        shifted = {k: v - min_inf + 1e-8 for k, v in influences.items()}

        # Normalize
        total = sum(shifted.values())
        weights = {k: v / total for k, v in shifted.items()}

        # Apply constraints
        for k in weights:
            weights[k] = max(
                self.config.min_weight,
                min(self.config.max_weight, weights[k])
            )

        # Re-normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> dict[str, float]:
        """Compute loss on a fixed evaluation set.

        Use this to monitor training progress on a held-out set.

        Args:
            eval_dataloader: DataLoader for evaluation set
            max_batches: Maximum batches to evaluate (None = all)

        Returns:
            Dict with 'loss' and 'perplexity'
        """
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        max_batches = max_batches or self.config.eval_batches or None
        iterator = eval_dataloader
        if max_batches:
            iterator = itertools.islice(eval_dataloader, max_batches)

        for batch in iterator:
            batch = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']

            # Count non-padding tokens
            labels = batch.get('labels', batch.get('input_ids'))
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        if was_training:
            self.model.train()

        avg_loss = total_loss / max(total_tokens, 1)
        return {
            'loss': avg_loss,
            'perplexity': min(float('inf'), torch.exp(torch.tensor(avg_loss)).item()),
            'tokens': total_tokens,
        }


def create_influence_distillation(
    model: nn.Module,
    config: Optional[InfluenceDistillationConfig] = None,
) -> InfluenceDistillation:
    """Factory function to create an InfluenceDistillation instance.

    Args:
        model: Transformer model
        config: Optional configuration

    Returns:
        Configured InfluenceDistillation instance
    """
    return InfluenceDistillation(model, config)
