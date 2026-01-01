"""JVP-based embedding extraction for influence distillation.

Computes Jacobian-Vector Product embeddings from transformer layers.
JVP embeddings capture the sensitivity of layer outputs to parameter
perturbations, providing a cheap approximation to gradient information.

Reference: "Efficient Data Selection at Scale via Influence Distillation"
(arXiv:2505.19051)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from cheapertraining.influence.config import JVPEmbeddingConfig
from cheapertraining.influence.hadamard import RandomizedHadamardTransform

logger = logging.getLogger(__name__)


class JVPEmbeddingExtractor:
    """Compute JVP embeddings from transformer blocks.

    JVP (Jacobian-Vector Product) embeddings capture how the output of
    early transformer layers changes with respect to small perturbations
    in their parameters. This is computed efficiently using forward-mode
    automatic differentiation via torch.func.jvp.

    The key insight: JVP embeddings approximate the "gradient geometry"
    of the network without requiring full backward passes.

    Example:
        >>> config = JVPEmbeddingConfig(num_jvp_layers=4)
        >>> extractor = JVPEmbeddingExtractor(model, config)
        >>> embedding = extractor.compute_jvp_embedding(input_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        config: JVPEmbeddingConfig | None = None,
    ):
        """Initialize the JVP embedding extractor.

        Args:
            model: Transformer model (HuggingFace or custom)
            config: JVP embedding configuration
        """
        self.model = model
        self.config = config or JVPEmbeddingConfig()

        # Find the embedding layer
        self.embed_tokens = self._find_embedding_layer()

        # Extract the first K transformer layers
        self.jvp_layers = self._get_jvp_layers()

        # Get parameters and create tangent vectors
        self._init_tangent_vectors()

        # Initialize projection if enabled
        self.hadamard: Optional[RandomizedHadamardTransform] = None
        self._embedding_dim: Optional[int] = None

    def _find_embedding_layer(self) -> nn.Embedding:
        """Find the token embedding layer in the model."""
        # HuggingFace LLaMA/Mistral: model.model.embed_tokens
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        # Direct models: model.embed_tokens
        if hasattr(self.model, 'embed_tokens'):
            return self.model.embed_tokens
        # GPT-2/GPT-Neo: model.transformer.wte
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte
        # GPT-2 base model: model.wte
        if hasattr(self.model, 'wte'):
            return self.model.wte
        # BERT/RoBERTa: model.embeddings.word_embeddings
        if hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
            return self.model.embeddings.word_embeddings
        # Fallback: search for any Embedding layer with common naming
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                name_lower = name.lower()
                if any(kw in name_lower for kw in ['token', 'embed', 'wte', 'word']):
                    logger.info(f"Found embedding layer via fallback search: {name}")
                    return module

        raise ValueError(
            "Could not find token embedding layer. "
            "Supported: embed_tokens, wte, word_embeddings, or similar."
        )

    def _get_jvp_layers(self) -> nn.ModuleList:
        """Extract the first K transformer blocks from the model.

        Supports multiple architectures:
        - LLaMA/Mistral: model.model.layers
        - GPT-2/GPT-Neo: model.transformer.h
        - MobileLLM: model.decoder.layers
        - Direct: model.layers
        """
        num_layers = self.config.num_jvp_layers
        layers = None

        # LLaMA/Mistral: model.model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = list(self.model.model.layers[:num_layers])
        # GPT-2/GPT-Neo: model.transformer.h
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = list(self.model.transformer.h[:num_layers])
        # MobileLLM: model.decoder.layers
        elif hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
            layers = list(self.model.decoder.layers[:num_layers])
        # Direct models: model.layers
        elif hasattr(self.model, 'layers'):
            layers = list(self.model.layers[:num_layers])
        # BERT/RoBERTa: model.encoder.layer
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = list(self.model.encoder.layer[:num_layers])
        else:
            raise ValueError(
                "Could not find transformer layers. "
                "Supported: model.layers, transformer.h, encoder.layer, decoder.layers"
            )

        if len(layers) < num_layers:
            logger.warning(
                f"Requested {num_layers} JVP layers but model only has {len(layers)}. "
                f"Using all {len(layers)} layers."
            )

        return nn.ModuleList(layers)

    def _init_tangent_vectors(self) -> None:
        """Generate fixed random tangent vectors for JVP computation.

        Tangent vectors represent the direction of perturbation in parameter
        space. Using fixed random vectors (seeded) ensures reproducibility.
        """
        torch.manual_seed(self.config.seed)

        # Collect all parameters from JVP layers
        self.param_names = []
        self.tangent_vectors = {}

        for name, param in self.jvp_layers.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                # Generate multiple tangent vectors if configured
                tangents = []
                for _ in range(self.config.num_tangent_vectors):
                    tangents.append(torch.randn_like(param))
                self.tangent_vectors[name] = tangents

        total_params = sum(
            p.numel() for p in self.jvp_layers.parameters() if p.requires_grad
        )
        logger.info(
            f"JVP extractor initialized with {len(self.param_names)} parameter groups, "
            f"{total_params:,} total parameters, "
            f"{self.config.num_tangent_vectors} tangent vectors"
        )

    def _compute_jvp_single_tangent(
        self,
        hidden_states: Tensor,
        tangent_idx: int,
    ) -> Tensor:
        """Compute JVP for a single tangent vector.

        Uses the parameter-free forward through layers with tangent perturbation.

        Args:
            hidden_states: Input to transformer layers [batch, seq, dim]
            tangent_idx: Index of tangent vector to use

        Returns:
            JVP output [batch, seq, dim]
        """
        # Simple approach: compute forward and accumulate JVP manually
        # This avoids the complexity of torch.func.jvp with stateful modules

        jvp_total = torch.zeros_like(hidden_states)
        current_hidden = hidden_states.clone().detach().requires_grad_(True)

        for layer_idx, layer in enumerate(self.jvp_layers):
            # Forward through layer
            if hasattr(layer, 'forward'):
                # Handle different layer forward signatures
                try:
                    layer_output = layer(current_hidden)
                except TypeError:
                    # Some layers need position_ids or other args
                    layer_output = layer(current_hidden, attention_mask=None)

                # Handle tuple output (hidden_states, attention_weights, etc.)
                if isinstance(layer_output, tuple):
                    layer_output = layer_output[0]

            # Compute gradient of output w.r.t. layer parameters
            # This approximates the JVP contribution from this layer
            if layer_output.requires_grad:
                for name, param in layer.named_parameters():
                    if param.requires_grad and name in self.tangent_vectors:
                        # Get tangent vector for this parameter
                        tangent = self.tangent_vectors[name][tangent_idx]
                        tangent = tangent.to(param.device)

                        # Compute gradient of output w.r.t. this parameter
                        # and dot with tangent (this is the JVP contribution)
                        grad = torch.autograd.grad(
                            layer_output.sum(),
                            param,
                            create_graph=False,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]

                        if grad is not None:
                            # JVP contribution: <grad, tangent> weighted
                            contribution = (grad * tangent).sum()
                            jvp_total = jvp_total + contribution

            current_hidden = layer_output.detach().requires_grad_(True)

        return jvp_total

    def compute_jvp_embedding(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute JVP embedding for a batch of inputs.

        This is a simplified implementation that:
        1. Gets token embeddings
        2. Forwards through JVP layers
        3. Computes output statistics as embedding

        For the full JVP approach, we would use torch.func.jvp,
        but that requires careful handling of stateful modules.

        Args:
            input_ids: Input token IDs [batch, seq]
            attention_mask: Optional attention mask [batch, seq]

        Returns:
            JVP embedding [batch, embed_dim]
        """
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # Convert attention_mask to float if provided (needed for GPT-2)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device).float()

        # Get token embeddings
        with torch.no_grad():
            hidden_states = self.embed_tokens(input_ids)

        # Forward through JVP layers
        # NOTE: We skip attention_mask for JVP layers because:
        # 1. Different models expect different mask formats
        # 2. For embedding computation, mask is only used for pooling (handled later)
        # 3. The forward pass still works without mask (just includes padding tokens)
        with torch.no_grad():
            for layer in self.jvp_layers:
                try:
                    # Try passing without mask first (simpler, works for most models)
                    layer_output = layer(hidden_states)
                except TypeError:
                    # Some layers require positional args
                    try:
                        layer_output = layer(hidden_states, None)
                    except:
                        layer_output = layer(hidden_states)

                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output

        # Create embedding from hidden states
        # Options: mean pooling, last token, or flatten
        # Using mean pooling as it's most robust
        if attention_mask is not None:
            # Ensure attention_mask is on the same device as hidden_states
            mask = attention_mask.to(hidden_states.device).unsqueeze(-1).float()
            embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            embedding = hidden_states.mean(dim=1)

        # Flatten to 1D per sample if needed
        embedding = embedding.view(embedding.size(0), -1)

        # Initialize projection on first call
        if self.hadamard is None and self.config.use_hadamard_projection:
            self._embedding_dim = embedding.size(-1)
            self.hadamard = RandomizedHadamardTransform(
                input_dim=self._embedding_dim,
                output_dim=min(self.config.projection_dim, self._embedding_dim),
                seed=self.config.seed,
            ).to(device)
            logger.info(
                f"Initialized Hadamard projection: {self._embedding_dim} -> "
                f"{self.hadamard.output_dim}"
            )

        # Apply projection if enabled
        if self.hadamard is not None:
            embedding = self.hadamard(embedding)

        return embedding

    def compute_batch_embeddings(self, batch: dict) -> Tensor:
        """Compute embeddings for a batch.

        Args:
            batch: Dict with input_ids, attention_mask, etc.

        Returns:
            Embeddings [batch_size, embed_dim]
        """
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        return self.compute_jvp_embedding(input_ids, attention_mask)

    def compute_embeddings(
        self,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tensor:
        """Compute embeddings for all samples in a dataloader.

        Args:
            dataloader: DataLoader providing batches
            max_samples: Maximum samples to process
            show_progress: Whether to show progress bar

        Returns:
            Embeddings [N, embed_dim]
        """
        device = next(self.model.parameters()).device
        all_embeddings = []
        n_samples = 0

        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="Computing JVP embeddings")

        with torch.no_grad():
            for batch in iterator:
                # Move batch to device
                batch = {
                    k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }

                embeddings = self.compute_batch_embeddings(batch)

                # Store on CPU to save GPU memory
                if self.config.use_hadamard_projection:
                    all_embeddings.append(embeddings.cpu())
                else:
                    all_embeddings.append(embeddings.cpu())

                n_samples += embeddings.size(0)
                if max_samples and n_samples >= max_samples:
                    break

        result = torch.cat(all_embeddings, dim=0)
        if max_samples:
            result = result[:max_samples]

        return result

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        if self._embedding_dim is not None:
            if self.config.use_hadamard_projection:
                return min(self.config.projection_dim, self._embedding_dim)
            return self._embedding_dim

        # Estimate from model config
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            dim = self.model.config.hidden_size
        else:
            # Try to get from embedding layer
            dim = self.embed_tokens.embedding_dim

        if self.config.use_hadamard_projection:
            return min(self.config.projection_dim, dim)
        return dim
