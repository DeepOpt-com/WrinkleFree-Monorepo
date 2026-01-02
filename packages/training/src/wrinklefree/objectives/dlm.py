"""DLM (Diffusion Language Model) objective.

Implements Fast-dLLM v2 block-wise masked language modeling with:
1. Token shift: logits[i-1] predicts token[i] (preserves AR representations)
2. Complementary masks: each sample duplicated with m and (1-m) masks

Reference: Fast-dLLM v2 (https://arxiv.org/abs/2509.26328)
- "If xi is masked, the model uses the hidden state at i−1 to predict xi"
- "Each training sample is duplicated into two views with masks m and m̄=1−m"
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from wrinklefree.objectives.base import Objective, ObjectiveOutput


class DLMObjective(Objective):
    """Fast-dLLM v2 masked language modeling objective.

    Implements the token shift strategy and complementary masks from
    the Fast-dLLM v2 paper. Key differences from BERT-style MLM:

    1. Token Shift: Uses logits[i-1] to predict masked token at position i.
       This preserves the pretrained AR model's next-token prediction behavior.

    2. Complementary Masks: Duplicates each sample with mask m and (1-m),
       ensuring every token is masked exactly once across the two views.

    Args:
        mask_token_id: Token ID to use for masking
        mask_prob: Probability of masking each token (default: 0.15)
        ignore_index: Label value to ignore in loss (default: -100)
        use_complementary_masks: If True, duplicate batch with complementary masks.
            Effectively halves unique samples per batch but ensures full coverage.
    """

    requires_teacher = False
    requires_hidden_states = False
    modifies_input = True

    def __init__(
        self,
        mask_token_id: int,
        mask_prob: float = 0.15,
        ignore_index: int = -100,
        use_complementary_masks: bool = True,
    ):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.ignore_index = ignore_index
        self.use_complementary_masks = use_complementary_masks

    @property
    def name(self) -> str:
        return "dlm"

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply Fast-dLLM v2 masking with complementary masks.

        Creates masked_input_ids and dlm_labels in the batch.
        With complementary masks, batch size is doubled: each sample appears
        twice with masks m and (1-m), ensuring every token is masked once.
        """
        input_ids = batch["input_ids"]
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Validate minimum sequence length
        if seq_len <= 2:
            raise ValueError(
                f"DLM requires seq_len > 2 (need at least 3 tokens: BOS, content, EOS), got {seq_len}"
            )

        # Store originals before modification
        batch["_original_input_ids"] = input_ids.clone()
        batch["_original_labels"] = batch.get("labels", input_ids).clone()

        # Create random mask
        mask = torch.rand(input_ids.shape, device=device) < self.mask_prob

        # Don't mask first token (BOS) - position 0 has no preceding token for shift
        # Don't mask last token (EOS)
        mask[:, 0] = False
        mask[:, -1] = False

        # Don't mask padding tokens
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            mask = mask & attention_mask.bool()

        if self.use_complementary_masks:
            # Create complementary mask (positions not masked in m)
            comp_mask = ~mask
            comp_mask[:, 0] = False  # Still protect BOS
            comp_mask[:, -1] = False  # Still protect EOS
            if attention_mask is not None:
                comp_mask = comp_mask & attention_mask.bool()

            # Duplicate batch: [sample1_view1, sample2_view1, ..., sample1_view2, sample2_view2, ...]
            input_ids_doubled = torch.cat([input_ids, input_ids], dim=0)
            masks_doubled = torch.cat([mask, comp_mask], dim=0)

            # Duplicate attention_mask if present
            if attention_mask is not None:
                batch["attention_mask"] = torch.cat([attention_mask, attention_mask], dim=0)

            # Duplicate other batch tensors that need to match batch size
            if "labels" in batch:
                batch["labels"] = torch.cat([batch["labels"], batch["labels"]], dim=0)

            # Also double _original_labels and _original_input_ids for distillation objectives
            batch["_original_labels"] = torch.cat(
                [batch["_original_labels"], batch["_original_labels"]], dim=0
            )
            batch["_original_input_ids"] = torch.cat(
                [batch["_original_input_ids"], batch["_original_input_ids"]], dim=0
            )

            # Apply masking
            masked_input_ids = input_ids_doubled.clone()
            masked_input_ids[masks_doubled] = self.mask_token_id

            # Set labels: only masked positions contribute to loss
            dlm_labels = torch.full_like(input_ids_doubled, self.ignore_index)
            dlm_labels[masks_doubled] = input_ids_doubled[masks_doubled]

            batch["input_ids"] = masked_input_ids
            batch["dlm_labels"] = dlm_labels
            batch["_dlm_batch_doubled"] = True
        else:
            # Single-view masking (no complementary masks)
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask] = self.mask_token_id

            dlm_labels = torch.full_like(input_ids, self.ignore_index)
            dlm_labels[mask] = input_ids[mask]

            batch["input_ids"] = masked_input_ids
            batch["dlm_labels"] = dlm_labels
            batch["_dlm_batch_doubled"] = False

        return batch

    def forward(
        self,
        model_outputs: dict[str, Any],
        batch: dict[str, Any],
        teacher_outputs: Optional[dict[str, Any]] = None,
    ) -> ObjectiveOutput:
        """Compute Fast-dLLM v2 loss with token shift.

        Token Shift Strategy: For masked position i, use logits[i-1] to predict.
        This preserves the AR model's representation where hidden[i-1] predicts token[i].

        Implementation: shift logits left by 1, shift labels left by 1.
        - shift_logits[k] = logits[k] (representing hidden state at position k)
        - shift_labels[k] = labels[k+1] (the token at position k+1)
        This means shift_logits[i-1] predicts shift_labels[i-1] = labels[i]
        """
        logits = model_outputs["logits"]  # (B, L, V)
        labels = batch.get("dlm_labels")  # (B, L)

        if labels is None:
            raise ValueError(
                "dlm_labels not found in batch. "
                "Did preprocess_batch run? Set modifies_input=True."
            )

        # Token shift: logits[i-1] predicts token at position i
        # Shift logits and labels so that position k in shifted tensors
        # represents: logits[k] predicting labels[k+1]
        shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = labels[:, 1:].contiguous()       # (B, L-1)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )

        # Count masked tokens for logging (in shifted labels)
        num_masked = (shift_labels != self.ignore_index).sum().float()
        total_tokens = shift_labels.numel()

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "num_masked": num_masked,
                "mask_ratio": num_masked / total_tokens if total_tokens > 0 else 0.0,
            },
        )

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, mask_token_id={self.mask_token_id}, "
            f"mask_prob={self.mask_prob}, use_complementary_masks={self.use_complementary_masks}"
        )
