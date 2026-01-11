"""Tests for DLMObjective (Fast-dLLM v2 with token shift and complementary masks)."""

import pytest
import torch

from wf_train.objectives._legacy.dlm import DLMObjective


class TestDLMPreprocessing:
    """Test DLM preprocessing (masking with complementary masks)."""

    def test_masks_tokens_without_complementary(self):
        """Test that DLM preprocessing masks tokens correctly (single view)."""
        mask_token_id = 999
        obj = DLMObjective(
            mask_token_id=mask_token_id, mask_prob=0.5, use_complementary_masks=False
        )

        # Create input that doesn't contain mask token
        input_ids = torch.randint(0, 100, (2, 10))
        input_ids[input_ids == mask_token_id] = 0

        # DLM requires labels to identify output tokens
        # All tokens are outputs here (labels != -100)
        labels = input_ids.clone()

        batch = {"input_ids": input_ids.clone(), "labels": labels}

        # Run preprocessing
        processed = obj.preprocess_batch(batch)

        # Batch size should remain the same (no complementary masks)
        assert processed["input_ids"].shape[0] == 2

        # Check that some tokens are masked (with high prob, should have some)
        masked_ids = processed["input_ids"]
        dlm_labels = processed["dlm_labels"]

        # Check masking was applied
        mask = masked_ids == mask_token_id
        assert mask.any(), "No tokens were masked (prob=0.5 should mask some)"

    def test_complementary_masks_doubles_batch(self):
        """Test that complementary masks double the batch size."""
        mask_token_id = 999
        obj = DLMObjective(
            mask_token_id=mask_token_id, mask_prob=0.3, use_complementary_masks=True
        )

        batch_size = 4
        input_ids = torch.randint(0, 100, (batch_size, 10))
        input_ids[input_ids == mask_token_id] = 0
        labels = input_ids.clone()  # All tokens are outputs

        batch = {"input_ids": input_ids.clone(), "labels": labels}
        processed = obj.preprocess_batch(batch)

        # Batch should be doubled
        assert processed["input_ids"].shape[0] == batch_size * 2
        assert processed["dlm_labels"].shape[0] == batch_size * 2
        assert processed["_dlm_batch_doubled"] is True

    def test_complementary_masks_cover_all_tokens(self):
        """Test that between both views, every OUTPUT token is masked exactly once."""
        mask_token_id = 999
        obj = DLMObjective(
            mask_token_id=mask_token_id, mask_prob=0.3, use_complementary_masks=True
        )

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        input_ids[input_ids == mask_token_id] = 0
        labels = input_ids.clone()  # All tokens are outputs

        batch = {"input_ids": input_ids.clone(), "labels": labels}
        processed = obj.preprocess_batch(batch)

        masked_ids = processed["input_ids"]

        # Check that for each original sample, between both views,
        # every non-boundary token is masked exactly once
        for i in range(batch_size):
            view1_mask = masked_ids[i] == mask_token_id  # First view
            view2_mask = masked_ids[batch_size + i] == mask_token_id  # Second view

            # Middle tokens (excluding first and last) should be masked in one view XOR other
            for j in range(1, seq_len - 1):
                xor = view1_mask[j].item() != view2_mask[j].item()
                assert xor, f"Token {j} should be masked in exactly one view"

    def test_preserves_first_last_tokens(self):
        """Test that first and last tokens are never masked."""
        mask_token_id = 999
        # Use high mask prob to ensure masking would happen
        obj = DLMObjective(
            mask_token_id=mask_token_id, mask_prob=0.9, use_complementary_masks=True
        )

        input_ids = torch.randint(0, 100, (4, 20))
        labels = input_ids.clone()  # All tokens are outputs
        batch = {"input_ids": input_ids.clone(), "labels": labels}

        processed = obj.preprocess_batch(batch)
        masked_ids = processed["input_ids"]

        # First and last tokens should never be masked (for all views)
        assert (masked_ids[:, 0] != mask_token_id).all()
        assert (masked_ids[:, -1] != mask_token_id).all()

    def test_respects_attention_mask(self):
        """Test that padding tokens are not masked."""
        mask_token_id = 999
        obj = DLMObjective(
            mask_token_id=mask_token_id, mask_prob=0.9, use_complementary_masks=False
        )

        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones_like(input_ids)
        # Mark last 3 tokens as padding
        attention_mask[:, -3:] = 0
        labels = input_ids.clone()  # All tokens are outputs

        batch = {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask,
            "labels": labels,
        }

        processed = obj.preprocess_batch(batch)
        masked_ids = processed["input_ids"]
        dlm_labels = processed["dlm_labels"]

        # Padding positions should not be masked
        assert (masked_ids[:, -3:] != mask_token_id).all()
        # Padding positions should have ignore_index in dlm_labels
        assert (dlm_labels[:, -3:] == -100).all()

    def test_stores_original_input(self):
        """Test that original input_ids are stored for debugging."""
        obj = DLMObjective(
            mask_token_id=999, mask_prob=0.5, use_complementary_masks=False
        )
        input_ids = torch.randint(0, 100, (2, 10))
        labels = input_ids.clone()  # All tokens are outputs
        batch = {"input_ids": input_ids.clone(), "labels": labels}

        processed = obj.preprocess_batch(batch)

        assert "_original_input_ids" in processed
        assert torch.equal(processed["_original_input_ids"], input_ids)


class TestDLMForward:
    """Test DLM forward pass (loss computation with token shift)."""

    def test_computes_loss_with_token_shift(self):
        """Test DLM forward pass loss calculation with token shift.

        Fast-dLLM v2: logits[i-1] predicts token at position i.
        After shift: shift_logits[k] predicts shift_labels[k] = labels[k+1]
        """
        obj = DLMObjective(mask_token_id=999, use_complementary_masks=False)

        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Create dummy labels with masked positions
        # Token shift means labels[i] is predicted by logits[i-1]
        labels = torch.full((batch_size, seq_len), -100)
        labels[0, 3] = 5  # Token at position 3, predicted by logits[2]
        labels[1, 5] = 10  # Token at position 5, predicted by logits[4]

        batch = {"dlm_labels": labels}
        model_outputs = {"logits": logits}

        output = obj.forward(model_outputs, batch)

        assert output.loss > 0
        assert output.loss.item() == output.metrics["loss"].item()
        # After shift, we have 2 non-ignored labels in shift_labels
        assert output.metrics["num_masked"] == 2
        # Mask ratio is computed over shifted labels (seq_len - 1)
        assert output.metrics["mask_ratio"] == 2 / (batch_size * (seq_len - 1))

    def test_raises_without_labels(self):
        """Test that forward raises if dlm_labels missing."""
        obj = DLMObjective(mask_token_id=999, use_complementary_masks=False)

        logits = torch.randn(2, 8, 100)
        batch = {}  # Missing dlm_labels
        model_outputs = {"logits": logits}

        with pytest.raises(ValueError, match="dlm_labels not found"):
            obj.forward(model_outputs, batch)

    def test_token_shift_behavior(self):
        """Verify Fast-dLLM v2 token shift: logits[i-1] predicts token[i].

        Unlike BERT-style MLM (logits[i] predicts token[i]),
        Fast-dLLM uses logits[i-1] to preserve AR representations.
        """
        obj = DLMObjective(mask_token_id=999, use_complementary_masks=False)

        batch_size, seq_len, vocab_size = 1, 8, 10
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        # To predict token at position 3, we need high logit at position 2
        # (due to token shift: logits[2] predicts labels[3])
        logits[0, 2, 5] = 10.0  # High logit for token 5 at position 2

        # Label says position 3 should be token 5
        labels = torch.full((batch_size, seq_len), -100)
        labels[0, 3] = 5

        batch = {"dlm_labels": labels}
        model_outputs = {"logits": logits}

        output = obj.forward(model_outputs, batch)

        # With correct prediction at position 2 for label at 3, loss should be low
        assert output.loss < 1.0

    def test_token_shift_wrong_position_high_loss(self):
        """Verify that putting logit at wrong position gives high loss."""
        obj = DLMObjective(mask_token_id=999, use_complementary_masks=False)

        batch_size, seq_len, vocab_size = 1, 8, 10
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        # Put high logit at position 3 (same as label position)
        # But with token shift, this predicts position 4, not 3!
        logits[0, 3, 5] = 10.0

        # Label says position 3 should be token 5
        labels = torch.full((batch_size, seq_len), -100)
        labels[0, 3] = 5

        batch = {"dlm_labels": labels}
        model_outputs = {"logits": logits}

        output = obj.forward(model_outputs, batch)

        # Wrong position should give high loss
        assert output.loss > 2.0


class TestDLMIntegration:
    """Integration tests for DLM with ObjectiveManager."""

    def test_factory_creates_dlm(self):
        """Test that factory can create DLM objective."""
        from wf_train.objectives.factory import create_objective

        obj = create_objective("dlm", {"mask_token_id": 999, "mask_prob": 0.2})

        assert isinstance(obj, DLMObjective)
        assert obj.mask_token_id == 999
        assert obj.mask_prob == 0.2
        assert obj.use_complementary_masks is True  # Default

    def test_factory_creates_dlm_without_complementary_masks(self):
        """Test that factory can disable complementary masks."""
        from wf_train.objectives.factory import create_objective

        obj = create_objective(
            "dlm",
            {"mask_token_id": 999, "mask_prob": 0.2, "use_complementary_masks": False},
        )

        assert isinstance(obj, DLMObjective)
        assert obj.use_complementary_masks is False

    def test_dlm_in_objective_manager(self):
        """Test DLM works with ObjectiveManager (requires SFT)."""
        from wf_train.objectives.manager import ObjectiveManager
        from wf_train.objectives.sft import SFTObjective

        dlm = DLMObjective(
            mask_token_id=999, mask_prob=0.15, use_complementary_masks=False
        )
        sft = SFTObjective()  # DLM requires SFT

        manager = ObjectiveManager(
            objectives={"dlm": dlm, "sft": sft},
            weights={"dlm": 1.0, "sft": 1.0},
        )

        assert "dlm" in manager.objectives
        assert manager.any_modifies_input  # DLM modifies input


class TestDLMConfigSaving:
    """Test dlm_config.json format for checkpoint saving."""

    def test_dlm_config_json_format(self):
        """Verify dlm_config.json has expected fields."""
        obj = DLMObjective(mask_token_id=999, mask_prob=0.15)

        # This is the config format saved by ContinuedPretrainingTrainer.save_checkpoint()
        config = {
            "mask_token_id": obj.mask_token_id,
            "mask_prob": obj.mask_prob,
            "ignore_index": obj.ignore_index,
            "use_complementary_masks": obj.use_complementary_masks,
            "training_method": "fast-dllm-v2",  # Updated to reflect correct algorithm
        }

        # Verify expected keys
        expected_keys = {
            "mask_token_id",
            "mask_prob",
            "ignore_index",
            "use_complementary_masks",
            "training_method",
        }
        assert set(config.keys()) == expected_keys

        # Verify values
        assert config["mask_token_id"] == 999
        assert config["mask_prob"] == 0.15
        assert config["ignore_index"] == -100  # Default
        assert config["use_complementary_masks"] is True  # Default
        assert config["training_method"] == "fast-dllm-v2"

    def test_dlm_objective_has_required_attributes(self):
        """Verify DLMObjective exposes all attributes needed for config saving."""
        obj = DLMObjective(
            mask_token_id=42, mask_prob=0.2, ignore_index=-200, use_complementary_masks=False
        )

        # These attributes are accessed by save_checkpoint() for dlm_config.json
        assert hasattr(obj, "mask_token_id")
        assert hasattr(obj, "mask_prob")
        assert hasattr(obj, "ignore_index")
        assert hasattr(obj, "use_complementary_masks")

        assert obj.mask_token_id == 42
        assert obj.mask_prob == 0.2
        assert obj.ignore_index == -200
        assert obj.use_complementary_masks is False

    def test_dlm_config_json_file_creation(self, tmp_path):
        """Test that dlm_config.json is created correctly during checkpoint save."""
        import json

        # Simulate what save_checkpoint does
        checkpoint_dir = tmp_path / "checkpoints" / "test_checkpoint"
        checkpoint_dir.mkdir(parents=True)

        dlm_obj = DLMObjective(mask_token_id=999, mask_prob=0.15)

        # This mirrors the logic in ContinuedPretrainingTrainer.save_checkpoint()
        dlm_config = {
            "mask_token_id": dlm_obj.mask_token_id,
            "mask_prob": dlm_obj.mask_prob,
            "ignore_index": dlm_obj.ignore_index,
            "use_complementary_masks": dlm_obj.use_complementary_masks,
            "training_method": "fast-dllm-v2",
        }
        with open(checkpoint_dir / "dlm_config.json", "w") as f:
            json.dump(dlm_config, f, indent=2)

        # Verify file was created
        assert (checkpoint_dir / "dlm_config.json").exists()

        # Verify content
        with open(checkpoint_dir / "dlm_config.json") as f:
            saved_config = json.load(f)

        assert saved_config["mask_token_id"] == 999
        assert saved_config["mask_prob"] == 0.15
        assert saved_config["ignore_index"] == -100
        assert saved_config["use_complementary_masks"] is True
        assert saved_config["training_method"] == "fast-dllm-v2"

    def test_objective_manager_dlm_access(self):
        """Test that ObjectiveManager provides access to DLM objective for config extraction."""
        from wf_train.objectives.manager import ObjectiveManager
        from wf_train.objectives.sft import SFTObjective

        dlm = DLMObjective(mask_token_id=42, mask_prob=0.3, use_complementary_masks=False)
        sft = SFTObjective()  # DLM requires SFT
        manager = ObjectiveManager(
            objectives={"dlm": dlm, "sft": sft},
            weights={"dlm": 1.0, "sft": 1.0},
        )

        # This is how save_checkpoint accesses the DLM objective
        # objectives is a ModuleDict, so use dictionary-style access
        assert "dlm" in manager.objectives
        retrieved_dlm = manager.objectives["dlm"]
        assert retrieved_dlm is not None
        assert retrieved_dlm.mask_token_id == 42
        assert retrieved_dlm.mask_prob == 0.3
        assert retrieved_dlm.use_complementary_masks is False


class TestDLMOutputOnlyMasking:
    """Tests for output-only masking (only masks where labels != -100)."""

    def test_only_masks_output_tokens(self):
        """Test that DLM only masks tokens where labels != -100."""
        mask_token_id = 999
        obj = DLMObjective(
            mask_token_id=mask_token_id, mask_prob=0.9, use_complementary_masks=False
        )

        # Create input with instruction tokens (labels=-100) and output tokens
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        # First 5 tokens are instruction (labels=-100), last 5 are output
        labels = torch.tensor([[-100, -100, -100, -100, -100, 10, 11, 12, 13, 14]])

        batch = {"input_ids": input_ids.clone(), "labels": labels}
        processed = obj.preprocess_batch(batch)

        masked_ids = processed["input_ids"]

        # Instruction tokens (positions 0-4) should NEVER be masked
        # Even with mask_prob=0.9, instruction tokens are protected
        assert (masked_ids[0, :5] == input_ids[0, :5]).all(), (
            "Instruction tokens (labels=-100) should never be masked"
        )

    def test_instruction_tokens_never_masked_with_complementary(self):
        """Test instruction tokens are never masked even with complementary masks."""
        mask_token_id = 999
        obj = DLMObjective(
            mask_token_id=mask_token_id, mask_prob=0.5, use_complementary_masks=True
        )

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        # First 3 tokens are instruction
        labels = torch.tensor([[-100, -100, -100, 10, 11, 12, 13, 14]])

        batch = {"input_ids": input_ids.clone(), "labels": labels}
        processed = obj.preprocess_batch(batch)

        masked_ids = processed["input_ids"]

        # Both views should have instruction tokens unmasked
        # View 1: masked_ids[0, :3], View 2: masked_ids[1, :3]
        assert (masked_ids[0, :3] == input_ids[0, :3]).all()
        assert (masked_ids[1, :3] == input_ids[0, :3]).all()

    def test_output_tokens_can_be_masked(self):
        """Test that output tokens (labels != -100) can be masked."""
        mask_token_id = 999
        obj = DLMObjective(
            mask_token_id=mask_token_id, mask_prob=0.9, use_complementary_masks=False
        )

        # 10 output tokens (all have valid labels)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        labels = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])

        batch = {"input_ids": input_ids.clone(), "labels": labels}
        processed = obj.preprocess_batch(batch)

        masked_ids = processed["input_ids"]

        # With mask_prob=0.9, some output tokens should be masked
        # (excluding first and last tokens which are protected)
        middle_tokens = masked_ids[0, 1:-1]
        assert (middle_tokens == mask_token_id).any(), (
            "Output tokens should be masked with mask_prob=0.9"
        )

    def test_requires_labels_in_batch(self):
        """Test that DLM raises error if labels not provided."""
        obj = DLMObjective(mask_token_id=999, mask_prob=0.5)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        batch = {"input_ids": input_ids}  # No labels!

        with pytest.raises(ValueError, match="DLM requires 'labels'"):
            obj.preprocess_batch(batch)


class TestDLMRequiresSFT:
    """Tests for DLM requires SFT validation in ObjectiveManager."""

    def test_dlm_without_sft_raises_error(self):
        """Test that DLM without SFT raises an error."""
        from wf_train.objectives.manager import ObjectiveManager

        dlm = DLMObjective(mask_token_id=999)

        with pytest.raises(ValueError, match="DLM requires SFT"):
            ObjectiveManager(
                objectives={"dlm": dlm},
                weights={"dlm": 1.0},
            )

    def test_dlm_with_sft_succeeds(self):
        """Test that DLM with SFT works correctly."""
        from wf_train.objectives.manager import ObjectiveManager
        from wf_train.objectives.sft import SFTObjective

        dlm = DLMObjective(mask_token_id=999)
        sft = SFTObjective()

        # Should not raise
        manager = ObjectiveManager(
            objectives={"dlm": dlm, "sft": sft},
            weights={"dlm": 0.5, "sft": 1.0},
        )

        assert "dlm" in manager.objectives
        assert "sft" in manager.objectives

    def test_dlm_ce_overlap_raises_error(self):
        """Test that DLM + CE active together raises an error."""
        from wf_train.objectives.manager import (
            ObjectiveManager,
            CurriculumScheduler,
            CurriculumPhase,
        )
        from wf_train.objectives.sft import SFTObjective
        from wf_train.objectives.continue_pretrain import ContinuePretrainObjective

        dlm = DLMObjective(mask_token_id=999)
        sft = SFTObjective()
        ce = ContinuePretrainObjective()

        # Curriculum with DLM + CE active together (invalid)
        phases = [
            CurriculumPhase(
                name="bad_phase",
                end_ratio=1.0,
                objective_weights={"dlm": 0.5, "continue_pretrain": 1.0, "sft": 1.0},
            )
        ]
        curriculum = CurriculumScheduler(phases=phases, total_steps=100)

        with pytest.raises(ValueError, match="DLM and continue_pretrain cannot be active together"):
            ObjectiveManager(
                objectives={"dlm": dlm, "sft": sft, "continue_pretrain": ce},
                weights={"dlm": 0.5, "sft": 1.0, "continue_pretrain": 1.0},
                curriculum=curriculum,
            )

    def test_dlm_after_ce_phase_succeeds(self):
        """Test that CE phase followed by SFT+DLM phase works (pretrain_then_sft pattern)."""
        from wf_train.objectives.manager import (
            ObjectiveManager,
            CurriculumScheduler,
            CurriculumPhase,
        )
        from wf_train.objectives.sft import SFTObjective
        from wf_train.objectives.continue_pretrain import ContinuePretrainObjective

        dlm = DLMObjective(mask_token_id=999)
        sft = SFTObjective()
        ce = ContinuePretrainObjective()

        # Curriculum with CE first, then SFT+DLM (valid)
        phases = [
            CurriculumPhase(
                name="pretrain",
                end_ratio=0.9,
                objective_weights={"continue_pretrain": 1.0, "dlm": 0.0, "sft": 0.0},
            ),
            CurriculumPhase(
                name="sft_dlm",
                end_ratio=1.0,
                objective_weights={"continue_pretrain": 0.0, "dlm": 0.5, "sft": 1.0},
            ),
        ]
        curriculum = CurriculumScheduler(phases=phases, total_steps=100)

        # Should not raise - DLM and CE are never active together
        manager = ObjectiveManager(
            objectives={"dlm": dlm, "sft": sft, "continue_pretrain": ce},
            weights={"dlm": 0.5, "sft": 1.0, "continue_pretrain": 1.0},
            curriculum=curriculum,
        )

        assert "dlm" in manager.objectives
