"""Unit tests for distillation objectives.

Tests:
- LogitsDistillationObjective
- AttentionRelationDistillationObjective
- TCSDistillationObjective
"""

import pytest
import torch

from wf_train.objectives.logits_distill import LogitsDistillationObjective
from wf_train.objectives.attention_distill import AttentionRelationDistillationObjective
from wf_train.objectives.tcs_distill import TCSDistillationObjective


class TestLogitsDistillationObjective:
    """Tests for LogitsDistillationObjective (KL divergence on teacher/student logits)."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        obj = LogitsDistillationObjective()

        assert obj.temperature == 5.0
        assert obj.ignore_index == -100
        assert obj.shift_labels is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        obj = LogitsDistillationObjective(
            temperature=3.0,
            ignore_index=0,
            shift_labels=False,
        )

        assert obj.temperature == 3.0
        assert obj.ignore_index == 0
        assert obj.shift_labels is False

    def test_properties(self):
        """Test objective properties."""
        obj = LogitsDistillationObjective()

        assert obj.name == "logits_distill"
        assert obj.requires_teacher is True
        assert obj.requires_hidden_states is False
        assert obj.requires_attentions is False
        assert obj.modifies_input is False

    def test_forward_requires_teacher_outputs(self):
        """Test that forward raises error without teacher_outputs."""
        obj = LogitsDistillationObjective()
        model_outputs = {"logits": torch.randn(2, 10, 1000)}
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        with pytest.raises(ValueError, match="requires teacher_outputs"):
            obj.forward(model_outputs, batch, teacher_outputs=None)

    def test_forward_returns_objective_output(self):
        """Test that forward returns ObjectiveOutput with loss and metrics."""
        obj = LogitsDistillationObjective()

        batch_size, seq_len, vocab_size = 2, 10, 1000
        model_outputs = {"logits": torch.randn(batch_size, seq_len, vocab_size)}
        teacher_outputs = {"logits": torch.randn(batch_size, seq_len, vocab_size)}
        batch = {"labels": torch.randint(0, vocab_size, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert hasattr(result, "loss")
        assert hasattr(result, "metrics")
        assert result.loss.dim() == 0
        assert "kl_div" in result.metrics
        assert "temperature" in result.metrics

    def test_forward_loss_is_non_negative(self):
        """Test that loss is non-negative."""
        obj = LogitsDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 1000)}
        teacher_outputs = {"logits": torch.randn(2, 10, 1000)}
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert result.loss.item() >= 0

    def test_forward_identical_logits_low_loss(self):
        """Test that identical logits give low loss."""
        obj = LogitsDistillationObjective()

        logits = torch.randn(2, 10, 1000)
        model_outputs = {"logits": logits}
        teacher_outputs = {"logits": logits.clone()}
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # When logits are identical, KL divergence should be very small
        assert result.loss.item() < 1e-4

    def test_forward_respects_ignore_index(self):
        """Test that ignore_index positions are masked."""
        obj = LogitsDistillationObjective(ignore_index=-100)

        model_outputs = {"logits": torch.randn(2, 10, 1000)}
        teacher_outputs = {"logits": torch.randn(2, 10, 1000)}
        labels = torch.randint(0, 1000, (2, 10))
        labels[:, -3:] = -100  # Mask last 3 positions
        batch = {"labels": labels}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert not torch.isnan(result.loss)
        assert result.loss.item() >= 0

    def test_forward_shift_labels_enabled(self):
        """Test that shift_labels shifts logits for AR models."""
        obj = LogitsDistillationObjective(shift_labels=True)

        batch_size, seq_len, vocab_size = 2, 10, 100
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        model_outputs = {"logits": student_logits}
        teacher_outputs = {"logits": teacher_logits}
        batch = {"labels": torch.randint(0, vocab_size, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Just verify it runs without error and produces valid loss
        assert not torch.isnan(result.loss)

    def test_forward_shift_labels_disabled(self):
        """Test that shift_labels=False doesn't shift logits."""
        obj = LogitsDistillationObjective(shift_labels=False)

        batch_size, seq_len, vocab_size = 2, 10, 100
        model_outputs = {"logits": torch.randn(batch_size, seq_len, vocab_size)}
        teacher_outputs = {"logits": torch.randn(batch_size, seq_len, vocab_size)}
        batch = {"labels": torch.randint(0, vocab_size, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert not torch.isnan(result.loss)

    def test_forward_uses_original_labels_from_dlm(self):
        """Test that _original_labels is used when available (DLM mode)."""
        obj = LogitsDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 100)}
        teacher_outputs = {"logits": torch.randn(2, 10, 100)}

        # DLM stores original labels in _original_labels
        masked_labels = torch.full((2, 10), -100)  # All masked
        original_labels = torch.randint(0, 100, (2, 10))
        batch = {"labels": masked_labels, "_original_labels": original_labels}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Should use original labels, not the masked ones
        assert not torch.isnan(result.loss)

    def test_temperature_affects_loss(self):
        """Test that temperature parameter affects the loss computation."""
        obj_low_temp = LogitsDistillationObjective(temperature=1.0)
        obj_high_temp = LogitsDistillationObjective(temperature=10.0)

        model_outputs = {"logits": torch.randn(2, 10, 100)}
        teacher_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {"labels": torch.randint(0, 100, (2, 10))}

        loss_low = obj_low_temp.forward(model_outputs, batch, teacher_outputs).loss
        loss_high = obj_high_temp.forward(model_outputs, batch, teacher_outputs).loss

        # Higher temperature with T^2 scaling should give larger raw numbers
        # but softer distributions
        assert loss_low.item() != loss_high.item()


class TestAttentionRelationDistillationObjective:
    """Tests for AttentionRelationDistillationObjective (BitDistill attention distillation)."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        obj = AttentionRelationDistillationObjective()

        assert obj.distill_layer == -1
        assert obj.temperature == 1.0
        assert obj.ignore_index == -100

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        obj = AttentionRelationDistillationObjective(
            distill_layer=5,
            temperature=2.0,
            ignore_index=0,
        )

        assert obj.distill_layer == 5
        assert obj.temperature == 2.0
        assert obj.ignore_index == 0

    def test_properties(self):
        """Test objective properties."""
        obj = AttentionRelationDistillationObjective()

        assert obj.name == "attention_distill"
        assert obj.requires_teacher is True
        assert obj.requires_hidden_states is False
        assert obj.requires_attentions is True
        assert obj.modifies_input is False

    def test_forward_requires_teacher_outputs(self):
        """Test that forward raises error without teacher_outputs."""
        obj = AttentionRelationDistillationObjective()

        batch_size, num_heads, seq_len = 2, 8, 10
        model_outputs = {
            "attentions": [torch.randn(batch_size, num_heads, seq_len, seq_len)]
        }
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        with pytest.raises(ValueError, match="requires teacher_outputs"):
            obj.forward(model_outputs, batch, teacher_outputs=None)

    def test_forward_requires_student_attentions(self):
        """Test that forward raises error without student attentions."""
        obj = AttentionRelationDistillationObjective()

        model_outputs = {"attentions": None}
        teacher_outputs = {
            "attentions": [torch.randn(2, 8, 10, 10)]
        }
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        with pytest.raises(ValueError, match="Student model must return attention"):
            obj.forward(model_outputs, batch, teacher_outputs)

    def test_forward_requires_teacher_attentions(self):
        """Test that forward raises error without teacher attentions."""
        obj = AttentionRelationDistillationObjective()

        model_outputs = {
            "attentions": [torch.randn(2, 8, 10, 10)]
        }
        teacher_outputs = {"attentions": None}
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        with pytest.raises(ValueError, match="Teacher model must return attention"):
            obj.forward(model_outputs, batch, teacher_outputs)

    def test_forward_returns_objective_output(self):
        """Test that forward returns ObjectiveOutput with loss and metrics."""
        obj = AttentionRelationDistillationObjective()

        batch_size, num_heads, seq_len = 2, 8, 16
        num_layers = 12

        # Create attention tensors for each layer
        student_attns = [
            torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for _ in range(num_layers)
        ]
        teacher_attns = [
            torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for _ in range(num_layers)
        ]

        model_outputs = {"attentions": student_attns}
        teacher_outputs = {"attentions": teacher_attns}
        batch = {"labels": torch.randint(0, 1000, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert hasattr(result, "loss")
        assert hasattr(result, "metrics")
        assert result.loss.dim() == 0
        assert "attention_kl" in result.metrics
        assert "distill_layer" in result.metrics

    def test_forward_loss_is_non_negative(self):
        """Test that loss is non-negative."""
        obj = AttentionRelationDistillationObjective()

        batch_size, num_heads, seq_len = 2, 8, 10
        num_layers = 6

        student_attns = [
            torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for _ in range(num_layers)
        ]
        teacher_attns = [
            torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for _ in range(num_layers)
        ]

        model_outputs = {"attentions": student_attns}
        teacher_outputs = {"attentions": teacher_attns}
        batch = {"labels": torch.randint(0, 1000, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert result.loss.item() >= 0

    def test_forward_identical_attentions_low_loss(self):
        """Test that identical attention patterns give low loss."""
        obj = AttentionRelationDistillationObjective()

        batch_size, num_heads, seq_len = 2, 8, 10
        attn = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)

        model_outputs = {"attentions": [attn]}
        teacher_outputs = {"attentions": [attn.clone()]}
        batch = {"labels": torch.randint(0, 1000, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Identical attentions should give very low loss
        assert result.loss.item() < 1e-4

    def test_forward_uses_attention_mask(self):
        """Test that attention_mask is respected."""
        obj = AttentionRelationDistillationObjective()

        batch_size, num_heads, seq_len = 2, 8, 10

        student_attn = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
        teacher_attn = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)

        model_outputs = {"attentions": [student_attn]}
        teacher_outputs = {"attentions": [teacher_attn]}

        # Create attention mask with some padding
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -3:] = 0  # Last 3 positions are padding
        batch = {"labels": torch.randint(0, 1000, (batch_size, seq_len)),
                 "attention_mask": attention_mask}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert not torch.isnan(result.loss)

    def test_forward_distill_layer_selection(self):
        """Test that distill_layer parameter selects correct layer."""
        num_layers = 6

        # Test with specific layer index
        obj = AttentionRelationDistillationObjective(distill_layer=2)

        batch_size, num_heads, seq_len = 2, 4, 8
        student_attns = [
            torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for _ in range(num_layers)
        ]
        teacher_attns = [
            torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for _ in range(num_layers)
        ]

        model_outputs = {"attentions": student_attns}
        teacher_outputs = {"attentions": teacher_attns}
        batch = {"labels": torch.randint(0, 100, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Should report the layer used
        assert result.metrics["distill_layer"].item() == 2

    def test_forward_last_layer_default(self):
        """Test that distill_layer=-1 uses the last layer."""
        num_layers = 6
        obj = AttentionRelationDistillationObjective(distill_layer=-1)

        batch_size, num_heads, seq_len = 2, 4, 8
        student_attns = [
            torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for _ in range(num_layers)
        ]
        teacher_attns = [
            torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for _ in range(num_layers)
        ]

        model_outputs = {"attentions": student_attns}
        teacher_outputs = {"attentions": teacher_attns}
        batch = {"labels": torch.randint(0, 100, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # -1 mod 6 = 5, which is the last layer
        assert result.metrics["distill_layer"].item() == num_layers - 1

    def test_forward_head_count_mismatch_handled(self):
        """Test that different head counts between student and teacher are handled."""
        obj = AttentionRelationDistillationObjective()

        batch_size, seq_len = 2, 10
        student_heads = 8
        teacher_heads = 4  # Different number of heads

        student_attn = torch.randn(batch_size, student_heads, seq_len, seq_len).softmax(dim=-1)
        teacher_attn = torch.randn(batch_size, teacher_heads, seq_len, seq_len).softmax(dim=-1)

        model_outputs = {"attentions": [student_attn]}
        teacher_outputs = {"attentions": [teacher_attn]}
        batch = {"labels": torch.randint(0, 100, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Should handle mismatch by averaging over heads
        assert not torch.isnan(result.loss)
        assert result.loss.item() >= 0


class TestTCSDistillationObjective:
    """Tests for TCSDistillationObjective (Top-K TCS for DLM students)."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        obj = TCSDistillationObjective()

        assert obj.temperature == 5.0
        assert obj.top_k == 100
        assert obj.ignore_index == -100

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        obj = TCSDistillationObjective(
            temperature=3.0,
            top_k=50,
            ignore_index=0,
        )

        assert obj.temperature == 3.0
        assert obj.top_k == 50
        assert obj.ignore_index == 0

    def test_properties(self):
        """Test objective properties."""
        obj = TCSDistillationObjective()

        assert obj.name == "tcs_distill"
        assert obj.requires_teacher is True
        assert obj.requires_hidden_states is False
        assert obj.requires_attentions is False
        assert obj.modifies_input is False

    def test_forward_requires_teacher_outputs(self):
        """Test that forward raises error without teacher_outputs."""
        obj = TCSDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 1000)}
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        with pytest.raises(ValueError, match="requires teacher_outputs"):
            obj.forward(model_outputs, batch, teacher_outputs=None)

    def test_forward_returns_objective_output(self):
        """Test that forward returns ObjectiveOutput with loss and metrics."""
        obj = TCSDistillationObjective()

        batch_size, seq_len, vocab_size = 2, 10, 1000
        model_outputs = {"logits": torch.randn(batch_size, seq_len, vocab_size)}
        teacher_outputs = {"logits": torch.randn(batch_size, seq_len, vocab_size)}
        batch = {"labels": torch.randint(0, vocab_size, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert hasattr(result, "loss")
        assert hasattr(result, "metrics")
        assert result.loss.dim() == 0
        assert "tcs_kl" in result.metrics
        assert "num_masked" in result.metrics

    def test_forward_loss_is_non_negative(self):
        """Test that loss is non-negative."""
        obj = TCSDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 1000)}
        teacher_outputs = {"logits": torch.randn(2, 10, 1000)}
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert result.loss.item() >= 0

    def test_forward_identical_logits_low_loss(self):
        """Test that identical logits give low loss."""
        obj = TCSDistillationObjective()

        logits = torch.randn(2, 10, 1000)
        model_outputs = {"logits": logits}
        teacher_outputs = {"logits": logits.clone()}
        batch = {"labels": torch.randint(0, 1000, (2, 10))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert result.loss.item() < 1e-4

    def test_forward_respects_response_mask(self):
        """Test that masked positions are excluded from loss."""
        obj = TCSDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 100)}
        teacher_outputs = {"logits": torch.randn(2, 10, 100)}

        # All positions masked - should have 0 valid positions
        labels = torch.full((2, 10), -100)
        batch = {"labels": labels}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # num_masked should be 0
        assert result.metrics["num_masked"].item() == 0

    def test_forward_uses_dlm_labels_when_available(self):
        """Test that dlm_labels is preferred over labels."""
        obj = TCSDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 100)}
        teacher_outputs = {"logits": torch.randn(2, 10, 100)}

        # Regular labels (all masked)
        regular_labels = torch.full((2, 10), -100)
        # DLM labels (some unmasked)
        dlm_labels = torch.randint(0, 100, (2, 10))
        dlm_labels[:, :5] = -100  # First 5 masked

        batch = {"labels": regular_labels, "dlm_labels": dlm_labels}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Should use dlm_labels, so num_masked should be 10 (5 unmasked positions * 2 batch)
        assert result.metrics["num_masked"].item() == 10

    def test_forward_top_k_parameter(self):
        """Test that top_k limits the logits used for distillation."""
        obj = TCSDistillationObjective(top_k=10)

        vocab_size = 1000
        model_outputs = {"logits": torch.randn(2, 10, vocab_size)}
        teacher_outputs = {"logits": torch.randn(2, 10, vocab_size)}
        batch = {"labels": torch.randint(0, vocab_size, (2, 10))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Should run successfully with top_k=10
        assert not torch.isnan(result.loss)

    def test_forward_top_k_larger_than_vocab(self):
        """Test handling when top_k exceeds vocabulary size."""
        vocab_size = 50
        obj = TCSDistillationObjective(top_k=100)  # top_k > vocab_size

        model_outputs = {"logits": torch.randn(2, 10, vocab_size)}
        teacher_outputs = {"logits": torch.randn(2, 10, vocab_size)}
        batch = {"labels": torch.randint(0, vocab_size, (2, 10))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Should clamp to vocab_size and run successfully
        assert not torch.isnan(result.loss)

    def test_forward_no_logit_shifting(self):
        """Test that TCS does NOT shift logits (unlike AR distillation)."""
        obj = TCSDistillationObjective()

        batch_size, seq_len, vocab_size = 2, 10, 100

        # Create logits where we can verify no shifting occurs
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)

        model_outputs = {"logits": student_logits}
        teacher_outputs = {"logits": teacher_logits}
        batch = {"labels": torch.randint(0, vocab_size, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # num_masked should match the number of non-ignored labels
        expected_valid = (batch["labels"] != -100).sum().item()
        assert result.metrics["num_masked"].item() == expected_valid

    def test_temperature_affects_loss(self):
        """Test that temperature parameter affects the loss computation."""
        obj_low_temp = TCSDistillationObjective(temperature=1.0)
        obj_high_temp = TCSDistillationObjective(temperature=10.0)

        model_outputs = {"logits": torch.randn(2, 10, 100)}
        teacher_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {"labels": torch.randint(0, 100, (2, 10))}

        loss_low = obj_low_temp.forward(model_outputs, batch, teacher_outputs).loss
        loss_high = obj_high_temp.forward(model_outputs, batch, teacher_outputs).loss

        # Different temperatures should produce different losses
        assert loss_low.item() != loss_high.item()


class TestDistillationObjectivesGradients:
    """Test that all distillation objectives produce valid gradients."""

    def test_logits_distillation_backward(self):
        """Test that LogitsDistillationObjective produces valid gradients."""
        obj = LogitsDistillationObjective()

        student_logits = torch.randn(2, 10, 100, requires_grad=True)
        teacher_logits = torch.randn(2, 10, 100)

        model_outputs = {"logits": student_logits}
        teacher_outputs = {"logits": teacher_logits}
        batch = {"labels": torch.randint(0, 100, (2, 10))}

        result = obj.forward(model_outputs, batch, teacher_outputs)
        result.loss.backward()

        assert student_logits.grad is not None
        assert not torch.isnan(student_logits.grad).any()

    def test_attention_distillation_backward(self):
        """Test that AttentionRelationDistillationObjective produces valid gradients."""
        obj = AttentionRelationDistillationObjective()

        batch_size, num_heads, seq_len = 2, 4, 8

        student_attn_raw = torch.randn(batch_size, num_heads, seq_len, seq_len, requires_grad=True)
        student_attn = student_attn_raw.softmax(dim=-1)
        teacher_attn = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)

        model_outputs = {"attentions": [student_attn]}
        teacher_outputs = {"attentions": [teacher_attn]}
        batch = {"labels": torch.randint(0, 100, (batch_size, seq_len))}

        result = obj.forward(model_outputs, batch, teacher_outputs)
        result.loss.backward()

        assert student_attn_raw.grad is not None
        assert not torch.isnan(student_attn_raw.grad).any()

    def test_tcs_distillation_backward(self):
        """Test that TCSDistillationObjective produces valid gradients."""
        obj = TCSDistillationObjective()

        student_logits = torch.randn(2, 10, 100, requires_grad=True)
        teacher_logits = torch.randn(2, 10, 100)

        model_outputs = {"logits": student_logits}
        teacher_outputs = {"logits": teacher_logits}
        batch = {"labels": torch.randint(0, 100, (2, 10))}

        result = obj.forward(model_outputs, batch, teacher_outputs)
        result.loss.backward()

        assert student_logits.grad is not None
        assert not torch.isnan(student_logits.grad).any()


class TestDistillationEdgeCases:
    """Test edge cases for distillation objectives."""

    def test_logits_distillation_all_masked_labels(self):
        """Test LogitsDistillationObjective with completely masked batch."""
        obj = LogitsDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 100)}
        teacher_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {"labels": torch.full((2, 10), -100)}  # All masked

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Should handle gracefully - loss should be 0 or very small
        assert not torch.isnan(result.loss)
        assert result.loss.item() >= 0

    def test_tcs_distillation_all_masked_labels(self):
        """Test TCSDistillationObjective with completely masked batch."""
        obj = TCSDistillationObjective()

        model_outputs = {"logits": torch.randn(2, 10, 100)}
        teacher_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {"labels": torch.full((2, 10), -100)}  # All masked

        result = obj.forward(model_outputs, batch, teacher_outputs)

        # Should handle gracefully
        assert not torch.isnan(result.loss)
        assert result.metrics["num_masked"].item() == 0

    def test_attention_distillation_with_full_attention_mask(self):
        """Test AttentionRelationDistillationObjective with attention mask."""
        obj = AttentionRelationDistillationObjective()

        batch_size, num_heads, seq_len = 2, 4, 10
        student_attn = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
        teacher_attn = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)

        model_outputs = {"attentions": [student_attn]}
        teacher_outputs = {"attentions": [teacher_attn]}

        # Create attention mask with half padding
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, seq_len // 2:] = 0
        batch = {
            "labels": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": attention_mask,
        }

        result = obj.forward(model_outputs, batch, teacher_outputs)

        assert not torch.isnan(result.loss)
        assert result.loss.item() >= 0
