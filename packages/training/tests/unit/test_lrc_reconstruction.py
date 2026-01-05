"""Tests for LRCReconstructionObjective."""

import pytest
import torch

from wf_train.objectives.lrc_reconstruction import (
    LRCReconstructionObjective,
    LRCLossType,
)


class TestLRCReconstructionObjective:
    """Test LRCReconstructionObjective."""

    def test_init(self):
        """Test initialization."""
        obj = LRCReconstructionObjective()
        assert obj.name == "lrc_reconstruction"
        assert obj.requires_teacher
        assert obj.requires_hidden_states
        assert not obj.modifies_input
        assert not obj.requires_attentions

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        obj = LRCReconstructionObjective(
            loss_type="cosine",
            layer_weights="progressive",
            temperature=2.0,
            normalize=True,
        )
        assert obj.loss_type == LRCLossType.COSINE
        assert obj.layer_weights_config == "progressive"
        assert obj.temperature == 2.0
        assert obj.normalize

    def test_requires_teacher_outputs(self):
        """Test that it raises without teacher outputs."""
        obj = LRCReconstructionObjective()

        model_outputs = {
            "logits": torch.randn(2, 10, 100),
            "hidden_states": (torch.randn(2, 10, 64),),
        }
        batch = {}

        with pytest.raises(ValueError, match="requires teacher_outputs"):
            obj(model_outputs, batch)

    def test_requires_hidden_states(self):
        """Test that it raises without hidden states."""
        obj = LRCReconstructionObjective()

        model_outputs = {"logits": torch.randn(2, 10, 100)}
        teacher_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {}

        with pytest.raises(ValueError, match="must output hidden_states"):
            obj(model_outputs, batch, teacher_outputs)

    def test_forward_basic(self):
        """Test basic forward pass."""
        obj = LRCReconstructionObjective()

        batch_size, seq_len, hidden_size = 2, 10, 64
        num_layers = 4

        # Create hidden states (including embedding layer at index 0)
        student_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )
        teacher_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )

        model_outputs = {
            "logits": torch.randn(2, 10, 100),
            "hidden_states": student_hidden,
        }
        teacher_outputs = {"hidden_states": teacher_hidden}
        batch = {"attention_mask": torch.ones(batch_size, seq_len)}

        output = obj(model_outputs, batch, teacher_outputs)

        assert output.loss.shape == ()
        assert output.loss.item() >= 0
        assert "mean_layer_loss" in output.metrics
        assert "num_layers" in output.metrics
        assert output.metrics["num_layers"].item() == num_layers

    def test_identical_outputs_zero_loss(self):
        """Test that identical hidden states give near-zero loss."""
        obj = LRCReconstructionObjective(loss_type="mse")

        batch_size, seq_len, hidden_size = 2, 8, 64
        num_layers = 3

        # Use identical hidden states for student and teacher
        hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )

        model_outputs = {"logits": torch.randn(2, 8, 100), "hidden_states": hidden}
        teacher_outputs = {"hidden_states": hidden}
        batch = {"attention_mask": torch.ones(batch_size, seq_len)}

        output = obj(model_outputs, batch, teacher_outputs)

        assert output.loss.item() < 1e-6

    def test_different_outputs_positive_loss(self):
        """Test that different hidden states give positive loss."""
        obj = LRCReconstructionObjective(loss_type="mse")

        batch_size, seq_len, hidden_size = 2, 8, 64
        num_layers = 3

        student_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )
        teacher_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )

        model_outputs = {"logits": torch.randn(2, 8, 100), "hidden_states": student_hidden}
        teacher_outputs = {"hidden_states": teacher_hidden}
        batch = {"attention_mask": torch.ones(batch_size, seq_len)}

        output = obj(model_outputs, batch, teacher_outputs)

        assert output.loss.item() > 0

    def test_layer_mismatch_raises(self):
        """Test that layer count mismatch raises error."""
        obj = LRCReconstructionObjective()

        student_hidden = tuple(torch.randn(2, 8, 64) for _ in range(5))
        teacher_hidden = tuple(torch.randn(2, 8, 64) for _ in range(4))  # Different count

        model_outputs = {"logits": torch.randn(2, 8, 100), "hidden_states": student_hidden}
        teacher_outputs = {"hidden_states": teacher_hidden}
        batch = {}

        with pytest.raises(ValueError, match="Layer count mismatch"):
            obj(model_outputs, batch, teacher_outputs)

    def test_layer_weights_progressive(self):
        """Test progressive layer weights."""
        obj = LRCReconstructionObjective(layer_weights="progressive")

        weights = obj._get_layer_weights(4)

        # Progressive weights should increase
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i + 1]
        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_layer_weights_exponential(self):
        """Test exponential layer weights."""
        obj = LRCReconstructionObjective(layer_weights="exponential")

        weights = obj._get_layer_weights(4)

        # Exponential weights should increase faster
        assert weights[-1] > weights[0] * 4
        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_layer_weights_uniform(self):
        """Test uniform layer weights (default)."""
        obj = LRCReconstructionObjective(layer_weights=None)

        weights = obj._get_layer_weights(4)

        # All weights should be equal
        assert all(abs(w - 0.25) < 1e-6 for w in weights)
        # Should sum to 1
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_layer_weights_custom(self):
        """Test custom layer weights."""
        obj = LRCReconstructionObjective(layer_weights=[1.0, 2.0, 3.0, 4.0])

        weights = obj._get_layer_weights(4)

        # Should be normalized
        expected = [0.1, 0.2, 0.3, 0.4]
        for w, e in zip(weights, expected):
            assert abs(w - e) < 1e-6

    def test_attention_mask_applied(self):
        """Test that attention mask is properly applied."""
        obj = LRCReconstructionObjective(loss_type="mse")

        batch_size, seq_len, hidden_size = 2, 8, 64
        num_layers = 2

        student_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )
        teacher_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )

        model_outputs = {"logits": torch.randn(2, 8, 100), "hidden_states": student_hidden}
        teacher_outputs = {"hidden_states": teacher_hidden}

        # Full attention mask
        batch_full = {"attention_mask": torch.ones(batch_size, seq_len)}
        output_full = obj(model_outputs, batch_full, teacher_outputs)

        # Half attention mask (only first 4 tokens)
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[:, :4] = 1
        batch_half = {"attention_mask": attention_mask}
        output_half = obj(model_outputs, batch_half, teacher_outputs)

        # Losses should be different
        assert output_full.loss.item() != output_half.loss.item()


class TestLRCLossTypes:
    """Test different loss types."""

    def test_mse_loss(self):
        """Test MSE loss computation."""
        obj = LRCReconstructionObjective(loss_type="mse")

        student_h = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        teacher_h = torch.tensor([[[1.1, 2.1], [3.1, 4.1]]])

        loss = obj._mse_loss(student_h, teacher_h, None, normalize=False)

        # MSE = mean((1-1.1)^2 + (2-2.1)^2 + (3-3.1)^2 + (4-4.1)^2) / 2 = 0.01
        expected = torch.tensor(0.01)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_mse_normalized_loss(self):
        """Test MSE normalized loss."""
        obj = LRCReconstructionObjective(loss_type="mse_normalized")

        batch_size, seq_len, hidden_size = 2, 10, 64

        # Identical hidden states (after normalization) should have zero loss
        student_h = torch.randn(batch_size, seq_len, hidden_size)
        teacher_h = student_h.clone()

        loss = obj._mse_loss(student_h, teacher_h, None, normalize=True)
        assert loss.item() < 1e-6

    def test_cosine_loss(self):
        """Test cosine loss computation."""
        obj = LRCReconstructionObjective(loss_type="cosine")

        # Parallel vectors should have zero cosine loss
        student_h = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        teacher_h = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])  # Same direction, different magnitude

        loss = obj._cosine_loss(student_h, teacher_h, None)
        assert loss.item() < 1e-6

        # Orthogonal vectors should have loss of 1
        student_h = torch.tensor([[[1.0, 0.0]]])
        teacher_h = torch.tensor([[[0.0, 1.0]]])

        loss = obj._cosine_loss(student_h, teacher_h, None)
        assert abs(loss.item() - 1.0) < 1e-6

    def test_temperature_scaling(self):
        """Test temperature scaling."""
        batch_size, seq_len, hidden_size = 2, 8, 64
        num_layers = 2

        student_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )
        teacher_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )

        model_outputs = {"logits": torch.randn(2, 8, 100), "hidden_states": student_hidden}
        teacher_outputs = {"hidden_states": teacher_hidden}
        batch = {"attention_mask": torch.ones(batch_size, seq_len)}

        # Temperature = 1.0
        obj1 = LRCReconstructionObjective(temperature=1.0)
        output1 = obj1(model_outputs, batch, teacher_outputs)

        # Temperature = 2.0 (should halve the loss)
        obj2 = LRCReconstructionObjective(temperature=2.0)
        output2 = obj2(model_outputs, batch, teacher_outputs)

        assert torch.allclose(output1.loss / 2, output2.loss, atol=1e-5)


class TestGradientFlow:
    """Test gradient flow through the objective."""

    def test_gradients_flow_to_student(self):
        """Test that gradients flow through to student hidden states."""
        obj = LRCReconstructionObjective()

        batch_size, seq_len, hidden_size = 2, 8, 64
        num_layers = 2

        # Student hidden states with requires_grad
        student_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
            for _ in range(num_layers + 1)
        )
        teacher_hidden = tuple(
            torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)
        )

        model_outputs = {"logits": torch.randn(2, 8, 100), "hidden_states": student_hidden}
        teacher_outputs = {"hidden_states": teacher_hidden}
        batch = {"attention_mask": torch.ones(batch_size, seq_len)}

        output = obj(model_outputs, batch, teacher_outputs)
        output.loss.backward()

        # Gradients should flow to student hidden states (except embedding at index 0)
        for i, h in enumerate(student_hidden[1:]):
            assert h.grad is not None, f"No gradient at layer {i+1}"
