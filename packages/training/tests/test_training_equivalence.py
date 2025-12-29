"""Tests for training code equivalence across optimizations.

These tests verify that optimized training code produces functionally
equivalent outputs to baseline implementations, ensuring optimizations
don't break training behavior.
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from wrinklefree.models.bitlinear import BitLinear
from wrinklefree.models.subln import SubLN
from wrinklefree.testing.equivalence import (
    EquivalenceResult,
    compare_gradients,
    compare_hidden_states,
    compare_logits_cosine,
    assert_models_equivalent,
    create_equivalence_checkpoint,
    compare_to_checkpoint,
    cosine_similarity,
)


class TestCosineSimility:
    """Test the cosine similarity utility function."""

    def test_identical_tensors(self):
        """Identical tensors should have cosine similarity of 1.0."""
        a = torch.randn(10, 20)
        b = a.clone()
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_opposite_tensors(self):
        """Negated tensors should have cosine similarity of -1.0."""
        a = torch.randn(10, 20)
        b = -a
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_tensors(self):
        """Orthogonal tensors should have cosine similarity near 0."""
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_zero_tensor(self):
        """Zero tensors should return 1.0 (both near-zero)."""
        a = torch.zeros(10)
        b = torch.zeros(10)
        assert cosine_similarity(a, b) == 1.0

    def test_one_zero_tensor(self):
        """One zero tensor should return 0.0."""
        a = torch.randn(10)
        b = torch.zeros(10)
        assert cosine_similarity(a, b) == 0.0


class _SimpleLogitModel(nn.Module):
    """Simple model that accepts input_ids kwarg for testing compare_logits_cosine."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.proj = nn.Linear(64, 100)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        x = self.embed(input_ids)
        logits = self.proj(x)
        return {"logits": logits}


class TestLogitsComparison:
    """Test logits comparison between models."""

    def test_same_model_identical_logits(self):
        """Same model should produce identical logits (cosine = 1.0)."""
        model = _SimpleLogitModel()

        input_ids = torch.randint(0, 100, (2, 16))

        # Compare model to itself
        similarity = compare_logits_cosine(model, model, input_ids)
        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_different_models_different_logits(self):
        """Different models should have different logits."""
        torch.manual_seed(42)
        model_a = _SimpleLogitModel()

        torch.manual_seed(123)
        model_b = _SimpleLogitModel()

        input_ids = torch.randint(0, 100, (2, 16))

        similarity = compare_logits_cosine(model_a, model_b, input_ids)
        # Different random init should give different outputs (not identical)
        assert similarity < 0.99


class TestGradientComparison:
    """Test gradient comparison between models."""

    def test_same_model_identical_gradients(self):
        """Same model architecture should produce identical gradients."""
        torch.manual_seed(42)

        # Create two identical models using HF-style interface
        model_a = _SimpleLogitModel()
        model_b = copy.deepcopy(model_a)

        input_ids = torch.randint(0, 100, (2, 16))
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        result = compare_gradients(model_a, model_b, input_ids, loss_fn)

        assert result.gradient_cosine_similarity == pytest.approx(1.0, abs=1e-5)
        assert result.passed

    def test_different_init_different_gradients(self):
        """Different initializations should produce different gradients."""
        torch.manual_seed(42)
        model_a = _SimpleLogitModel()

        torch.manual_seed(123)
        model_b = _SimpleLogitModel()

        input_ids = torch.randint(0, 100, (2, 16))
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        result = compare_gradients(model_a, model_b, input_ids, loss_fn)

        # Gradients should be different (but test still returns a result)
        assert result.gradient_cosine_similarity < 1.0


class TestBitLinearEquivalence:
    """Test BitLinear layer equivalence under different conditions."""

    def test_bitlinear_deterministic(self):
        """BitLinear should be deterministic (same input = same output)."""
        torch.manual_seed(42)
        layer = BitLinear(512, 256)

        x = torch.randn(2, 16, 512)

        out1 = layer(x)
        out2 = layer(x)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_bitlinear_with_subln_equivalence(self):
        """BitLinear with SubLN should produce consistent outputs."""
        torch.manual_seed(42)

        subln = SubLN(512)
        proj = BitLinear(512, 256)
        wrapped = nn.Sequential(subln, proj)

        x = torch.randn(2, 16, 512)

        # Manual application
        x_normed = subln(x)
        manual_output = proj(x_normed)

        # Sequential application
        seq_output = wrapped(x)

        assert torch.allclose(manual_output, seq_output, atol=1e-6)


class TestCheckpointComparison:
    """Test checkpoint-based comparison."""

    def test_checkpoint_same_model(self):
        """Checkpoint comparison should pass for same model."""
        model = _SimpleLogitModel()

        input_ids = torch.randint(0, 100, (2, 16))

        # Create checkpoint
        checkpoint = create_equivalence_checkpoint(model, input_ids)

        # Compare to same model
        result = compare_to_checkpoint(model, checkpoint)

        assert result.passed
        assert result.logit_cosine_similarity == pytest.approx(1.0, abs=1e-6)

    def test_checkpoint_modified_model(self):
        """Checkpoint comparison should detect model changes."""
        model = _SimpleLogitModel()

        input_ids = torch.randint(0, 100, (2, 16))

        # Create checkpoint
        checkpoint = create_equivalence_checkpoint(model, input_ids)

        # Modify model weights
        with torch.no_grad():
            model.proj.weight.add_(torch.randn_like(model.proj.weight) * 0.5)

        # Compare to modified model
        result = compare_to_checkpoint(model, checkpoint)

        # Should detect the difference
        assert result.logit_cosine_similarity < 1.0


class TestHiddenStatesComparison:
    """Test hidden states comparison."""

    def test_same_model_identical_hidden_states(self):
        """Same model should have identical hidden states."""
        # Use a simple model that returns hidden states
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.layer1 = nn.Linear(64, 64)
                self.layer2 = nn.Linear(64, 100)

            def forward(self, input_ids, output_hidden_states=False, **kwargs):
                x = self.embed(input_ids)
                h1 = self.layer1(x)
                h2 = self.layer2(F.relu(h1))

                if output_hidden_states:
                    return {"logits": h2, "hidden_states": [x, h1, h2]}
                return {"logits": h2}

        model = SimpleModel()
        input_ids = torch.randint(0, 100, (2, 16))

        result = compare_hidden_states(model, model, input_ids)

        for layer_name, similarity in result.items():
            assert similarity == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
class TestTorchCompileEquivalence:
    """Test that torch.compile doesn't change model behavior.

    These tests are marked slow because torch.compile has compilation overhead.
    """

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_bitlinear_compile_equivalence(self):
        """torch.compile on BitLinear should produce equivalent outputs."""
        torch.manual_seed(42)

        layer_eager = BitLinear(256, 128)
        layer_compiled = copy.deepcopy(layer_eager)
        layer_compiled = torch.compile(layer_compiled, mode="reduce-overhead")

        # Use CUDA for compiled model
        device = torch.device("cuda")
        layer_eager = layer_eager.to(device)
        layer_compiled = layer_compiled.to(device)

        x = torch.randn(2, 16, 256, device=device)

        # Warm up compiled model
        for _ in range(3):
            _ = layer_compiled(x)

        out_eager = layer_eager(x)
        out_compiled = layer_compiled(x)

        similarity = cosine_similarity(out_eager, out_compiled)
        assert similarity > 0.999, f"Compiled output diverged: {similarity}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for torch.compile"
    )
    def test_model_compile_gradient_equivalence(self):
        """torch.compile should produce equivalent gradients."""
        torch.manual_seed(42)

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.proj = BitLinear(64, 100)

            def forward(self, input_ids, **kwargs):
                x = self.embed(input_ids)
                return {"logits": self.proj(x)}

        model_eager = SimpleModel()
        model_compiled = copy.deepcopy(model_eager)
        model_compiled = torch.compile(model_compiled, mode="reduce-overhead")

        device = torch.device("cuda")
        model_eager = model_eager.to(device)
        model_compiled = model_compiled.to(device)

        input_ids = torch.randint(0, 100, (2, 16), device=device)

        # Warm up compiled model
        for _ in range(3):
            out = model_compiled(input_ids)
            loss = out["logits"].sum()
            loss.backward()
            model_compiled.zero_grad()

        # Compare gradients
        loss_fn = nn.CrossEntropyLoss()
        result = compare_gradients(model_eager, model_compiled, input_ids, loss_fn)

        assert result.gradient_cosine_similarity > 0.99, (
            f"Compiled gradients diverged: {result.gradient_cosine_similarity}"
        )


class TestAssertModelsEquivalent:
    """Test the assertion helper function."""

    def test_equivalent_models_pass(self):
        """Equivalent models should pass assertion."""
        model = _SimpleLogitModel()

        input_ids = torch.randint(0, 100, (2, 16))

        # Should not raise
        assert_models_equivalent(model, model, input_ids)

    def test_different_models_fail(self):
        """Different models should fail assertion."""
        torch.manual_seed(42)
        model_a = _SimpleLogitModel()

        torch.manual_seed(123)
        model_b = _SimpleLogitModel()

        input_ids = torch.randint(0, 100, (2, 16))

        with pytest.raises(AssertionError, match="Models not equivalent"):
            assert_models_equivalent(model_a, model_b, input_ids)


@pytest.mark.slow
class TestSmolLM2Equivalence:
    """Test equivalence on actual SmolLM2-135M model.

    These tests download and run the actual model, so they're marked slow.
    """

    @pytest.fixture
    def smollm2_model(self):
        """Load SmolLM2-135M model."""
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M",
            torch_dtype=torch.float32,
        )
        return model

    @pytest.fixture
    def smollm2_tokenizer(self):
        """Load SmolLM2-135M tokenizer."""
        return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    def test_smollm2_deterministic(self, smollm2_model, smollm2_tokenizer):
        """SmolLM2 should produce deterministic outputs."""
        smollm2_model.eval()

        text = "Hello, world!"
        inputs = smollm2_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            out1 = smollm2_model(**inputs)
            out2 = smollm2_model(**inputs)

        assert torch.allclose(out1.logits, out2.logits, atol=1e-6)

    def test_smollm2_copy_equivalence(self, smollm2_model, smollm2_tokenizer):
        """Copied model should produce identical outputs."""
        model_copy = copy.deepcopy(smollm2_model)

        text = "The quick brown fox"
        inputs = smollm2_tokenizer(text, return_tensors="pt")

        similarity = compare_logits_cosine(
            smollm2_model, model_copy, inputs["input_ids"]
        )
        # Relaxed tolerance to account for floating point differences in deep copies
        assert similarity == pytest.approx(1.0, abs=1e-4)
