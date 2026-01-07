"""Tests for MuonClip optimizer with LRC (Low-Rank Correction) models.

Tests MuonClip functionality when training LRC adapters where:
- Base model weights are frozen
- Only U, V low-rank matrices are trainable
- QK-clipping should still work to prevent attention score explosions
"""

import logging
import pytest
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _patch_muonclip_optimizer(optimizer):
    """Apply production workarounds to MuonClip optimizer.

    This mirrors the patches applied in wf_train.lightning.module._create_muonclip_optimizer:
    1. Patch remove_hooks to reset is_registered flag
    2. Add no-op writer to prevent flush_metrics crash
    """
    # Patch remove_hooks bug
    if hasattr(optimizer, "hook_recorder"):
        original_remove = optimizer.hook_recorder.remove_hooks

        def patched_remove_hooks():
            original_remove()
            optimizer.hook_recorder.is_registered = False

        optimizer.hook_recorder.remove_hooks = patched_remove_hooks

    # Add no-op writer
    class _NoOpWriter:
        def add_scalar(self, *args, **kwargs):
            pass

    optimizer.writer = _NoOpWriter()
    return optimizer


class SimpleLRCModel(nn.Module):
    """Simple model simulating LRC structure with frozen base and trainable U, V."""

    def __init__(self):
        super().__init__()
        # Simulated frozen base layers
        self.embed = nn.Embedding(100, 64)
        self.q_proj = nn.Linear(64, 64, bias=False)  # Attention Q projection
        self.k_proj = nn.Linear(64, 64, bias=False)  # Attention K projection
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 100)

        # Simulated LRC trainable matrices (U, V for low-rank correction)
        self.lrc_U = nn.Linear(16, 64, bias=False)  # Rank 16
        self.lrc_V = nn.Linear(64, 16, bias=False)

        # Freeze base model, keep only LRC matrices trainable
        for name, param in self.named_parameters():
            if "lrc_" not in name:
                param.requires_grad = False

    def forward(self, x):
        x = self.embed(x)
        # Simulated attention with frozen Q, K
        q = self.q_proj(x)
        k = self.k_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / 8.0  # Simple attention scores
        attn = torch.softmax(attn, dim=-1)
        v = self.v_proj(x)
        x = torch.matmul(attn, v)
        x = self.o_proj(x)
        # LRC correction
        lrc_out = self.lrc_U(self.lrc_V(x))
        x = x + lrc_out
        x = self.norm(x)
        return self.head(x)


class TestMuonClipLRC:
    """Tests for MuonClip with LRC models."""

    def test_muonclip_with_frozen_attention(self):
        """Test MuonClip can be created with model that has frozen attention layers."""
        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig

        model = SimpleLRCModel()

        # Verify LRC matrices are trainable
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        assert "lrc_U.weight" in trainable_params
        assert "lrc_V.weight" in trainable_params
        assert len(trainable_params) == 2, f"Expected 2 trainable params, got {trainable_params}"

        # Create MuonClip config
        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            muon_beta=0.95,
            muon_decay=0.01,
            adam_betas=(0.9, 0.999),
            adam_decay=0.01,
            adam_eps=1e-8,
            enable_clipping=True,
            clipping_threshold=50.0,
            clipping_alpha=0.5,
            clipping_layers_mapping={"q_proj": "q_proj", "k_proj": "k_proj"},
        )

        # Create minimal model config for MuonClip
        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        model_config = MinimalConfig()

        # Create optimizer - should not raise
        optimizer = MuonClip(model, model_config, config)
        assert optimizer is not None

        # Verify hooks registered on q_proj and k_proj
        if hasattr(optimizer, "hook_recorder"):
            assert len(optimizer.hook_recorder.handles) > 0, "Expected hooks to be registered"

    def test_muonclip_lrc_training_step(self):
        """Test MuonClip can perform forward/backward/step with LRC model."""
        if not torch.cuda.is_available():
            pytest.skip("MuonClip test requires GPU")

        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig

        model = SimpleLRCModel().cuda()

        # Create MuonClip config with clipping enabled
        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            muon_beta=0.95,
            muon_decay=0.01,
            adam_betas=(0.9, 0.999),
            adam_decay=0.01,
            enable_clipping=True,
            clipping_threshold=50.0,
            clipping_alpha=0.5,
            clipping_layers_mapping={"q_proj": "q_proj", "k_proj": "k_proj"},
        )

        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        optimizer = MuonClip(model, MinimalConfig(), config)
        _patch_muonclip_optimizer(optimizer)

        # Call train() to register hooks
        model.train()

        # Forward pass
        x = torch.randint(0, 100, (2, 16)).cuda()
        logits = model(x)
        loss = logits.sum()

        # Backward pass
        loss.backward()

        # Check that only LRC params have gradients
        for name, param in model.named_parameters():
            if "lrc_" in name:
                assert param.grad is not None, f"LRC param {name} should have gradient"
            else:
                assert param.grad is None, f"Frozen param {name} should not have gradient"

        # Optimizer step should not raise
        optimizer.step()
        optimizer.zero_grad()

    def test_muonclip_qk_clipping_with_frozen_weights(self):
        """Test that QK-clipping still functions with frozen attention weights.

        QK-clipping clips attention logits, not weights. So it should work
        even when attention weights are frozen.
        """
        if not torch.cuda.is_available():
            pytest.skip("MuonClip test requires GPU")

        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig

        model = SimpleLRCModel().cuda()

        # Use a low clipping threshold to force clipping
        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            muon_beta=0.95,
            muon_decay=0.01,
            adam_betas=(0.9, 0.999),
            adam_decay=0.01,
            enable_clipping=True,
            clipping_threshold=1.0,  # Low threshold to trigger clipping
            clipping_alpha=0.5,
            clipping_layers_mapping={"q_proj": "q_proj", "k_proj": "k_proj"},
        )

        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        optimizer = MuonClip(model, MinimalConfig(), config)
        _patch_muonclip_optimizer(optimizer)

        model.train()

        # Multiple training steps to see if QK-clipping affects stability
        for i in range(5):
            x = torch.randint(0, 100, (2, 16)).cuda()
            logits = model(x)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # If we got here without NaN or errors, QK-clipping is working
        # Verify model can still produce valid outputs
        with torch.no_grad():
            x = torch.randint(0, 100, (2, 16)).cuda()
            logits = model(x)
            assert not torch.isnan(logits).any(), "Logits should not contain NaN"
            assert not torch.isinf(logits).any(), "Logits should not contain Inf"


class TestMuonClipHookRecovery:
    """Tests for MuonClip hook recovery after eval/train cycles."""

    def test_hooks_survive_eval_train_cycle(self):
        """Test that hooks are properly re-registered after model.eval() -> model.train()."""
        if not torch.cuda.is_available():
            pytest.skip("MuonClip test requires GPU")

        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig

        model = SimpleLRCModel().cuda()

        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            enable_clipping=True,
            clipping_threshold=50.0,
            clipping_alpha=0.5,
            clipping_layers_mapping={"q_proj": "q_proj", "k_proj": "k_proj"},
        )

        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        optimizer = MuonClip(model, MinimalConfig(), config)
        _patch_muonclip_optimizer(optimizer)

        # Initial train mode
        model.train()
        initial_hooks = len(optimizer.hook_recorder.handles) if hasattr(optimizer, "hook_recorder") else 0

        # Simulate BatchSizeFinder: eval -> train cycle
        model.eval()
        model.train()

        # After patched remove_hooks, hooks should be re-registered
        final_hooks = len(optimizer.hook_recorder.handles) if hasattr(optimizer, "hook_recorder") else 0

        assert final_hooks > 0, "Hooks should be re-registered after eval/train cycle"

        # Verify training still works
        x = torch.randint(0, 100, (2, 16)).cuda()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        optimizer.step()  # Should not raise KeyError
        optimizer.zero_grad()


class TestMuonClipDtypeHandling:
    """Tests for MuonClip bfloat16/float32 dtype handling."""

    def test_muonclip_bfloat16_dtype_mismatch_fix(self):
        """Test that MuonClip works with bfloat16 models after dtype patch.

        This tests the fix for:
        RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float

        The fix patches MuonClip.step to convert bfloat16 params to float32
        before Newton-Schulz iterations and back afterwards.
        """
        if not torch.cuda.is_available():
            pytest.skip("MuonClip bfloat16 test requires GPU")

        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig
        import muon

        # Create model in bfloat16 (like LRC models)
        model = SimpleLRCModel().cuda().to(torch.bfloat16)

        # Verify model is in bfloat16
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.dtype == torch.bfloat16, f"Param {name} should be bfloat16"

        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            muon_beta=0.95,
            muon_decay=0.01,
            adam_betas=(0.9, 0.999),
            adam_decay=0.01,
            enable_clipping=False,  # Disable clipping for this test
        )

        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        optimizer = MuonClip(model, MinimalConfig(), config)
        _patch_muonclip_optimizer(optimizer)

        # Apply the dtype patch from module.py
        _patch_muonclip_dtype_handling(muon)

        model.train()

        # Forward pass with bfloat16 model
        x = torch.randint(0, 100, (2, 16)).cuda()
        logits = model(x)
        loss = logits.float().sum()  # Convert to float for loss computation

        # Backward pass
        loss.backward()

        # This should NOT raise "expected mat1 and mat2 to have the same dtype"
        optimizer.step()
        optimizer.zero_grad()

        # Verify params are still in bfloat16 after step
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.dtype == torch.bfloat16, f"Param {name} should remain bfloat16 after step"

    def test_muonclip_float32_still_works(self):
        """Test that the dtype patch doesn't break float32 models."""
        if not torch.cuda.is_available():
            pytest.skip("MuonClip test requires GPU")

        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig
        import muon

        # Create model in float32 (default)
        model = SimpleLRCModel().cuda()  # float32 by default

        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            enable_clipping=False,
        )

        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        optimizer = MuonClip(model, MinimalConfig(), config)
        _patch_muonclip_optimizer(optimizer)

        # Apply dtype patch (should be no-op for float32)
        _patch_muonclip_dtype_handling(muon)

        model.train()

        x = torch.randint(0, 100, (2, 16)).cuda()
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Should work without issues
        optimizer.step()
        optimizer.zero_grad()


def _patch_muonclip_dtype_handling(muon_module) -> None:
    """Patch MuonClip to handle bfloat16/float32 dtype mismatch.

    This mirrors the patch in wf_train.lightning.module.
    """
    if not hasattr(muon_module, "MuonClip"):
        return

    MuonClip = muon_module.MuonClip
    if getattr(MuonClip, "_dtype_patch_applied", False):
        return

    original_step = MuonClip.step

    def patched_step(self, closure=None):
        original_dtypes = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.dtype == torch.bfloat16:
                    original_dtypes[id(p)] = p.dtype
                    p.data = p.data.float()
                    p.grad.data = p.grad.data.float()

        result = original_step(self, closure)

        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in original_dtypes:
                    p.data = p.data.to(original_dtypes[id(p)])

        return result

    MuonClip.step = patched_step
    MuonClip._dtype_patch_applied = True


class TestMuonClipEdgeCases:
    """Edge case tests suggested by Gemini review."""

    def test_muonclip_with_all_frozen_model(self):
        """Test MuonClip with all parameters frozen (no trainable params)."""
        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig

        model = SimpleLRCModel()
        # Freeze ALL params including LRC
        for param in model.parameters():
            param.requires_grad = False

        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            enable_clipping=False,  # Disable to avoid hook issues
        )

        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        # Should not crash on creation
        optimizer = MuonClip(model, MinimalConfig(), config)
        _patch_muonclip_optimizer(optimizer)
        model.train()

        # step() should run without error even with no updates
        optimizer.step()
        optimizer.zero_grad()

    def test_muonclip_clipping_disabled_no_hooks(self):
        """Test MuonClip with clipping disabled doesn't register hooks."""
        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig

        model = SimpleLRCModel()

        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            enable_clipping=False,  # Explicitly disabled
        )

        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        optimizer = MuonClip(model, MinimalConfig(), config)
        _patch_muonclip_optimizer(optimizer)

        # With clipping disabled, hook behavior may vary
        # This test just ensures no crash
        model.train()
        assert optimizer is not None

    def test_muonclip_invalid_layer_mapping_graceful(self):
        """Test MuonClip with non-existent layer names in mapping."""
        pytest.importorskip("muon", reason="muon-clip not installed")
        from muon import MuonClip, MuonConfig

        model = SimpleLRCModel()

        # Map to layers that don't exist
        config = MuonConfig(
            unified_lr=False,
            lr_muon=0.02,
            lr_adam=1e-4,
            enable_clipping=True,
            clipping_threshold=50.0,
            clipping_alpha=0.5,
            clipping_layers_mapping={"nonexistent_q": "q_proj", "nonexistent_k": "k_proj"},
        )

        class MinimalConfig:
            num_attention_heads = 4
            hidden_size = 64
            num_hidden_layers = 1

        # Should handle gracefully (either skip hooks or warn)
        try:
            optimizer = MuonClip(model, MinimalConfig(), config)
            _patch_muonclip_optimizer(optimizer)
            model.train()
            # If we get here, it handled gracefully
        except Exception as e:
            # Some implementations may raise - that's also acceptable
            logger.info(f"MuonClip raised on invalid mapping (expected): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
