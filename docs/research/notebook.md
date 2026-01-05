## 01-05-2026

### Feature: Lazy Teacher Loading for Distillation

**Problem**: Teacher model was loaded at startup even when distillation weight started at 0 in curriculum phases, wasting GPU memory during early training.

**Solution**: Implemented lazy loading that defers teacher model loading until distillation weight first becomes non-zero.

**Key Changes**:
1. `_get_distill_status()` - Analyzes curriculum to determine if/when teacher is needed
2. `_lazy_load_teacher()` - Loads teacher on-demand when distill weight > 0
3. `_reduce_batch_size()` - Automatically reduces batch size when teacher loads (default 0.5x)
4. Skip distillation on transition step to let reduced batch size take effect

**Files Modified**:
- `packages/training/scripts/train_lightning.py` - Lazy loading logic
- `packages/training/src/wf_train/lightning/module.py` - On-demand teacher loading

**Config Option**:
```yaml
teacher:
  batch_size_factor: 0.5  # Reduce batch by 50% when teacher loads
```

### Fix: max_steps Computation from total_tokens

**Problem**: `training=full_run` config uses `total_tokens: 10_000_000_000` with `max_steps: null`, causing Lightning Trainer to crash with `TypeError: '<' not supported between instances of 'NoneType' and 'int'`.

**Fix**: Added computation of `max_steps` from `total_tokens`:
```python
max_steps = total_tokens / (batch_size * seq_length * grad_accum)
```

### Fix: Misleading GCS Error Message

**Problem**: Audit logger printed "TRAINING ABORTED" when GCS credentials were missing, but training actually continued without checkpoint sync.

**Fix**: Changed message to "GCS DISABLED - training continues without checkpoint sync".

### Fix: NaN Loss with MuonClip + Qwen3-0.6B

**Problem**: Training with `model=qwen3_0.6b training=full_run` produces `loss: nan` after ~776 steps.

**Root Cause**: MuonClip hyperparameters were tuned for smaller models. The high learning rate (0.02) combined with sudden quantization caused gradient explosions.

**Fix** (in `full_run.yaml`):
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `lr_muon` | 0.02 | 0.005 | 4x reduction for larger models |
| `lr_adam` | 3e-4 | 1e-4 | 3x reduction for stability |
| `clipping_threshold` | 50.0 | 10.0 | Catch gradient explosions earlier |
| `warmup_steps` | 550 | 1000 | Longer warmup for stability |
| `lambda_warmup.enabled` | false | true | Gradual quantization prevents sudden instability |

**MuonClip Best Practices for BitNet**:
1. Scale `lr_muon` inversely with model size: 0.02 for 135M, 0.01 for 350M, 0.005 for 0.6B+
2. Always enable lambda warmup for gradual quantization transition
3. Keep `clipping_threshold` low (10-20) to catch explosions early
4. Match `warmup_steps` between scheduler and lambda_warmup
5. Use `clipping_alpha=0.5` for balanced clipping

---

## 01-02-2026

### Bug Fix: DLM Preprocessing Applied When Weight=0 (Catastrophic Loss Issue)

**Problem**: BitNet training showed catastrophic loss (20-41 instead of expected 2-5) during warmup phase when DLM objective weight was 0.

**Root Cause Analysis**:

The `ObjectiveManager.preprocess_batch()` method was applying DLM preprocessing (50% token masking with complementary masks) **regardless of the objective's current weight**. During the warmup curriculum phase:
- DLM weight = 0.0 (objective disabled)
- But DLM's `preprocess_batch()` was still called
- Input tokens were masked, batch size was doubled
- `continue_pretrain` computed CE loss from masked input → artificially high loss

**Investigation Path**:
1. Initially suspected SubLN insertion breaking pretrained weights (fixed in v12)
2. Local tests showed model conversion works correctly (loss=2.14)
3. But training still showed loss=20+ even with SubLN disabled
4. Created debug script that mimicked full training pipeline
5. Discovered that at lambda=0 with DLM preprocessing: loss=9.73 (vs 2.24 without)
6. Found that `preprocess_batch()` doesn't check objective weights

**Fix**: Modified `ObjectiveManager.preprocess_batch()` to check if objective weight > 0 before applying preprocessing:

```python
# packages/training/src/wf_train/objectives/manager.py
def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
    # Get current weights to check which objectives are active
    weights = self.get_current_weights()

    for name, obj in self.objectives.items():
        if obj.modifies_input:
            # Only apply preprocessing if objective weight > 0
            obj_weight = weights.get(name, self.base_weights.get(name, 1.0))
            if obj_weight > 0:
                batch = obj.preprocess_batch(batch)
    return batch
```

**Files Modified**:
- `packages/training/src/wf_train/objectives/manager.py` - Skip preprocessing when weight=0
- `packages/architecture/src/wf_arch/conversion/convert.py` - Added `insert_subln` flag (v12 fix)
- `packages/training/configs/training/lrc_dlm_influence.yaml` - Set `insert_subln: false`

**Verification**:
- Debug script confirms: At step 0 with DLM weight=0, masked tokens=0, batch size=1
- At step 50 with DLM weight=0.1875, masked tokens=12, batch size=2 (correctly doubled)

**Training Runs**:
- v12 (lrc_dlm_v12_no_subln): Still showed high loss due to DLM preprocessing bug
- v13 (lrc_dlm_v13_dlm_preproc_fix): Pending - should show normal loss ~2-3 during warmup

**Key Insight**: When using curriculum learning with phase-based objective weights, preprocessing must respect the current weights, not just whether the objective exists.

---

## 12-31-2025

### TCS Distillation Hyperparameter Corrections (Job 19)

**Problem**: Job 18 showed loss plateau at ~4.9-5.5 instead of consistent decrease. WandB was also disabled, making monitoring impossible.

**Root Cause Analysis** (via BitDistill paper review):

Our `lambda_logits=0.1` was **100x too low** compared to BitDistill recommendations:
- BitDistill paper: `lambda=10` (classification) or `lambda=1` (summarization)
- Our config: `lambda=0.1` (way too low!)
- With T²=25 internal scaling: effective weight was only 2.5x instead of 25x

Similarly, `gamma_attention=1e-5` was too low:
- BitDistill paper: `gamma=1e3-1e5` (normalized internally)
- Our config: `gamma=1e-5` (at the very bottom)

**Corrections Applied**:

| Parameter | Before | After | BitDistill Reference |
|-----------|--------|-------|---------------------|
| `lambda_logits` | 0.1 | **1.0** | 1-10 |
| `gamma_attention` | 1e-5 | **1e-4** | 1e3-1e5 (scaled down for stability) |
| WandB | disabled | **enabled** | - |

**Files Modified**:
- `packages/distillation/configs/distillation/tcs.yaml` - Updated hyperparameters
- `packages/distillation/src/distillation/training/trainer.py` - Added VERY LOUD ASCII art warning if WANDB_API_KEY not set
- `packages/distillation/src/distillation/trainer.py` - Same warning
- `packages/deployer/skypilot/tcs_distill_train.yaml` - Added WANDB_API_KEY and HF_TOKEN env vars
- `packages/distillation/CLAUDE.md` - Added MUST-DO rule #5 for WANDB_API_KEY

**Loss Formula** (for reference):
```
L = L_CE + lambda_logits * L_TCS + gamma_attention * L_BlockAttn

Where:
- L_CE: Cross-entropy loss (no shift for DLM)
- L_TCS: KL(softmax(teacher_topk/T) || softmax(student[topk_indices]/T)) * T²
- L_BlockAttn: Block-wise attention relation distillation
- T=5 → T²=25x internal scaling on KL term
```

**Expected Outcome**: With lambda=1.0 (effective 25x weight on TCS loss), the model should learn better from teacher logits and show consistently decreasing loss.

---

## 12-21-2025

### Bug Fix: BitNetLlamaForSequenceClassification lm_head crash

**Problem**: `BitNetLlamaForSequenceClassification` sets `self.model.lm_head = None` but `BitNetLlama.forward()` would crash trying to call `self.lm_head(hidden_states)`.

**Fix**: Modified `BitNetLlama.forward()` to handle `lm_head is None` case - returns hidden states directly when no LM head is present.

---

### Known Issue: position_ids RoPE Shape Mismatch in WrinkleFree-1.58Quant

**Problem**: When `position_ids` is provided to `BitNetAttention.forward()`, the RoPE frequency indexing produces wrong shapes.

**Root Cause**: In `attention.py:176`, when `position_ids` has shape `(batch, seq_len)`, `self.freqs_cis[position_ids]` returns `(batch, seq_len, head_dim//2)`. However, `apply_rotary_emb()` expects `(seq_len, head_dim//2)` and does its own broadcasting.

**Workaround**: Don't pass explicit `position_ids` - let the module auto-generate sequential positions (which works correctly).

**Status**: Test skipped, needs fix in attention module.

---

### Bug Fix: Missing haar.py in WrinkleFree-1.58Quant

**Problem**: The `wf_train.quantization.haar` module was missing, causing all tests to fail with `ModuleNotFoundError`.

**Root Cause**: The `__init__.py` and `haar_triton.py` files were importing from `wf_train.quantization.haar` but the file was never created.

**Fix**: Created `/src/wf_train/quantization/haar.py` with pure PyTorch implementations of:
- `haar_transform_1d_row` - Forward Haar transform
- `inverse_haar_transform_1d_row` - Inverse Haar transform
- `haar_weight_quantization` - Full Haar wavelet quantization
- `haar_weight_quantization_no_scale` - Returns raw quantized values + scale

---

## 12-18-2025
