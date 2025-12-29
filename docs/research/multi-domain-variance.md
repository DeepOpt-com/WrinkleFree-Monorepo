# Multi-Domain Pre-Training Loss Variance

## Observation

Training Qwen3-4B Stage 2 (BitNet 1.58-bit) with mixed data sources shows high loss variance:
- Loss oscillates between **2.5 and 4.0** across steps
- Standard deviation: ~0.5

Example trajectory:
```
Step 6100: 3.06    Step 6170: 4.06
Step 6110: 2.81    Step 6180: 3.54
Step 6120: 2.55    Step 6190: 3.67
Step 6130: 2.70    Step 6200: 3.06
Step 6140: 3.56    Step 6210: 3.35
Step 6150: 3.43    Step 6220: 3.60
Step 6160: 3.55    Step 6230: 3.85
```

## Root Causes (Gemini Analysis)

### 1. Optimizer Mismatch - Adam vs MuonClip

**Problem**: Using Adam for BitNet 1.58-bit training causes instability.

BitNet models have spiky gradients due to quantization. Adam doesn't handle this well.
The project defaults to **MuonClip** (`muonclip`) which is specifically designed for BitNet:
- Momentum optimization for quantized weights
- QK-norm clipping for stability

**Recommendation**: Switch from Adam to MuonClip, or reduce LR significantly (3e-4 → 1e-4).

### 2. Influence Updates May Be Disabled

**Potential Bug**: The `InfluenceAwareOptimizer` may be passing empty `{}` to the weight calculator,
effectively disabling dynamic data reweighting.

**Effect**: Data mixture weights stay static (25/35/10/15/15) rather than adapting to model needs.

**Location**: `packages/cheapertraining/src/cheapertraining/training/optimizer.py`

### 3. Expected Variance from Data Domains

Even with perfect optimization, some variance is expected:
- **Code** (10%): Highly structured, lower entropy → lower loss
- **Math** (15%): Symbolic, specialized vocabulary → variable loss
- **Web/Edu** (60%): Natural language, higher entropy → moderate loss

A batch dominated by code will have lower loss than one dominated by web text.

## Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| High | Switch to MuonClip optimizer | -50% variance |
| High | Reduce LR to 1e-4 if staying with Adam | -30% variance |
| Medium | Increase batch size to 512+ | -20% variance |
| Medium | Fix influence optimizer bug | Better domain balancing |
| Low | Increase shuffle buffer to 100k | Smoother domain mixing |

## Training Config for Reduced Variance

```bash
# Recommended: MuonClip optimizer
uv run python scripts/train.py \
  model=qwen3_4b \
  training=stage2_pretrain \
  training.optimizer.name=muonclip \
  training.batch_size=8 \
  training.gradient_accumulation_steps=64  # effective 512

# If using Adam: lower LR
uv run python scripts/train.py \
  model=qwen3_4b \
  training=stage2_pretrain \
  training.optimizer.lr=1e-4 \
  training.batch_size=8 \
  training.gradient_accumulation_steps=64
```

## References

- BitDistill paper: arxiv.org/abs/2510.13998
- MuonClip: Momentum optimization for quantized training
- ZClip: Adaptive gradient clipping (arxiv:2504.02507)
