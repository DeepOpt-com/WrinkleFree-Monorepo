# System Architecture

## Overview

WrinkleFree-1.58Quant is a framework for training 1.58-bit LLMs using the BitDistill approach. The system is designed around a 3-stage training pipeline that progressively adapts a full-precision model to ternary weights.

## Core Components

### 1. Model Architecture
- **BitLinear**: Replaces standard `nn.Linear` layers. Implements ternary quantization {-1, 0, 1} for weights and 8-bit quantization for activations. Uses Straight-Through Estimator (STE) for gradient computation.
- **SubLN**: Sub-Layer Normalization (RMSNorm) inserted before output projections in Self-Attention and FFN blocks to stabilize training.
- **Llama-style Backbone**: Based on Llama/Qwen architectures with RoPE, GQA, and SwiGLU.

### 2. Training Pipeline

#### Stage 1: SubLN Insertion
- **Goal**: Stabilize the model for quantization.
- **Process**: Inserts `SubLN` modules into a pre-trained full-precision model.
- **Training**: Short fine-tuning to adapt the new normalization layers.

#### Stage 2: Continue Pre-training
- **Goal**: Adapt weight distributions for quantization.
- **Process**: Trains the model with `BitLinearWithWarmup` layers.
- **Techniques**:
    - **Quantization-Aware Training (QAT) with Warmup**: 
        - **Implementation**: Uses `BitLinearWithWarmup` layers.
            ```python
            # Simplified Logic
            lam = min(step / 1000, 1.0)  # Linear warmup
            if lam < 1.0:
                w_mixed = (1 - lam) * w + lam * w_quant
                w_out = w + (w_mixed - w).detach()  # STE
            else:
                w_out = w + (w_quant - w).detach()
            ```
        - **Schedule**: 
            - Steps 0-1000: Linear interpolation from FP to Ternary.
            - Steps 1000+: Full ternary {-1, 0, 1} quantization.
        - **Activation Quantization**: 8-bit per-token quantization (always on).
        - **Rationale**: Gradual warmup prevents gradient instability and helps models adapt to discrete weight spaces.
        - **Comparison**:
            | Method | Warmup | Stability | Performance |
            |--------|--------|-----------|-------------|
            | Immediate QAT | None | ⚠️ Unstable | ~85% recovery |
            | Post-Training Quant | None | ✅ Stable | ~75% recovery |
            | **QAT with Warmup** | **1000 steps** | ✅ **Stable** | **~95% recovery** |
    - **Influence-based Data Selection**: Uses `WrinkleFree-CheaperTraining` library to dynamically adjust data mixture weights based on influence functions, ensuring the model sees the most relevant data for the target task/domain.

#### Stage 3: Distillation Fine-tuning
- **Goal**: Recover performance using a teacher model.
- **Process**: Distills knowledge from a full-precision teacher to the 1.58-bit student.
- **Loss Function**: `BitDistillLoss` = Cross-Entropy + Logits KL Divergence + Attention Distillation.

### 4. Q-Sparse Activation Sparsity (Optional)

Q-Sparse ([arxiv:2407.10969](https://arxiv.org/abs/2407.10969)) adds activation sparsity on top of weight quantization for additional inference efficiency.

- **Mechanism**: Top-K sparsification keeps only the largest (1 - sparsity_ratio) activations per token
- **Order**: Sparsify → Quantize (preserves important activations for quantization)
- **STE Gradient Flow**: Uses detach trick for straight-through estimation
- **Optimal Sparsity**: 61% for 1-bit models (per paper)

**Implementation Files:**
- `src/wrinklefree/quantization/activation_sparse.py` - Core sparsification functions
- `src/wrinklefree/quantization/sparsity_warmup.py` - Gradual warmup schedule
- Integration in `src/wrinklefree/models/bitlinear.py`

**Trade-offs (200-step ablation on SmolLM2-135M):**
| Config | Final Loss | Training Time |
|--------|------------|---------------|
| Without Q-Sparse | ~6.93 | 480s |
| With Q-Sparse (61%) | ~6.90 | 648s (~35% slower) |

**Recommendation**: Disabled by default due to training slowdown. Enable for production models where inference efficiency is critical:
```bash
training.activation_sparsity.enabled=true
```

The ~35% training overhead pays off at inference where 61% of activations can be skipped, compounding with 1.58-bit weight quantization.

### 3. LRC Calibration (Post-Quantization Recovery)

Low-Rank Correction (LRC) provides post-training quantization error recovery by adding trainable low-rank matrices.

Based on paper: [Low-Rank Correction for Quantized LLMs](https://arxiv.org/abs/2412.07902)

**Key Idea**: Add trainable U, V matrices that operate on unquantized activations to correct quantization errors.

```
output = W_quant @ Q_a(X) + U @ V^T @ X
         ─────────────────   ───────────
         Frozen quantized     Trainable LRC
         path                 correction
```

**LRC Training**:
1. Convert BitLinear → BitLinearLRC (adds U, V matrices)
2. Freeze ALL parameters except U, V
3. Train using hidden state matching loss vs. fp16 teacher

**Loss Function**: `||h_teacher - h_student||²` per layer
- h_teacher: Hidden states from original fp16 model
- h_student: Hidden states from LRC-corrected quantized model

**Parameter Counts** (Qwen3-4B example):
| Component | Parameters | Trainable |
|-----------|------------|-----------|
| Frozen weights | ~4B | ❌ No |
| LRC U, V | ~400M (10% rank) | ✅ Yes |
| Embeddings | ~200M | ❌ No |

**Usage**:
```bash
uv run python scripts/train.py \
  model=smollm2_135m \
  training=lrc_calibration \
  data=fineweb
```

**Config** (`lrc_calibration.yaml`):
```yaml
lrc:
  rank_percentage: 0.1    # 10% of min(in, out) per layer
  init_method: zeros      # or "svd_residual"

objectives:
  lrc_reconstruction:
    enabled: true
    weight: 1.0
    loss_type: mse
    layer_weights: progressive
```

### 4. Influence Integration
The system integrates with `WrinkleFree-CheaperTraining` to optimize data selection:
- **InfluenceAwareOptimizer**: Wraps the standard optimizer to intercept steps.
- **DataInfCalculator**: Computes influence scores of training data on a validation "probe" set.
- **MixtureWeightCalculator**: Optimizes dataset mixture weights to maximize influence on the probe set.

## Directory Structure

```
src/wrinklefree/
├── models/           # BitNet architecture components
├── quantization/     # Quantization logic (STE, weight/activation quant)
├── distillation/     # Distillation losses (Logits, Attention)
├── training/         # Training stages and trainer loop
├── data/             # Data loading and streaming
└── serving/          # Inference and export tools
```
