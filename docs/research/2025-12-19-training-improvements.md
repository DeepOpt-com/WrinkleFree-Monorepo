# Training Improvements Research

**Date:** 2025-12-19
**Topic:** Opportunities to make WrinkleFree training faster and better

---

## Executive Summary

Research into recent advances in LLM training, 1.58-bit quantization, and datasets reveals several high-impact opportunities for improving WrinkleFree's training pipeline. Key findings include better datasets (MegaMath, DCLM), training optimizations (sequence packing, optimal quantization transitions), and distillation improvements.

---

## 1. Better Datasets

### MegaMath (Recommended)
- **Size:** 370B tokens
- **Source:** [HuggingFace](https://huggingface.co/datasets/LLM360/MegaMath) | [Paper](https://arxiv.org/abs/2504.02807) | [GitHub](https://github.com/LLM360/MegaMath)
- **Why:** Largest open math pre-training dataset, accepted to COLM 2025. Provides 15-20% performance boost on downstream math benchmarks vs smaller alternatives like OpenWebMath.
- **Contents:**
  - Re-extracted mathematical documents from Common Crawl with math-oriented HTML optimizations
  - Math-related code from Stack-V2
  - Synthesized QA-style text and interleaved text-code blocks
- **Recommendation:** Replace OpenWebMath with MegaMath for math reasoning capability

### DCLM-Baseline
- **Size:** 4T tokens / 3B documents
- **Source:** [HuggingFace](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | [GitHub](https://github.com/mlfoundations/dclm)
- **Why:** Apple's DataComp-LM achieves strong performance through rigorous cleaning, filtering, and deduplication of Common Crawl. Outperforms FineWeb on many benchmarks.
- **Recommendation:** Consider as primary web corpus replacement for FineWeb

### DCLM-Edu
- **Size:** Educational subset of DCLM
- **Source:** [HuggingFace](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu)
- **Why:** Filtered using FineWeb-Edu classifier (score > 2). Used to train SmolLM2-135M and SmolLM2-360M. May be better than FineWeb-Edu for small models.

### Ultra-FineWeb
- **Size:** ~120B Chinese tokens (Chinese focus, but methodology applicable)
- **Source:** [Paper](https://arxiv.org/html/2505.05427)
- **Why:** FastText-based lightweight classifier filtering of FineWeb. Significant quality improvements on benchmarks.

### Other Notable Datasets

| Dataset | Size | Use Case |
|---------|------|----------|
| Falcon RefinedWeb | 500-650B tokens | High-quality filtered web data |
| Nemotron-CC-HQ | Large | NVIDIA's CC corpus for high-quality pre-training |
| FineWeb2 | Multilingual | 1000+ languages support |
| StarCoder Data | 783GB | Code in 86 programming languages |

---

## 2. Training Technique Improvements

### Continual Quantization-Aware Pre-Training

**Source:** [arXiv:2502.11895](https://arxiv.org/abs/2502.11895)

Research shows there's an **optimal transition point** from 16-bit to 1.58-bit training:

> "Results on 11 downstream tasks show that this 16-to-1.58-bit training strategy is preferable over full 1.58-bit training and leaves models closer to those which have undergone 16-bit training."

**Key Finding:** The data-optimal transition point exists - switching too early or too late hurts final model quality.

**Current State:** WrinkleFree uses a 3-stage approach (SubLN insertion → continue pre-training → distillation). May need to investigate optimal transition timing.

### Sequence Packing

**Source:** [HuggingFace Blog](https://huggingface.co/blog/sirluk/llm-sequence-packing) | [Paper](https://arxiv.org/abs/2107.02027)

**Problem:** Up to 50% of tokens can be padding in standard training, wasting compute.

**Solution:** Concatenate multiple shorter sequences into single inputs with proper attention masking.

**Benefits:**
- 2x speedup for pre-training
- 40% speed increase demonstrated with Vicuna-1.5 13B
- Effectively increases batch size with minimal overhead

**Implementation:**
- Available in Transformers 4.43+ via `DataCollatorWithFlattening`
- Flash Attention 2 supports proper masking for packed sequences
- Must adjust position IDs to demarcate sequence boundaries
- Requires hyperparameter adjustment (LR, optimizer params) due to effective batch size increase

**Caution:** Cross-contamination can occur if attention isn't properly masked between sequences.

### Curriculum Learning + LR Decay Interaction

**Source:** [arXiv:2511.18903](https://arxiv.org/abs/2511.18903)

**Critical Finding:** Standard learning rate decay schedules are incompatible with curriculum learning (easy→hard data ordering):

> "While curriculum-based training substantially outperforms random shuffling when using a constant LR, its advantage diminishes under standard LR decay schedules."

**Problem:** Best/hardest data arrives when LR is lowest, wasting its potential.

**Solutions:**
1. Use moderate LR decay (final LR only slightly smaller than peak)
2. Replace LR decay with model averaging
3. Combining both strategies: 1.64% improvement over random shuffling

### Multi-Stage Pretraining (Mid-Training)

**Source:** [OLMo 2 Survey](https://arxiv.org/html/2510.23081)

Successful curriculum strategy:
1. First stage: Data mixture dominated by massive web data
2. Second stage (mid-training): Shift to high-quality data with LR annealing

**Techniques:**
- **Microanneals:** Extrapolate efficacy of large-scale mixtures from small-scale trials
- **RegMix:** Formalizes ratio optimization as regression task

---

## 3. Optimizer Improvements

### Current State
WrinkleFree uses Muon (default) and Apollo optimizers. Apollo emerged as winner in Bayesian hyperparameter search.

### 8-bit Muon

**Source:** [GitHub/Muon](https://github.com/KellerJordan/Muon)

Blockwise quantization of Muon optimizer:
- ~74% reduction in memory footprint vs full-precision Muon
- Supports both linear and dynamic quantization schemes
- Minimal quality loss

### MuonClip

**Source:** [Moonlight Paper](https://arxiv.org/abs/2502.16982)

Used in Kimi-2 (1 trillion parameter LLM):
- Dramatically smoother loss curves
- Better training dynamics than standard Muon

### NorMuon (Neuron-wise Normalized Muon)

Addresses non-uniform neuron norms in Muon updates:
- Combines orthogonalization with neuron-level adaptive learning rates
- Fixes the finding that Muon's resulting updates exhibit highly non-uniform neuron norms

### Muon Scalability Improvements (2025)

**Source:** [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)

Two crucial techniques for large-scale Muon:
1. Adding weight decay
2. Carefully adjusting per-parameter update scale

**Result:** Muon achieves comparable performance to AdamW while requiring only ~52% of training FLOPs.

---

## 4. Distillation Improvements

### Current State
WrinkleFree BitDistill uses:
```
L = L_CE + λ×L_LD + γ×L_AD
```
- L_CE: Cross-entropy loss
- L_LD: Logits KL divergence (λ=10.0 classification, 1.0 summarization)
- L_AD: Attention distribution distillation (γ=1e-5 classification, 1e-3 summarization)

### Intermediate Layer Distillation

**Source:** [Survey](https://www.sciencedirect.com/science/article/pii/S2666827024000811)

Add hidden state distillation from intermediate layers:
```
L = L_CE + λ×L_LD + γ×L_AD + δ×L_hidden
```

Benefits:
- Transfers more detailed and structured information
- Student captures teacher's internal representations, not just outputs

### Multi-Teacher Distillation

Use multiple teacher models (e.g., Qwen + LLaMA):
- More comprehensive knowledge transfer
- Improved robustness
- Student benefits from diverse teacher perspectives

### Synthetic Data Generation

**Source:** [LLM QAT Research](https://www.exxactcorp.com/blog/deep-learning/what-is-llm-distillation-vs-quantization)

> "LLM QAT shows the utility of synthetic data for QAT, generating synthetic training data from full-precision model and using knowledge distillation to train quantized models."

The teacher model can generate synthetic data specifically tuned for the student's weaknesses.

### Combined Compression Pipeline

**Source:** [NVIDIA Blog](https://developer.nvidia.com/blog/llm-model-pruning-and-knowledge-distillation-with-nvidia-nemo-framework/)

Optimal workflow:
1. First prune the model
2. Then quantize
3. Use knowledge distillation to fine-tune and recover lost performance

Result: ~4x size reduction (e.g., 2.7B → 700M parameters), then int8 quantization for ~14x total reduction.

---

## 5. Quick Wins Summary

| Improvement | Effort | Expected Gain | Priority |
|-------------|--------|---------------|----------|
| Switch to MegaMath (replace OpenWebMath) | Low | 15-20% on math benchmarks | High |
| Add sequence packing | Medium | Up to 2x training throughput | High |
| Use DCLM instead of FineWeb | Low | Better downstream performance | Medium |
| Implement 8-bit Muon | Low | 74% less optimizer memory | Medium |
| Investigate optimal 16→1.58bit transition | Medium | Better final quality | Medium |
| Add intermediate layer distillation | Medium | More structured knowledge transfer | Low |
| Fix LR decay + curriculum interaction | Low | ~1.6% improvement | Low |

---

## 6. Recommended Data Mix

Based on research, suggested replacement for current mix:

**Current Mix:**
- FineWeb (40%)
- SlimPajama (30%)
- OpenWebMath (15%)
- CodeParrot (15%)

**Proposed Mix:**
- DCLM-Baseline (40%) - Higher quality than FineWeb
- MegaMath-Web (25%) - Much larger than OpenWebMath
- MegaMath-Code (15%) - Math-focused code
- StarCoder/CodeParrot (20%) - General code

---

## References

1. [MegaMath Dataset](https://huggingface.co/datasets/LLM360/MegaMath) - LLM360
2. [DCLM-Baseline Dataset](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) - ML Foundations
3. [Ultra-FineWeb Paper](https://arxiv.org/html/2505.05427) - arXiv
4. [Continual QAT for BitNet](https://arxiv.org/abs/2502.11895) - ACL 2025
5. [Muon Optimizer](https://github.com/KellerJordan/Muon) - GitHub
6. [Muon Scalability Paper](https://arxiv.org/abs/2502.16982) - Moonshot AI
7. [Sequence Packing Guide](https://huggingface.co/blog/sirluk/llm-sequence-packing) - HuggingFace
8. [Curriculum + LR Decay](https://arxiv.org/abs/2511.18903) - arXiv
9. [BitDistill Paper](https://arxiv.org/pdf/2510.13998) - Microsoft
10. [NVIDIA Pruning + Distillation](https://developer.nvidia.com/blog/llm-model-pruning-and-knowledge-distillation-with-nvidia-nemo-framework/) - NVIDIA
11. [DataComp-LM](https://github.com/mlfoundations/dclm) - GitHub
12. [OLMo 2 Mid-Training Survey](https://arxiv.org/html/2510.23081) - arXiv
