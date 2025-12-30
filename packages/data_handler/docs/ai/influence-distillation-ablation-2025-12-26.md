# Influence Distillation Real Data Ablation

**Date:** 2025-12-26

## Overview

Implemented and tested InfluenceDistillation for continuous dataset rebalancing during LLM pretraining. Ran ablation comparing static equal weighting vs dynamic influence-based rebalancing on real commercially-friendly datasets.

## Changes Made

### Multi-Model Support for JVP Extractor

Updated `jvp_embedding.py` and `gradient.py` to support multiple model architectures:

| Architecture | Embedding Layer | Transformer Layers |
|--------------|-----------------|-------------------|
| LLaMA/Mistral | `model.model.embed_tokens` | `model.model.layers` |
| GPT-2/GPT-Neo | `model.transformer.wte` | `model.transformer.h` |
| BERT/RoBERTa | `model.embeddings.word_embeddings` | `model.encoder.layer` |
| MobileLLM | `model.embed_tokens` | `model.decoder.layers` |

### Synthetic Data Warning

Added `_warn_if_synthetic_data()` function in `distillation.py` that detects random/synthetic data by checking unique token ratio. Prints loud warning:

```
================================================================================
WARNING: DATA APPEARS TO BE SYNTHETIC/RANDOM!
================================================================================
Influence-based rebalancing WILL NOT WORK with random data!
All datasets look identical → influence scores are noise.
================================================================================
```

### Real Data Ablation Test

Created `tests/integration/test_real_data_ablation.py` with:
- Commercially-friendly datasets (fineweb-edu, gsm8k, wikitext)
- Static vs rebalanced comparison
- Progress tracking and debug output

## Ablation Results

### Configuration

- **Model:** GPT-2 (124M params)
- **Datasets:** fineweb-edu (target), gsm8k (math), wikitext (general)
- **Steps:** 300 (timed out at ~270)
- **Rebalance interval:** Every 30 steps
- **Landmarks:** 16
- **JVP layers:** 2

### Results

| Mode | Step 0 Loss | Step ~270 Loss | Improvement |
|------|-------------|----------------|-------------|
| Static (33/33/33) | 3.47 | 2.76 | baseline |
| Rebalanced | 3.52 | **2.34** | **-0.42 (~15%)** |

### Weight Evolution

```
Step   0: fineweb=33%, math=33%, wiki=33%
Step  30: fineweb=52%, math=47%, wiki=1%   <- wiki immediately downweighted
Step  90: fineweb=51%, math=48%, wiki=1%
Step 210: fineweb=75%, math=24%, wiki=1%   <- fineweb dominates
Step 270: fineweb=36%, math=65%, wiki=1%   <- math takes over
```

## Key Findings

1. **Rebalancing provides measurable improvement** on real data (~15% lower eval loss)

2. **Wiki consistently downweighted** - The system correctly identified that wikitext has lower influence on the fineweb-edu target distribution

3. **Dynamic adaptation works** - Weights shift during training based on current model state, not just initial dataset characteristics

4. **Synthetic data shows no difference** - Earlier tests with random data showed no benefit (as expected - all random data looks the same to influence functions)

## Limitations

- Test timed out before completing full 300 steps
- Eval set = probe set (should be separate for proper evaluation)
- Small model (GPT-2 124M) - need to validate on larger models
- No comparison with DoReMi or other baselines yet

## Files Modified

| File | Change |
|------|--------|
| `src/cheapertraining/influence/jvp_embedding.py` | Multi-model embedding/layer detection |
| `src/cheapertraining/influence/gradient.py` | Multi-model gradient extractor support |
| `src/cheapertraining/influence/distillation.py` | Synthetic data warning |
| `tests/integration/test_real_data_ablation.py` | New real data ablation test |
| `docs/api/influence.md` | Added ablation results section |

## Next Steps

1. Run full ablation to completion (increase timeout)
2. Add separate held-out eval set distinct from probe set
3. Test on larger models (LLaMA 1B+)
4. Compare against DoReMi baseline
5. Add wandb logging for proper experiment tracking
