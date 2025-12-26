# Ablation: Influence-Based Data Re-weighting

**Date:** 2025-12-26
**Status:** In Progress

## Hypothesis

Does influence-based dynamic re-weighting of training data sources improve model quality compared to fixed initial weights?

The CheaperTraining library implements MobileLLM-R1's Phase II methodology: compute influence of training samples on a multi-domain probe set (code, math, web), then adjust data source weights every N steps.

## Experiment Design

| Experiment | Description | Config Override |
|------------|-------------|-----------------|
| A (control) | Dynamic re-weighting enabled | `training.influence.enabled=true` (default) |
| B (treatment) | Fixed initial weights | `training.influence.enabled=false` |

### Common Configuration

| Parameter | Value |
|-----------|-------|
| Model | SmolLM2-135M |
| Stage | 2 (continue_pretrain) |
| Dataset | `data=mixed_pretrain` |
| Data Sources | DCLM (25%), FineWeb-Edu (30%), GitHub (15%), FineMath (15%), SlimPajama (15%) |
| Steps | 5000 |
| Batch Size | 32 |
| GPU | H100 (Nebius, 1x) |
| Update Interval | 250 steps (for Exp A) → 20 weight updates |

## Commands

```bash
# Experiment A: Dynamic re-weighting
uv run wf train -m smollm2_135m -s 2 --scale dev \
  training.max_steps=5000 \
  data=mixed_pretrain \
  experiment_name=ablation_influence_dynamic

# Experiment B: Fixed weights
uv run wf train -m smollm2_135m -s 2 --scale dev \
  training.max_steps=5000 \
  data=mixed_pretrain \
  training.influence.enabled=false \
  experiment_name=ablation_influence_fixed
```

## Metrics

### Training Metrics (W&B)
- `train/loss` - overall cross-entropy loss
- `train/accuracy` - next-token prediction accuracy
- `train/perplexity` - exponentiated loss
- `influence/weight_*` - per-source weights (Exp A only)

### Downstream Evaluation
- BitDistill benchmark via WrinkleFree-Eval
- Compare: HellaSwag, ARC, WinoGrande, etc.

## Results

### Run IDs
- Exp A (dynamic): `sky-smollm2_135m-s2:36` (Nebius H100 on-demand, update_interval=250, no W&B)
- Exp B (fixed): `sky-smollm2_135m-s2:39` (Nebius H100 on-demand, no W&B)

### Training Curves
*To be added after runs complete*

### Final Metrics
| Metric | Exp A (Dynamic) | Exp B (Fixed) | Delta |
|--------|-----------------|---------------|-------|
| Final Loss | TBD | TBD | TBD |
| Final Perplexity | TBD | TBD | TBD |

### Weight Evolution (Exp A)
*To be added - shows how source weights changed over training*

### Downstream Eval
| Benchmark | Exp A (Dynamic) | Exp B (Fixed) | Delta |
|-----------|-----------------|---------------|-------|
| HellaSwag | TBD | TBD | TBD |
| ARC-Easy | TBD | TBD | TBD |
| WinoGrande | TBD | TBD | TBD |

## Conclusion

*To be written after experiment completes*

## Notes

- Both experiments use the same random seed for reproducibility
- GCS checkpoints stored at `gs://wrinklefree-checkpoints/smollm2_135m/stage2/`
- W&B project: `wrinklefree`
