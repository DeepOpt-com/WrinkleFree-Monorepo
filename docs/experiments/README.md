# Experiments

Documentation for experiments and ablation studies in the WrinkleFree project.

## Index

| Experiment | Date | Status | Description |
|------------|------|--------|-------------|
| [Influence Reweighting Ablation](ablation_influence_reweighting_2025-12-26.md) | 2025-12-26 | In Progress | Tests whether influence-based dynamic re-weighting of training data sources improves model quality |
| [Coverage Baseline](coverage-baseline.md) | 2025-12-26 | Baseline | Test coverage report across all packages |

## Running Experiments

Most experiments use the standard training pipeline with config overrides:

```bash
# Example: Run influence ablation experiment
uv run --package wf-train python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=base \
  training.influence.enabled=true

# Disable influence for control
uv run --package wf-train python packages/training/scripts/train_lightning.py \
  model=smollm2_135m \
  training=base \
  training.influence.enabled=false
```

## Adding New Experiments

1. Create a new markdown file with naming convention: `{topic}_{YYYY-MM-DD}.md`
2. Include: Hypothesis, Design, Results, Conclusions
3. Update this README with a link to the new experiment
