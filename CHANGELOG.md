# Changelog

All notable changes to the WrinkleFree monorepo.

## [Unreleased]

### Added
- `packages/architecture/` - New package for BitNet layers (BitLinear, SubLN) and model conversion
  - `BitLinear`: Ternary weight quantization with 8-bit activation quantization
  - `SubLN`: Sub-Layer Normalization for training stability
  - `LambdaWarmup`: Gradual quantization schedule management
  - `convert_model_to_bitnet()`: On-the-fly model conversion
- `packages/distillation/` - Dedicated knowledge distillation package (moved from training)
  - **BitDistill loss**: Logits + attention relation distillation
  - **TCS loss**: Target Concrete Score for DLM (diffusion) students
  - **Teacher backends**: LocalTeacher (HuggingFace), VLLMTeacher (remote)
  - **Muon optimizer support** with separate AdamW params for embeddings
- `packages/training/src/wrinklefree/objectives/` - Composable objectives system with curriculum scheduling
  - `ObjectiveManager`: Combines multiple objectives with weights
  - `CurriculumScheduler`: Adjusts weights during training
- `packages/training/configs/training/unified.yaml` - Unified training config with auto-convert support
- `packages/deployer/` - New distillation commands
  - `wf distill` - Launch BitDistill distillation jobs
  - `wf tcs-distill` - Launch TCS distillation jobs
- On-the-fly BitNet conversion (no separate stage1 step needed)

### Changed
- Renamed `cheapertraining` to `data_handler` (import as `data_handler`, package name `data-handler`)
- Renamed `fairy2` to `distillation` (dedicated distillation package, NOT Fairy2i)
- Renamed `Stage2Trainer` to `ContinuedPretrainingTrainer` (backward-compatible alias preserved)
- Training package now imports from `bitnet_arch` for BitNet components
- Stage 3 distillation moved from training package to distillation package
- All documentation updated to reflect new package names

### Removed
- `packages/fairy2/` - Complex-valued quantization package (Fairy2i)
- `packages/deployer/skypilot/fairy2_*.yaml` - SkyPilot configs for fairy2
- `packages/training/src/wrinklefree/distillation/` - Moved to distillation package
- `packages/training/configs/training/stage3_distill*.yaml` - Moved to distillation package

### Migration Guide

**Package renames:**
- `cheapertraining` → `data_handler`
  - Import: `from data_handler.data import ...`
  - Package: `--package data-handler`
- `fairy2` → `distillation` (different purpose - now for knowledge distillation)

**Stage 3 distillation:**
```bash
# OLD (training package)
uv run --package wrinklefree python scripts/train.py training=stage3_distill

# NEW (distillation package)
uv run --package wrinklefree-distillation python packages/distillation/scripts/distill.py \
  student.checkpoint_path=outputs/stage2/checkpoint.pt
```

**Workspace dependencies:**
```toml
# In pyproject.toml
[tool.uv.sources]
data-handler = { workspace = true }
bitnet-arch = { workspace = true }
```

### Previous Changes
- Root documentation: `docs/architecture.md`, `docs/quick-start.md`, `docs/dependencies.md`, `docs/development.md`
- Monorepo integration sections in all package CLAUDE.md files
- `packages/data_handler/CLAUDE.md` (was missing)
- "Part of WrinkleFree Monorepo" notes in all package READMEs
- Updated all documentation to use monorepo paths (`packages/*`)
- Standardized clone instructions across all READMEs
- Expanded `packages/deployer/CLAUDE.md` with troubleshooting

## [0.1.0] - 2024-12-28

### Migration from Meta-Repo

Converted from meta-repo (7 separate git repositories managed by `meta.js`) to uv workspace monorepo.

#### Breaking Changes

**Directory Structure**:
```
# Old (meta-repo)
WrinkleFree/
├── WrinkleFree-1.58Quant/
├── WrinkleFree-CheaperTraining/
├── WrinkleFree-Fairy2/
├── WrinkleFree-Inference-Engine/
├── WrinkleFree-Eval/
├── WrinkleFree-Deployer/
├── WrinkleFree-DLM-Converter/
└── .meta

# New (monorepo)
WrinkleFree-Monorepo/
├── packages/
│   ├── training/
│   ├── cheapertraining/
│   ├── fairy2/
│   ├── inference/
│   ├── eval/
│   ├── deployer/
│   └── converter/
├── pyproject.toml
└── uv.lock
```

**Command Changes**:
```bash
# Old
meta git clone git@github.com:DeepOpt-com/WrinkleFree.git
cd WrinkleFree-1.58Quant && uv sync

# New
git clone --recurse-submodules git@github.com:DeepOpt-com/WrinkleFree-Monorepo.git
uv sync --all-packages
```

**Package References**:
- `../WrinkleFree-Deployer` → `../deployer`
- `cd WrinkleFree-1.58Quant` → `cd packages/training`

#### Added
- Single `uv.lock` for unified dependency resolution
- Workspace dependencies via `[tool.uv.sources]`
- Root `pyproject.toml` with `[tool.uv.workspace]`

#### Removed
- `.meta` configuration file
- Individual `uv.lock` files per package
- Individual `.git` directories per package
- `package.json` (was used for meta.js)
