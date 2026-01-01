# Changelog

All notable changes to the WrinkleFree monorepo.

## [Unreleased]

### Added
- `packages/architecture/` - BitNet layers package
  - `BitLinear`: Ternary weight quantization with 8-bit activation quantization
  - `BitLinearLRC`: Low-Rank Correction layer for post-quantization recovery
  - `SubLN`: Sub-Layer Normalization for training stability
  - `LambdaWarmup`: Gradual quantization schedule management
  - `convert_model_to_bitnet()`: On-the-fly model conversion
  - `convert_bitlinear_to_lrc()`: Convert BitLinear to BitLinearLRC
- `packages/training/src/wrinklefree/objectives/` - Composable objectives system
  - `ObjectiveManager`: Combines multiple objectives with weights
  - `CurriculumScheduler`: Phase-based weight transitions
  - `logits_distill.py`: KL divergence on teacher logits
  - `attention_distill.py`: Attention relation matching
  - `tcs_distill.py`: Target Concrete Score for DLM students
  - `bitdistill.py`: Combined BitDistill (logits + attention)
  - `lrc_reconstruction.py`: Low-Rank Correction training
- New training configs:
  - `training=bitdistill_full` - Full BitDistill with curriculum
  - `training=lrc_calibration` - LRC adapter training
- `packages/mobile/` - Android inference with BitNet.cpp

### Changed
- **Distillation integrated into training package** via objectives system
  - Legacy `distillation` package moved to `packages/_legacy/distillation/`
  - Use `training=bitdistill_full` instead of separate distillation commands
- Legacy packages archived to `packages/_legacy/`:
  - `distillation/` - Knowledge distillation (now integrated into training)
  - `converter/` - DLM conversion (functionality distributed)
  - `cheapertraining/` - Renamed to `data_handler`
- Updated all documentation to reflect integrated distillation
- Removed `wf distill` and `wf tcs-distill` commands (use training objectives)

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
