# Dependency Graph

## Package Dependencies

```
data_handler (library)
    │
    ├──► training (wrinklefree)
    │       Dependencies: torch, transformers, hydra-core, datasets, wandb
    │       Uses: data_handler.data, data_handler.influence
    │
    └──► distillation (wrinklefree-distillation)
            Dependencies: torch, transformers, hydra-core, vllm (optional)
            Uses: data_handler.data, data_handler.influence

architecture (library)
    │
    └──► training (wrinklefree)
            Uses: bitnet_arch.layers, bitnet_arch.conversion

inference
    │   Dependencies: sglang, torch, transformers
    │   External: sglang-bitnet submodule
    │
    └──► eval
            Dependencies: transformers, datasets
            Uses: inference for model loading

deployer
    │   Dependencies: modal, skypilot, typer
    │   Orchestrates: training, distillation, inference, eval
    │
    └──► References all other packages via cloud deployment

converter
        Dependencies: torch, transformers, safetensors
        Uses: data_handler.data
```

## Workspace Dependencies

Packages that import other packages must declare workspace sources:

```toml
# packages/training/pyproject.toml
[project]
dependencies = [
    "data-handler",  # Shared data library
    "bitnet-arch",   # BitNet layers & conversion
    # ... other deps
]

[tool.uv.sources]
data-handler = { workspace = true }
bitnet-arch = { workspace = true }
```

```toml
# packages/distillation/pyproject.toml
[project]
dependencies = [
    "data-handler",  # Shared data library
    # ... other deps
]

[tool.uv.sources]
data-handler = { workspace = true }
```

## External Submodules

| Submodule | Path | Source | Purpose |
|-----------|------|--------|---------|
| BitNet | `extern/BitNet/` | github.com/microsoft/BitNet | C++ inference engine |
| sglang-bitnet | `packages/inference/extern/sglang-bitnet/` | Fork of sglang | SGLang with BitNet support |

### Updating Submodules

```bash
# Update all submodules
git submodule update --remote --merge

# Update specific submodule
cd extern/BitNet
git pull origin main
cd ../..
git add extern/BitNet
git commit -m "Update BitNet submodule"
```

## Key External Dependencies

### Training Stack

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.5.0 | Core training |
| transformers | >=4.40.0 | Model architectures |
| datasets | >=2.0.0 | Data loading |
| hydra-core | >=1.3.0 | Configuration |
| wandb | >=0.16.0 | Experiment tracking |
| bitsandbytes | >=0.42.0 | 8-bit optimizers |

### Distillation Stack

| Package | Version | Purpose |
|---------|---------|---------|
| vllm | >=0.6.0 | Remote teacher backend (optional) |
| safetensors | >=0.4.0 | Model serialization |
| google-cloud-storage | >=2.0.0 | GCS checkpoint access |

### Inference Stack

| Package | Version | Purpose |
|---------|---------|---------|
| sglang | Custom fork | Serving framework |
| safetensors | >=0.4.0 | Model serialization |
| streamlit | >=1.30.0 | Demo UI |

### Deployment Stack

| Package | Version | Purpose |
|---------|---------|---------|
| modal | >=0.60.0 | Serverless GPU |
| skypilot | >=0.6.0 | Cloud orchestration |
| typer | >=0.9.0 | CLI framework |

## Dependency Conflicts

Known version constraints:
- `transformers>=4.57.3` required by inference (for sglang compatibility)
- `torch>=2.5.0` required for BF16 training stability
- Avoid `muon-clip` in benchmark extras (pins transformers==4.53.0)

## Lockfile

Single `uv.lock` at repo root resolves all dependencies across packages. This ensures:
- Consistent versions across all packages
- No duplicate installations
- Reproducible environments
