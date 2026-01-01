# Test Coverage Baseline Report

Generated: 2025-12-26 (Updated)

> **Note**: Package renames since this baseline:
> - `WrinkleFree-1.58Quant` → `packages/training` (wrinklefree)
> - `WrinkleFree-CheaperTraining` → `packages/data_handler` (data-handler)
> - `WrinkleFree-Fairy2` → `packages/distillation` (wrinklefree-distillation)

## Summary

| Sub-Repo | Tests | Passing | Coverage | Status |
|----------|-------|---------|----------|--------|
| **training** (was 1.58Quant) | 324 | 315 | **36%** | MoE router at 95%, 9 equivalence failures |
| **data_handler** (was CheaperTraining) | 37 | 37 | **~15%** | Data/influence tests added |
| **deployer** | 260 | 236 | **47%** | Source tests added, constants/config 100% |
| **inference** | TBD | TBD | TBD | Submodule conflicts |
| **eval** | TBD | TBD | TBD | Submodule conflicts |
| **converter** | 28 | 28 | **25%** | Source tests added, constants 100% |

**Target**: 70% coverage across all repos

## Key Improvements (2025-12-26)

### Tests Added
- **1.58Quant**: `tests/test_moe.py` - 23 comprehensive MoE router tests
- **Deployer**: `tests/unit/test_wf_deployer.py` - 24 source-importing tests
- **DLM-Converter**: `tests/test_wf_dlm_converter.py` - 19 source-importing tests
- **CheaperTraining**: `tests/unit/test_cheapertraining.py` - 20 data/influence tests

### Coverage Gains
- **Deployer**: 0% → 47% (constants 100%, config 100%)
- **DLM-Converter**: 0% → 25% (constants 100%, __init__ 100%)
- **1.58Quant MoE router**: 0% → 95%

## Detailed Analysis

### WrinkleFree-1.58Quant (37% coverage)

**Well Covered (>80%):**
- `distillation/attention_loss.py` - 93%
- `distillation/layerwise_loss.py` - 95%
- `models/attention.py` - 97%
- `models/ffn.py` - 100%
- `quantization/activation_quant.py` - 100%
- `quantization/ste.py` - 100%
- `quantization/weight_quant.py` - 100%

**Gaps (<30%):**
- `moe/` - 0% (entire MoE module untested)
- `serving/` - 0% (converter, bitnet_wrapper)
- `training/stage1_9.py` - 8%
- `training/stage2.py` - 8%
- `models/llama.py` - 17%
- `models/fp8_bitlinear.py` - 18%

**16 Test Failures:** Mostly dataset config tests that reference outdated configs (MegaMath, etc.)

### WrinkleFree-CheaperTraining (8% coverage)

**Issues:**
- Most code in `_legacy/` directory (0% coverage)
- Active code has minimal tests
- Only `test_sequence_packing.py` provides coverage

**Priority areas:**
- `data/mixing.py` - 27% (needs more tests)
- `influence/` modules - 11-22%
- `training/optimizer.py` - 18%

### WrinkleFree-Deployer (0% coverage)

**Issues:**
- 83 tests pass but don't import `src/wf_deployer`
- Tests in `tests/unit/` test `scripts/` instead of source

**Needs:**
- Tests that import and test `wf_deployer.core`, `wf_deployer.deployer`, etc.
- Mock SkyPilot/Modal for unit testing

### WrinkleFree-DLM-Converter (0% coverage)

**Issues:**
- 9 tests pass but don't import `src/wf_dlm_converter`
- Tests likely testing external functionality

**Needs:**
- Unit tests for `core.py`, `cli.py`
- Mock transformers/model loading
- Test conversion logic

## Infrastructure Added

### Coverage Configuration
Added to all repos' `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src/<package>"]
branch = true
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:", ...]
fail_under = 0
```

### conftest.py Files
Created for repos missing them with:
- Device fixtures (CPU/GPU)
- Mock fixtures (CUDA, HuggingFace, W&B)
- Test data fixtures
- Auto-skip for GPU tests

### GitHub Actions Workflows
Added `.github/workflows/test.yml` to all repos with:
- Python 3.12 + uv setup
- pytest with coverage
- Codecov upload

## Next Steps

### Phase 1: Fix Existing Tests
1. **1.58Quant**: Fix 16 failing tests (update dataset config references)
2. **Deployer/DLM-Converter**: Add tests that actually import source modules

### Phase 2: Add Critical Tests
Priority modules to test:

**1.58Quant:**
- [ ] `moe/` - Router, expert, fake_moe tests
- [ ] `serving/converter.py` - GGUF conversion tests
- [ ] `training/stage1_9.py` - Layer-wise distillation tests
- [ ] `training/stage2.py` - Continue pretraining tests

**CheaperTraining:**
- [ ] `data/mixing.py` - Dataset mixing tests
- [ ] `influence/` - Influence function tests
- [ ] `training/optimizer.py` - Optimizer tests

**Deployer:**
- [ ] `core.py` - train(), logs(), cancel() tests
- [ ] `deployer.py` - Deployment logic tests
- [ ] `modal_deployer.py` - Modal integration tests (mocked)

**DLM-Converter:**
- [ ] `core.py` - convert(), validate() tests
- [ ] `conversion/training.py` - SFT training tests
- [ ] `models/adapter.py` - Model adapter tests

### Phase 3: CI/CD Activation
1. Set up Codecov tokens for each repo
2. Enable coverage gates (initially 0%, increase to 70% over time)
3. Add coverage badges to READMEs

## Running Coverage Locally

```bash
# On Desktop
ssh Desktop "cd /home/lev/code/WrinkleFree/<REPO> && ~/.local/bin/uv run pytest --cov=src/<package> --cov-report=html --cov-report=term-missing"

# View HTML report
open htmlcov/index.html
```

## Mock Patterns Reference

### CUDA Mocking
```python
@pytest.fixture
def mock_cuda(mocker):
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.cuda.device_count", return_value=0)
```

### HuggingFace Model Mocking
```python
@pytest.fixture
def mock_hf_model(mocker):
    mock = mocker.patch("transformers.AutoModel.from_pretrained")
    mock.return_value = MagicMock()
    return mock
```

### SkyPilot Mocking
```python
@pytest.fixture
def mock_skypilot(mocker):
    mocker.patch("sky.launch", return_value=MagicMock(cluster_name="test"))
    mocker.patch("sky.down")
```
