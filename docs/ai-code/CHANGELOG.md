# AI Code Changes Log

Development changes tracked by AI assistance.

---

## 2025-12-26

### Bug Fixes

#### GCS Bucket Config Lookup (Critical)
**File:** `WrinkleFree-1.58Quant/scripts/train.py`

**Problem:** Stage 2 training was starting with random weights (loss ~17) instead of loading the Stage 1.9 checkpoint from GCS (should start at loss ~1.38).

**Root Cause:** The `get_gcs_bucket()` function was looking for GCS config in the wrong location:
- Checked: `cfg.checkpoint.gcs`, `cfg.training.checkpoint.gcs` (legacy paths)
- Actual location: `cfg.gcs` (top-level in config.yaml)

**Fix:** Updated `get_gcs_bucket()` to check `cfg.get("gcs", {})` first:
```python
def get_gcs_bucket(cfg: DictConfig) -> str | None:
    # Check environment variable first
    bucket = os.environ.get("GCS_BUCKET")
    if bucket:
        return bucket

    # Check top-level gcs config (preferred location)
    gcs_config = cfg.get("gcs", {})
    if gcs_config.get("enabled", False):
        return gcs_config.get("bucket")

    # Legacy paths (fallback)
    ...
```

#### Added Debug Prints for Checkpoint Loading
**File:** `WrinkleFree-1.58Quant/scripts/train.py`

Added print statements to Stage 2 checkpoint loading (lines 507-564) since Hydra logging is disabled:
- `[DEBUG Stage 2] Looking for Stage 1.9 checkpoint...`
- `[DEBUG Stage 2] gcs_bucket=..., gcs_prefix=...`
- `[DEBUG Stage 2] stage1_9_path=...`
- `[DEBUG Stage 2] Loading checkpoint from: ...`
- `[DEBUG Stage 2] ✓ Loaded Stage 1.9 checkpoint: X missing, Y unexpected keys`

### Smoke Test (Nebius 8x H100)

**Config:** `WrinkleFree-Deployer/skypilot/smoke_test_nebius.yaml`

Successfully running Stage 2 training on Nebius with:
- 8x H100 80GB GPUs
- SmolLM2-135M model
- Mixed pretrain dataset with influence-based data selection
- GCS checkpointing enabled
- W&B logging active

**Training Progress (as of writing):**
- Step 289/1000
- Loss: 4.1 (started at ~5.4)
- PPL: 61.8
- Speed: ~1 step/sec

### Technical Notes

1. **Hydra Logging:** Disabled in config.yaml, use `print()` for debug output
2. **RunManager vs Checkpoint Discovery:**
   - RunManager (fingerprint-based): Resuming same run
   - `get_or_download_checkpoint()`: Stage-to-stage handoff
3. **GCS Checkpoint Path:**
   ```
   gs://wrinklefree-checkpoints/checkpoints/{experiment_name}/stage1_9_checkpoint/checkpoints/final/checkpoint.pt
   ```

### Files Modified

| File | Changes |
|------|---------|
| `WrinkleFree-1.58Quant/scripts/train.py` | Fixed `get_gcs_bucket()`, added debug prints |
| `WrinkleFree-Deployer/skypilot/smoke_test_nebius.yaml` | 8x H100 smoke test config |

### Next Steps

1. Verify Stage 2 completes and checkpoints to GCS
2. Verify Stage 3 loads Stage 2 checkpoint
3. Create `sky_deployer.py` SkyPilot backend
4. Archive Modal code to `_deprecated/`
