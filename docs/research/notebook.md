## 12-22-2025

### WrinkleFree-Eval: Remote Evaluation Support

**Added Features**:
- W&B logging support in `evaluate()` API (`wandb_project`, `wandb_run_id`, `wandb_run_name` params)
- CLI arguments for W&B logging (`--wandb-project`, `--wandb-run-id`, `--wandb-run-name`)
- GCS upload script for results (`scripts/upload_results.py`)
- Optional dependency groups: `wandb`, `gcs`, `all`
- Remote evaluation via WrinkleFree-Deployer (`skypilot/eval.yaml`)

**Bug Fixes**:
- Renamed `scripts/evaluate.py` → `scripts/run_eval.py` to avoid circular import with HuggingFace `evaluate` package

**Known Issues - SkyPilot in Docker/Sandbox**:
1. **Read-only ~/.ssh directory**: Sandbox environments may mount `~/.ssh/` as read-only, causing SkyPilot to fail when updating SSH config after cluster launch
2. **Workaround**: Copy `~/.sky/*` to `/tmp/sky_home/`, set `HOME=/tmp/sky_home`, or use direct SSH with credentials from `~/.sky/clients/*/ssh/sky-key`
3. **Spot instance preemption**: Use `sky jobs launch` for automatic recovery (requires jobs controller with 40GB disk limit on RunPod)

**lm-eval API Compatibility (v0.4+)**:
1. Task names simplified: `glue_sst2` → `sst2`, `glue_mnli` → `mnli`, etc.
2. Custom tasks: Use `TaskManager(include_path="...", include_defaults=True)` instead of `lm_eval.tasks.include_path()`
3. CNN/DailyMail summarization requires `unitxt` package - removed from default benchmarks

**Full Evaluation Results - SmolLM2-135M (HuggingFaceTB/SmolLM2-135M)**:
| Task | Accuracy | Samples |
|------|----------|---------|
| SST-2 | **0.6732** | 872 |
| QNLI | **0.4931** | 5,463 |
| MNLI | **0.3410** | 9,815 |

*Evaluated on RTX 4090, bfloat16, batch_size=64 (auto)*

**Example - Direct cluster access after launch failure**:
```bash
# Get cluster info from SkyPilot database
python -c "
import sqlite3, pickle
conn = sqlite3.connect('~/.sky/state.db')
handle = pickle.loads(conn.execute('SELECT handle FROM clusters WHERE name=\"cluster-name\"').fetchone()[0])
print(f'IP: {handle.head_ip}, Port: {handle.head_ssh_port}')
"

# SSH directly
ssh -i ~/.sky/clients/*/ssh/sky-key -p <port> root@<ip>
```

---

## 12-21-2025

### Bug Fix: BitNetLlamaForSequenceClassification lm_head crash

**Problem**: `BitNetLlamaForSequenceClassification` sets `self.model.lm_head = None` but `BitNetLlama.forward()` would crash trying to call `self.lm_head(hidden_states)`.

**Fix**: Modified `BitNetLlama.forward()` to handle `lm_head is None` case - returns hidden states directly when no LM head is present.

---

### Known Issue: position_ids RoPE Shape Mismatch in WrinkleFree-1.58Quant

**Problem**: When `position_ids` is provided to `BitNetAttention.forward()`, the RoPE frequency indexing produces wrong shapes.

**Root Cause**: In `attention.py:176`, when `position_ids` has shape `(batch, seq_len)`, `self.freqs_cis[position_ids]` returns `(batch, seq_len, head_dim//2)`. However, `apply_rotary_emb()` expects `(seq_len, head_dim//2)` and does its own broadcasting.

**Workaround**: Don't pass explicit `position_ids` - let the module auto-generate sequential positions (which works correctly).

**Status**: Test skipped, needs fix in attention module.

---

### Bug Fix: Missing haar.py in WrinkleFree-1.58Quant

**Problem**: The `wrinklefree.quantization.haar` module was missing, causing all tests to fail with `ModuleNotFoundError`.

**Root Cause**: The `__init__.py` and `haar_triton.py` files were importing from `wrinklefree.quantization.haar` but the file was never created.

**Fix**: Created `/src/wrinklefree/quantization/haar.py` with pure PyTorch implementations of:
- `haar_transform_1d_row` - Forward Haar transform
- `inverse_haar_transform_1d_row` - Inverse Haar transform
- `haar_weight_quantization` - Full Haar wavelet quantization
- `haar_weight_quantization_no_scale` - Returns raw quantized values + scale

---

## 12-18-2025