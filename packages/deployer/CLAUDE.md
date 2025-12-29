# WrinkleFree-Deployer

Training job launcher for 1.58-bit quantized LLMs. Uses SkyPilot for managed GPU jobs with spot recovery.

**For detailed AI discovery docs, see `docs/AIDEV.md`.**

## Quick Reference

```bash
# Launch training
wf train -m qwen3_4b -s 2

# With specific scale (4x H100)
wf train -m qwen3_4b -s 2 --scale large

# Check logs
wf logs <run_id>

# List recent runs
wf runs
```

## Key Files

| File | Purpose |
|------|---------|
| `src/wf_deployer/constants.py` | All magic strings, defaults, scales |
| `src/wf_deployer/core.py` | Main API: train(), logs(), cancel() |
| `src/wf_deployer/cli.py` | CLI commands |
| `skypilot/train.yaml` | SkyPilot training job template |
| `skypilot/service.yaml` | SkyServe inference template |
