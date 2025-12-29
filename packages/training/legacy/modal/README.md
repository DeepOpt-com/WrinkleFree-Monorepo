# Legacy Modal Code

This directory contains deprecated Modal training scripts.

**Status:** Deprecated (December 2025)

**Reason:** Training now runs via WrinkleFree-Deployer on SkyPilot.

## Files

- `modal_train.py` - Self-contained Modal training script

## Migration

Use the `wf` CLI from WrinkleFree-Deployer:

```bash
wf train -m qwen3_4b -s 2
```
