# Legacy Modal Code

This directory contains deprecated Modal deployment code.

**Status:** Deprecated (December 2025)

**Reason:** Migrated to SkyPilot-only backend for:
- Better spot instance recovery
- Multi-cloud support
- Unified infrastructure management

## Files

- `modal_deployer.py` - Former Modal backend implementation

## Migration

Use the SkyPilot backend instead:

```bash
wf train -m qwen3_4b -s 2
```

See `skypilot/train.yaml` for the current training configuration.
