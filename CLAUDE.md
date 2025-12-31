# WrinkleFree Monorepo

1.58-bit quantized LLM research using uv workspaces.

## MUST-DO Rules

1. **SYNC BEFORE REMOTE:** Before ANY ssh/remote command:
   `uv run gcd sync-ssh <host> --smart`

2. **NO GCP GPU:** Use Nebius or RunPod only

3. **NEVER CANCEL OTHERS' JOBS:** Only cancel SkyPilot jobs you started

4. **Read Package CLAUDE.md:** Before modifying a package, read its `packages/<pkg>/CLAUDE.md`

## Quick Commands

| Task | Command |
|------|---------|
| Sync to Desktop | `uv run gcd sync-ssh desktop --smart` |
| Start watch mode | `uv run gcd sync-ssh desktop --watch` |
| Run training | `uv run --package wrinklefree python scripts/train.py` |
| Run distillation | `uv run --package wrinklefree-distillation python scripts/distill.py` |
| Deploy to cloud | `wf train -m smollm2_135m -s 2 --cloud nebius` |

## Package Navigation

| Package | CLAUDE.md | Purpose |
|---------|-----------|---------|
| training | `packages/training/CLAUDE.md` | Stages 1, 1.9, 2 |
| distillation | `packages/distillation/CLAUDE.md` | Stage 3+ |
| architecture | `packages/architecture/CLAUDE.md` | BitLinear/SubLN |
| data_handler | `packages/data_handler/CLAUDE.md` | Data loading |
| inference | `packages/inference/CLAUDE.md` | Model serving |
| deployer | `packages/deployer/CLAUDE.md` | Cloud deploy |

## Reference
For detailed docs (pipeline diagrams, GCP config, troubleshooting): `docs/ai-code/reference.md`
