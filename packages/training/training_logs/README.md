# Training Audit Logs

This folder contains **persistent warnings and audit logs** from training runs.

## Purpose

Unlike runtime logs (excluded from git via `.gitignore`), these logs provide a **permanent audit trail** for:

- **Dirty git state** during training - tracks when runs were done with uncommitted changes
- **GCS authentication failures** - documents credential issues
- **Checkpoint corruption events** - records when checkpoints fail to load
- **Config mismatches on resume** - flags when resuming with different configs
- **Training interruptions and failures** - documents unexpected terminations

## Why Commit These Logs?

These logs are **intentionally committed to git** because:

1. **Accountability**: Know exactly which runs had uncommitted code
2. **Debugging**: Track down why production runs failed
3. **Reproducibility Auditing**: Identify runs that may not be reproducible
4. **Team Visibility**: Share critical warnings across team members

## Log Format

Each warning is a **separate JSON file** with schema:

```json
{
  "timestamp": "2025-01-15T10:30:00.123456",
  "type": "dirty_git",
  "severity": "WARNING",
  "fingerprint": "abc123def456...",
  "git_commit": "789abc012...",
  "message": "Training with uncommitted changes",
  "recommendation": "Commit changes before production runs"
}
```

### Warning Types

| Type | Severity | Description |
|------|----------|-------------|
| `dirty_git` | WARNING | Repository has uncommitted changes |
| `credentials_missing` | CRITICAL | GCS/WandB credentials not found |
| `gcs_auth_failed` | CRITICAL | GCS operation failed |
| `checkpoint_corrupted` | ERROR | Checkpoint file invalid |
| `resume_config_mismatch` | WARNING | Resuming with different config |
| `training_interrupted` | INFO | Training stopped (Ctrl+C) |
| `training_failed` | ERROR | Training crashed with exception |

## Retention Policy

Logs are automatically rotated:

- **Max files**: 100 per type (oldest deleted)
- **Max age**: 90 days
- **Total size limit**: ~10MB (enforced by file count)

## Best Practices

1. **Review `dirty_git` warnings** before production runs
2. **Commit code changes** before final training
3. **Investigate repeated `credentials_missing`** warnings
4. **Check `checkpoint_corrupted`** if runs restart unexpectedly
5. **Use `--skip-recovery`** flag if you intentionally want to skip resume

## File Naming

Files are named: `{timestamp}_{type}_{fingerprint}.json`

Example: `2025-01-15_10-30-00_dirty_git_abc123de.json`

This allows sorting by date and filtering by type using shell commands:

```bash
# List all dirty git warnings
ls training_logs/warnings/*dirty_git*.json

# Find recent credential issues
ls -la training_logs/warnings/*credentials_missing*.json | tail -5

# Count warnings by type
for type in dirty_git gcs_auth_failed checkpoint_corrupted; do
  echo "$type: $(ls training_logs/warnings/*${type}*.json 2>/dev/null | wc -l)"
done
```
