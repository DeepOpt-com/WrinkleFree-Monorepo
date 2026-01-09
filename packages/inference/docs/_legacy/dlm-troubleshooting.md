# DLM Troubleshooting Guide

Common issues and solutions for DLM (Diffusion Language Model) inference.

## Quick Diagnosis

### "Model does not appear to be DLM-trained"

**Symptom**: Server fails to start or logs indicate missing mask token.

**Causes**:
1. Model was not trained with DLM objective
2. Mask token not in vocabulary

**Solutions**:
```bash
# Check if model has mask token in tokenizer
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('models/dlm-checkpoint')
print('Mask tokens:', [t for t in tok.get_vocab() if 'MASK' in t.upper()])
"

# Override mask token ID if auto-detection fails
./dlm_server --model model.gguf --mask-token-id 0
```

### Gibberish / Garbage Output

**Symptom**: Output is repetitive characters like "GGGGG..." or nonsensical.

**Causes**:
1. Wrong GGUF format (TQ2_0 with bf16 checkpoint)
2. Wrong converter used
3. Mask token ID mismatch between training and inference

**Solutions**:
```bash
# Re-convert with correct format (I2_S recommended)
python scripts/convert_checkpoint_to_gguf.py \
    models/dlm-checkpoint --outfile model.gguf --outtype i2_s

# Verify mask_token_id matches training config
# Check training config: objectives.dlm.mask_token_id
# Check inference: server logs should show detected mask token
```

### Slow Inference (Not ~2.5x Faster)

**Symptom**: DLM inference is similar speed or slower than autoregressive.

**Causes**:
1. Block size too small
2. Threshold too low (causes many iterations per block)
3. DualCache disabled
4. Note: ~2.5x is theoretical, actual speedup varies

**Solutions**:
```bash
# Increase block size (more parallelism)
./dlm_server --model model.gguf --block-size 64

# Increase threshold (fewer iterations, but may reduce quality)
./dlm_server --model model.gguf --threshold 0.98

# Verify DualCache is enabled (default)
# Check server logs for "DualCache enabled"
```

### Shape Mismatch Errors

**Symptom**: "expected 2560, 2560, got 660, 2560" or similar.

**Cause**: Packed 2-bit weights not unpacked during conversion.

**Solution**: Use the inference package converter:
```bash
python scripts/convert_checkpoint_to_gguf.py \
    models/dlm-checkpoint --outfile model.gguf --outtype i2_s
```

### "tokenizer not found" Error

**Symptom**: Conversion fails with missing tokenizer.

**Solution**:
```bash
# Ensure tokenizer files are present
ls models/dlm-checkpoint/tokenizer*

# Download if missing
huggingface-cli download organization/model tokenizer.json \
    --local-dir models/dlm-checkpoint/
```

### "BitnetForCausalLM not found" Error

**Symptom**: llama.cpp or conversion fails to recognize model architecture.

**Solution**:
```bash
# Fix architecture name (capital N -> lowercase n)
sed -i 's/BitNetForCausalLM/BitnetForCausalLM/g' models/dlm-checkpoint/config.json
```

## Validation Checklist

Before deploying a DLM model, verify:

- [ ] Model was trained with `objectives.dlm.enabled=true`
- [ ] `config.json` architecture is `BitnetForCausalLM` (lowercase n)
- [ ] GGUF conversion used Microsoft's converter (not standard llama.cpp)
- [ ] GGUF format is TQ1_0 or I2_S (NOT TQ2_0 for bf16 checkpoints)
- [ ] Server logs show "Detected mask token: X"
- [ ] Test output is coherent (not repeated characters)
- [ ] `mask_token_id` matches training config

## Debug Commands

```bash
# Check GGUF metadata (requires llama.cpp)
./extern/llama.cpp/build/bin/llama-cli \
    -m model.gguf --show-info

# Test single inference
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'

# Check server health
curl http://localhost:30000/health

# View server logs (if using pm2)
pm2 logs dlm-server
```

## Getting Help

If issues persist:
1. Check server logs for detailed error messages
2. Verify checkpoint files are complete (all .safetensors, config.json)
3. Compare training config `mask_token_id` with inference detection
4. Try with a known-working checkpoint first

See also:
- [DLM Pipeline Guide](dlm-pipeline.md)
- [GGUF Conversion Guide](gguf-conversion.md)
