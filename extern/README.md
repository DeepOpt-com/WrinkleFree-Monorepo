# External Dependencies

## BitNet/
Microsoft BitNet inference engine, vendored from upstream.

**Upstream:** https://github.com/microsoft/BitNet (commit 404980e)

**Local modifications:**
- `utils/convert-hf-to-gguf-bitnet.py`: Fixed ternary quantization (np.sign() -> proper online quant), added 2-bit weight unpacking, model alias support

## reference/BitNet.cpp/
Reference copy for comparison/backup.
