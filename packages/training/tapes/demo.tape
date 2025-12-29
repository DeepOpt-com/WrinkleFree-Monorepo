# VHS Demo Tape for WrinkleFree BitNet Training
# Run with: vhs tapes/demo.tape

Output demo.gif

Set FontSize 14
Set Width 1200
Set Height 800
Set Theme "Monokai"

Type "# WrinkleFree - BitNet 1.58-bit LLM Training"
Enter
Sleep 1s

Type "# Show project structure"
Enter
Type "tree -L 2 --dirsfirst"
Enter
Sleep 3s

Type "# Check dependencies in pyproject.toml"
Enter
Type "head -30 pyproject.toml"
Enter
Sleep 3s

Type "# View BitNet submodule"
Enter
Type "ls extern/BitNet/"
Enter
Sleep 2s

Type "# Core BitLinear layer (ternary quantization)"
Enter
Type "head -60 src/wrinklefree/models/bitlinear.py"
Enter
Sleep 4s

Type "# BitDistill combined loss"
Enter
Type "head -50 src/wrinklefree/distillation/combined_loss.py"
Enter
Sleep 3s

Type "# View training configuration"
Enter
Type "cat configs/training/stage3_distill.yaml"
Enter
Sleep 3s

Type "# Run tests"
Enter
Type "uv run pytest tests/ -v --tb=short"
Enter
Sleep 5s

Type "# Training command example (not executed)"
Enter
Type "echo 'uv run python scripts/train.py model=llama_7b training=stage3_distill'"
Enter
Sleep 2s

Type "# Project ready for BitNet 1.58-bit training!"
Enter
Sleep 2s
