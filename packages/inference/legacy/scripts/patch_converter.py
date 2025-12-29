#!/usr/bin/env python3
"""Patch BitNet converter to support BPE tokenizers (like Qwen)"""

import sys

converter_path = sys.argv[1] if len(sys.argv) > 1 else "utils/convert-hf-to-gguf-bitnet.py"

with open(converter_path, "r") as f:
    content = f.read()

# Find the BitnetModel.set_vocab method and patch it
old_code = """    def set_vocab(self):
        self._set_vocab_sentencepiece()"""

new_code = """    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            try:
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                self._set_vocab_gpt2()"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(converter_path, "w") as f:
        f.write(content)
    print("Patched BitnetModel.set_vocab to handle BPE tokenizers")
else:
    print("Already patched or code structure changed")
