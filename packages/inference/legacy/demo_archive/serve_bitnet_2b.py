#!/usr/bin/env python3
"""Serve microsoft/BitNet-b1.58-2B-4T with Gradio UI.

Run: uv run python demo/serve_bitnet_2b.py
Open: http://localhost:7860
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download

# Will be imported after we check for tokenizer
tokenizer = None
model = None


def load_tokenizer(repo_id: str):
    """Load tokenizer from HuggingFace."""
    global tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        print(f"Loaded tokenizer: {len(tokenizer)} tokens")
        return True
    except ImportError:
        print("transformers not installed, using simple tokenizer")
        return False


def simple_tokenize(text: str) -> list[int]:
    """Simple character-level tokenizer as fallback."""
    # Map ASCII characters to token IDs
    return [ord(c) % 128256 for c in text]


def simple_detokenize(token_ids: list[int]) -> str:
    """Simple character-level detokenizer as fallback."""
    # Map token IDs back to ASCII (approximate)
    return "".join(chr(t % 256) if t < 256 else "?" for t in token_ids)


def create_demo():
    """Create Gradio demo for BitNet 2B model."""
    global model, tokenizer

    print("=" * 60)
    print("BitNet-b1.58-2B-4T Inference Demo")
    print("=" * 60)

    repo_id = "microsoft/BitNet-b1.58-2B-4T"

    # Load tokenizer
    print("\n[1/2] Loading tokenizer...")
    has_transformers = load_tokenizer(repo_id)

    # Load model
    print("\n[2/2] Loading model...")
    from wrinklefree_inference.models.bitnet import load_model
    model = load_model(repo_id)

    print("\nModel ready for inference!")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden: {model.config.hidden_size}")
    print(f"  Vocab: {model.config.vocab_size}")
    print(f"  Threads: {model.kernel.num_threads()}")

    def generate_response(prompt: str, max_tokens: int, temperature: float):
        """Generate text from prompt."""
        if not prompt.strip():
            return "Please enter a prompt.", "", ""

        # Tokenize
        if tokenizer is not None:
            input_ids = tokenizer.encode(prompt, return_tensors="np")[0]
        else:
            input_ids = np.array(simple_tokenize(prompt))

        # Generate
        start = time.perf_counter()
        generated_ids, tok_per_sec = model.generate(
            input_ids,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        total_time = time.perf_counter() - start

        # Decode
        if tokenizer is not None:
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            output_text = simple_detokenize(generated_ids)

        stats = f"Generated {len(generated_ids)} tokens in {total_time:.2f}s ({tok_per_sec:.1f} tok/s)"
        kernel_info = f"Native kernel: {model.kernel.num_threads()} threads"

        return output_text, stats, kernel_info

    def run_benchmark():
        """Run kernel microbenchmarks."""
        from wrinklefree_inference.kernels.native import (
            pack_weights,
            quantize_activations,
        )

        kernel = model.kernel
        results = []

        # GEMV benchmarks
        for shape in [(2560, 2560), (6912, 2560)]:
            out_f, in_f = shape
            weights = np.random.choice([-1, 0, 1], size=shape).astype(np.float32)
            packed = pack_weights(weights)
            act = np.random.randn(in_f).astype(np.float32)
            act_i8, scale = quantize_activations(act)

            # Warmup
            for _ in range(5):
                kernel.gemv(packed, act_i8, scale)

            # Benchmark
            times = []
            for _ in range(50):
                start = time.perf_counter()
                kernel.gemv(packed, act_i8, scale)
                times.append(time.perf_counter() - start)

            avg_ms = np.mean(times) * 1000
            results.append(f"GEMV {out_f}x{in_f}: {avg_ms:.3f}ms")

        # GEMM benchmarks
        out_f, in_f = 2560, 2560
        weights = np.random.choice([-1, 0, 1], size=(out_f, in_f)).astype(np.float32)
        packed = pack_weights(weights)

        for batch in [1, 8, 32, 64]:
            act = np.random.randn(batch, in_f).astype(np.float32)
            act_i8, scale = quantize_activations(act)

            # Warmup
            for _ in range(3):
                kernel.gemm(packed, act_i8, scale)

            times = []
            for _ in range(20):
                start = time.perf_counter()
                kernel.gemm(packed, act_i8, scale)
                times.append(time.perf_counter() - start)

            avg_ms = np.mean(times) * 1000
            throughput = batch / np.mean(times)
            results.append(f"GEMM batch={batch:2d}: {avg_ms:.2f}ms ({throughput:,.0f} tok/s)")

        return "\n".join(results)

    # Create interface
    with gr.Blocks(title="BitNet-2B Demo") as demo:
        gr.Markdown("""
        # BitNet-b1.58-2B-4T Inference

        This demo runs the **microsoft/BitNet-b1.58-2B-4T** model with native AVX2 kernels.

        **Model**: 2B parameters with 1.58-bit ternary weights (-1, 0, +1)
        **Compression**: ~10x vs FP16 (400MB vs 4GB)
        **Kernel**: OpenMP-parallelized C++ with AVX2 auto-vectorization
        """)

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3,
                        value="The quick brown fox",
                    )
                    with gr.Row():
                        max_tokens = gr.Slider(10, 200, value=50, step=10, label="Max Tokens")
                        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column(scale=2):
                    output_text = gr.Textbox(label="Generated Output", lines=5)
                    stats_text = gr.Textbox(label="Performance Stats")
                    kernel_text = gr.Textbox(label="Kernel Info")

            generate_btn.click(
                generate_response,
                inputs=[prompt_input, max_tokens, temperature],
                outputs=[output_text, stats_text, kernel_text],
            )

        with gr.Tab("Benchmark"):
            gr.Markdown("Run kernel microbenchmarks to see raw GEMV/GEMM performance.")
            benchmark_btn = gr.Button("Run Benchmark", variant="primary")
            benchmark_output = gr.Textbox(label="Benchmark Results", lines=12)

            benchmark_btn.click(run_benchmark, outputs=benchmark_output)

        with gr.Tab("About"):
            gr.Markdown(f"""
            ## Technical Details

            - **Model**: microsoft/BitNet-b1.58-2B-4T
            - **Parameters**: 2B
            - **Layers**: {model.config.num_hidden_layers}
            - **Hidden Size**: {model.config.hidden_size}
            - **Vocab Size**: {model.config.vocab_size}
            - **Weight Format**: 1.58-bit ternary ({{-1, 0, +1}}), packed 4 per byte
            - **Kernel**: Native C++ with OpenMP ({model.kernel.num_threads()} threads)

            ## Memory Usage

            - **Weights**: ~400MB (packed ternary)
            - **Embeddings**: ~660MB (FP16)
            - **Total**: ~1.1GB

            ## Performance

            The native kernel provides significant speedup through:
            - Efficient 2-bit unpacking
            - OpenMP parallel for loops
            - AVX2 auto-vectorization with -march=native
            - Cache-friendly memory access patterns

            ## Source

            [WrinkleFree-Inference-Engine](https://github.com/DeepOpt-com/WrinkleFree-Inference-Engine)
            """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
