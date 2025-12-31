#!/usr/bin/env python3
"""Ax Bayesian optimization for Stage 1.9 (layer-wise distillation) hyperparameters.

Usage:
    uv run python scripts/benchmark_stage1_9.py --num-trials 10
    uv run python scripts/benchmark_stage1_9.py --num-trials 20 --model smollm2_135m
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.core.metrics import BenchmarkMetrics
from benchmark.core.memory import MemoryTracker, clear_memory
from benchmark.optimization.ax_client import BenchmarkAxClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Stage19RunnerConfig:
    """Configuration for Stage 1.9 benchmark runner."""

    warmup_steps: int = 10
    measurement_steps: int = 200
    sequence_length: int = 512
    device: str = "cuda"
    total_tokens: int = 10_000_000


class Stage19BenchmarkRunner:
    """Benchmark runner for Stage 1.9 layer-wise distillation."""

    def __init__(
        self,
        runner_config: Stage19RunnerConfig,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        stage1_checkpoint: Path | None = None,
    ):
        self.runner_config = runner_config
        self.model_name = model_name
        self.stage1_checkpoint = stage1_checkpoint
        self.device = torch.device(runner_config.device if torch.cuda.is_available() else "cpu")
        self.memory_tracker = MemoryTracker(self.device)

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run_trial(self, trial_params: dict[str, Any], trial_id: int = 0) -> BenchmarkMetrics:
        """Run a Stage 1.9 benchmark trial."""
        logger.info(f"Trial {trial_id}: {trial_params}")

        clear_memory()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Build student model
            student_model = self._build_student_model().to(self.device)

            # Build teacher model
            from wrinklefree.distillation import HiddenStateTeacherWrapper
            teacher = HiddenStateTeacherWrapper(
                model_name_or_path=self.model_name,
                device=self.device,
                load_in_fp16=True,
                load_in_4bit=trial_params.get("teacher_4bit", False),
            )

            # Create optimizer
            optimizer = self._create_optimizer(student_model, trial_params)

            # Create dataloader
            dataloader = self._create_dataloader(trial_params)

            # Create layerwise loss config
            layerwise_config = OmegaConf.create({
                "loss_type": trial_params.get("loss_type", "mse_normalized"),
                "temperature": trial_params.get("temperature", 1.0),
                "normalize": True,
                "hidden_size": student_model.config.hidden_size,
                "vocab_size": student_model.config.vocab_size,
                "lm_loss_weight": trial_params.get("lm_loss_weight", 0.0),
            })

            # Create training config
            batch_size = trial_params.get("batch_size", 8)
            config = OmegaConf.create({
                "max_steps": self.runner_config.measurement_steps,
                "batch_size": batch_size,
                "max_seq_length": self.runner_config.sequence_length,
                "gradient_accumulation_steps": trial_params.get("gradient_accumulation_steps", 1),
                "gradient_clipping": 1.0,
                "log_interval": 50,
                "eval_interval": 99999,
                "save_interval": 99999,
                "total_tokens": self.runner_config.total_tokens,
            })

            # Create trainer (Stage19Trainer merged into Stage2Trainer)
            from wrinklefree.training.continued_pretraining import ContinuedPretrainingTrainer as Stage2Trainer
            # Convert layerwise_config to pre_stage_2 format
            pre_stage_2_config = OmegaConf.create({
                "enabled": True,
                "teacher": {"fp16": True, "offload_to_cpu": False, "load_in_4bit": False},
                "layerwise": layerwise_config,
                "distill_schedule": {"enabled": False},
            })
            trainer = Stage2Trainer(
                model=student_model,
                optimizer=optimizer,
                train_dataloader=dataloader,
                config=config,
                teacher=teacher,
                pre_stage_2_config=pre_stage_2_config,
                device=self.device,
                rank=0,
                world_size=1,
            )
            trainer.output_dir = Path("/tmp/benchmark_stage1_9")

            # Run training
            start_time = time.time()
            metrics_dict = trainer.train()
            wall_time = time.time() - start_time

            # Extract metrics
            initial_loss = trainer.train_losses[0] if trainer.train_losses else 0.0
            final_loss = trainer.train_losses[-1] if trainer.train_losses else 0.0
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

            # Compute throughput
            seq_len = self.runner_config.sequence_length
            num_steps = self.runner_config.measurement_steps
            total_tokens = num_steps * batch_size * seq_len
            throughput = total_tokens / wall_time

            metrics = BenchmarkMetrics.compute(
                throughput_tokens_per_sec=throughput,
                peak_memory_gb=peak_memory_gb,
                allocated_memory_gb=torch.cuda.memory_allocated() / 1e9,
                final_loss=final_loss,
                initial_loss=initial_loss,
                num_steps=num_steps,
                grad_norms=[],
                optimizer_type=trial_params.get("optimizer_type", "adamw_8bit"),
                batch_size=batch_size,
                learning_rate=trial_params.get("learning_rate", 1e-4),
                gradient_accumulation_steps=trial_params.get("gradient_accumulation_steps", 1),
                wall_time_seconds=wall_time,
                trial_id=trial_id,
            )

            logger.info(
                f"Trial {trial_id}: convergence={metrics.convergence_per_sec_per_gb:.4f}, "
                f"loss {initial_loss:.4f} -> {final_loss:.4f}"
            )
            return metrics

        finally:
            if "student_model" in locals():
                del student_model
            if "teacher" in locals():
                del teacher
            clear_memory()

    def _build_student_model(self):
        """Build BitNet student model from Stage 1 checkpoint."""
        from transformers import AutoModelForCausalLM
        from wrinklefree.training.stage1 import convert_model_to_bitnet

        if self.stage1_checkpoint and self.stage1_checkpoint.exists():
            logger.info(f"Loading Stage 1 checkpoint from {self.stage1_checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            hidden_size = model.config.hidden_size
            intermediate_size = model.config.intermediate_size
            model = convert_model_to_bitnet(model, hidden_size, intermediate_size)

            from safetensors.torch import load_file
            state_dict = load_file(self.stage1_checkpoint / "model.safetensors")
            model.load_state_dict(state_dict, strict=False)
            model.gradient_checkpointing_enable()
            return model.to(torch.bfloat16)
        else:
            logger.info(f"Converting {self.model_name} to BitNet on-the-fly")
            from wrinklefree.training.stage1 import run_stage1
            model, _ = run_stage1(
                pretrained_model_name=self.model_name,
                output_dir=Path("/tmp/stage1_checkpoint"),
            )
            return model

    def _create_dataloader(self, trial_params: dict):
        """Create dataloader for Stage 1.9 training."""
        from wrinklefree.data.pretrain_dataset import create_pretrain_dataloader

        batch_size = trial_params.get("batch_size", 8)
        return create_pretrain_dataloader(
            dataset_path="HuggingFaceFW/fineweb-edu",
            dataset_name="sample-10BT",
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_length=self.runner_config.sequence_length,
            num_workers=2,
            seed=42,
            packed=True,
        )

    def _create_optimizer(self, model, trial_params: dict):
        """Create optimizer based on trial parameters."""
        optimizer_type = trial_params.get("optimizer_type", "adamw_8bit")
        learning_rate = trial_params.get("learning_rate", 1e-4)
        weight_decay = trial_params.get("weight_decay", 0.1)

        # Separate params for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if optimizer_type == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(param_groups, lr=learning_rate, betas=(0.9, 0.95))
            except ImportError:
                pass

        return torch.optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))


def create_stage19_search_space() -> DictConfig:
    """Create search space for Stage 1.9 optimization."""
    return OmegaConf.create({
        "parameters": {
            "optimizer_type": {
                "type": "choice",
                "values": ["adamw_8bit", "adamw"],
                "is_ordered": False,
            },
            "learning_rate": {
                "type": "range",
                "bounds": [1e-5, 1e-3],
                "log_scale": True,
                "value_type": "float",
            },
            "batch_size": {
                "type": "choice",
                "values": [4, 8, 16, 32],
                "is_ordered": True,
            },
            "loss_type": {
                "type": "choice",
                "values": ["mse_normalized", "cosine", "kl"],
                "is_ordered": False,
            },
            "lm_loss_weight": {
                "type": "range",
                "bounds": [0.0, 0.5],
                "log_scale": False,
                "value_type": "float",
            },
            "temperature": {
                "type": "range",
                "bounds": [1.0, 4.0],
                "log_scale": False,
                "value_type": "float",
            },
        },
    })


def run_optimization(
    num_trials: int = 10,
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    stage1_checkpoint: Path | None = None,
    output_dir: Path = Path("./benchmark_results/stage1_9"),
    measurement_steps: int = 200,
) -> None:
    """Run Ax optimization for Stage 1.9 hyperparameters."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create search space
    search_space = create_stage19_search_space()

    # Create Ax client
    ax_client = BenchmarkAxClient(
        search_space_config=search_space,
        experiment_name=f"stage1_9_{model_name.split('/')[-1]}",
    )

    # Create runner
    runner_config = Stage19RunnerConfig(
        measurement_steps=measurement_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    runner = Stage19BenchmarkRunner(
        runner_config=runner_config,
        model_name=model_name,
        stage1_checkpoint=stage1_checkpoint,
    )

    # Run optimization loop
    logger.info(f"Starting {num_trials} trials for Stage 1.9 optimization")

    for i in range(num_trials):
        try:
            params, trial_idx = ax_client.get_next_trial()
            logger.info(f"Trial {trial_idx}/{num_trials}: {params}")

            metrics = runner.run_trial(params, trial_id=trial_idx)
            ax_client.complete_trial(trial_idx, metrics)

            # Save experiment state
            ax_client.save_experiment(output_dir / "experiment.json")

        except Exception as e:
            logger.error(f"Trial {i} failed: {e}")
            ax_client.mark_trial_failed(trial_idx, str(e))

    # Report best parameters
    best_params = ax_client.get_best_parameters()
    logger.info(f"Best parameters: {best_params}")

    # Save results
    df = ax_client.get_trials_dataframe()
    df.to_csv(output_dir / "trials.csv", index=False)
    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Ax optimization for Stage 1.9 training")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of optimization trials")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Model name")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None, help="Stage 1 checkpoint path")
    parser.add_argument("--output-dir", type=Path, default=Path("./benchmark_results/stage1_9"))
    parser.add_argument("--measurement-steps", type=int, default=200, help="Steps per trial")
    args = parser.parse_args()

    run_optimization(
        num_trials=args.num_trials,
        model_name=args.model,
        stage1_checkpoint=args.stage1_checkpoint,
        output_dir=args.output_dir,
        measurement_steps=args.measurement_steps,
    )


if __name__ == "__main__":
    main()
