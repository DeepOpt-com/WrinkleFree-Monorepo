"""Plateau-based early stopping for training."""

import json
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class PlateauEarlyStopping:
    """
    Stop training when metric stops improving.

    Standard early stopping pattern used by PyTorch Lightning, Ignite, etc.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.01,
        mode: Literal["min", "max"] = "min",
        min_evals: int = 3,
        enabled: bool = True,
        rank: int = 0,
    ):
        """
        Args:
            patience: Number of evals without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss (lower is better), "max" for accuracy
            min_evals: Minimum evals before early stopping can trigger
            enabled: Whether early stopping is active
            rank: Process rank (only rank 0 logs)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_evals = min_evals
        self.enabled = enabled
        self.rank = rank

        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.wait: int = 0
        self.eval_count: int = 0
        self.stopped_early: bool = False
        self.best_step: int = 0

    def check(self, metric: float, step: int = 0) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current metric value (e.g., eval loss)
            step: Current training step (for logging)

        Returns:
            True if training should stop
        """
        if not self.enabled:
            return False

        self.eval_count += 1

        # Check if improved
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.best_step = step
            self.wait = 0
        else:
            self.wait += 1

        # Log progress
        if self.rank == 0:
            self._log_wandb(metric, step)

            if self.wait > 0:
                logger.info(
                    f"No improvement for {self.wait}/{self.patience} evals. "
                    f"Best: {self.best:.4f} at step {self.best_step}"
                )

        # Don't stop before min_evals
        if self.eval_count < self.min_evals:
            return False

        if self.wait >= self.patience:
            self.stopped_early = True
            if self.rank == 0:
                self._notify_early_stop(step)
            return True

        return False

    def _notify_early_stop(self, step: int):
        """Log and notify when early stopping triggers."""
        msg = (
            f"EARLY STOPPING TRIGGERED at step {step}. "
            f"Loss plateaued at {self.best:.4f} for {self.patience} consecutive evals. "
            f"Best was at step {self.best_step}."
        )
        logger.warning(msg)

        # Log to WandB with alert
        try:
            import wandb

            if wandb.run is not None:
                wandb.alert(
                    title="Early Stopping Triggered",
                    text=msg,
                    level=wandb.AlertLevel.WARN,
                )
                wandb.run.summary["early_stopped"] = True
                wandb.run.summary["early_stop_step"] = step
                wandb.run.summary["plateau_loss"] = self.best
        except (ImportError, Exception):
            pass

    def _log_wandb(self, metric: float, step: int):
        """Log to WandB if available."""
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "early_stopping/metric": metric,
                        "early_stopping/best": self.best,
                        "early_stopping/wait": self.wait,
                        "early_stopping/patience": self.patience,
                    },
                    step=step,
                )
        except ImportError:
            pass

    def save_json(self, output_dir: Path):
        """Save state to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "early_stopping.json", "w") as f:
            json.dump(
                {
                    "stopped_early": self.stopped_early,
                    "best_metric": self.best,
                    "best_step": self.best_step,
                    "total_evals": self.eval_count,
                    "config": {
                        "patience": self.patience,
                        "min_delta": self.min_delta,
                        "mode": self.mode,
                        "min_evals": self.min_evals,
                    },
                },
                f,
                indent=2,
            )

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "best": self.best,
            "wait": self.wait,
            "eval_count": self.eval_count,
            "best_step": self.best_step,
            "stopped_early": self.stopped_early,
        }

    def load_state_dict(self, state: dict):
        """Restore state from checkpoint."""
        self.best = state.get("best", self.best)
        self.wait = state.get("wait", 0)
        self.eval_count = state.get("eval_count", 0)
        self.best_step = state.get("best_step", 0)
        self.stopped_early = state.get("stopped_early", False)
