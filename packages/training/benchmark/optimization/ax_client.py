"""Ax Bayesian Optimization client wrapper."""

import logging
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from benchmark.core.metrics import BenchmarkMetrics

logger = logging.getLogger(__name__)


class BenchmarkAxClient:
    """Wrapper around Ax Service API for hyperparameter optimization.

    Provides a simplified interface for:
    - Creating experiments with search space from config
    - Getting next trial parameters
    - Completing trials with metrics
    - Saving/loading experiment state

    Args:
        search_space_config: Configuration defining the search space
        experiment_name: Name for the Ax experiment
        random_seed: Random seed for reproducibility
    """

    def __init__(
        self,
        search_space_config: DictConfig,
        experiment_name: str = "wrinklefree_benchmark",
        random_seed: int = 42,
    ):
        # Import Ax here to allow module import without Ax installed
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties

        self.search_space_config = search_space_config
        self.experiment_name = experiment_name
        self.random_seed = random_seed

        # Initialize Ax client
        self.ax_client = AxClient(
            random_seed=random_seed,
            verbose_logging=True,
        )

        # Store ObjectiveProperties for setup
        self._ObjectiveProperties = ObjectiveProperties

        self._setup_experiment()

    def _setup_experiment(self) -> None:
        """Configure Ax experiment with search space."""
        parameters = self._build_parameters()

        # Get constraints if defined
        param_constraints = list(
            self.search_space_config.get("parameter_constraints", [])
        )
        outcome_constraints = list(
            self.search_space_config.get("outcome_constraints", [])
        )

        self.ax_client.create_experiment(
            name=self.experiment_name,
            parameters=parameters,
            objectives={
                # Primary objective: maximize convergence efficiency
                # = (loss reduction) / (wall time) / (memory)
                "convergence_per_sec_per_gb": self._ObjectiveProperties(minimize=False),
            },
            parameter_constraints=param_constraints,
            outcome_constraints=outcome_constraints,
        )

        logger.info(
            f"Created Ax experiment '{self.experiment_name}' with {len(parameters)} parameters"
        )

    def _build_parameters(self) -> list[dict]:
        """Build Ax parameter list from config."""
        params = []

        for name, spec in self.search_space_config.parameters.items():
            spec_dict = OmegaConf.to_container(spec) if hasattr(spec, "items") else spec
            param = {"name": name, "type": spec_dict["type"]}

            if spec_dict["type"] == "range":
                param["bounds"] = list(spec_dict["bounds"])
                param["log_scale"] = spec_dict.get("log_scale", False)
                param["value_type"] = spec_dict.get("value_type", "float")
            elif spec_dict["type"] == "choice":
                param["values"] = list(spec_dict["values"])
                param["is_ordered"] = spec_dict.get("is_ordered", False)

            params.append(param)

        return params

    def get_next_trial(self) -> tuple[dict[str, Any], int]:
        """Get next trial parameters from Ax.

        Returns:
            Tuple of (parameters dict, trial_index)
        """
        parameters, trial_index = self.ax_client.get_next_trial()
        logger.info(f"Trial {trial_index}: {parameters}")
        return parameters, trial_index

    def complete_trial(
        self,
        trial_index: int,
        metrics: BenchmarkMetrics,
    ) -> None:
        """Report trial results to Ax.

        Args:
            trial_index: Index of the trial to complete
            metrics: BenchmarkMetrics with measured values
        """
        raw_data = metrics.to_ax_metrics()
        self.ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=raw_data,
        )
        logger.info(
            f"Completed trial {trial_index}: "
            f"convergence={metrics.convergence_per_sec_per_gb:.4f} loss/sec/GB, "
            f"throughput={metrics.tokens_per_sec_per_gb:.1f} tokens/sec/GB"
        )

    def mark_trial_failed(self, trial_index: int, reason: str = "") -> None:
        """Mark a trial as failed.

        Args:
            trial_index: Index of the failed trial
            reason: Reason for failure
        """
        self.ax_client.log_trial_failure(
            trial_index=trial_index,
            metadata={"reason": reason},
        )
        logger.warning(f"Trial {trial_index} failed: {reason}")

    def get_best_parameters(self) -> dict[str, Any]:
        """Get best parameters found so far.

        Returns:
            Dictionary of best parameter values
        """
        best_params, values = self.ax_client.get_best_parameters()
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best values: {values}")
        return best_params

    def get_trials_dataframe(self):
        """Get all trials as a pandas DataFrame.

        Returns:
            DataFrame with trial parameters and results
        """
        return self.ax_client.get_trials_data_frame()

    def get_optimization_trace(self) -> list[float]:
        """Get the optimization trace (best value at each trial).

        Returns:
            List of best objective values over trials
        """
        df = self.get_trials_dataframe()
        if df.empty or "convergence_per_sec_per_gb" not in df.columns:
            return []

        trace = []
        best = float("-inf")
        for val in df["convergence_per_sec_per_gb"]:
            if val is not None and val > best:
                best = val
            trace.append(best if best > float("-inf") else None)
        return trace

    def get_parameter_importance(self) -> Optional[dict[str, float]]:
        """Compute parameter importance via Ax's sensitivity analysis.

        Returns:
            Dictionary mapping parameter names to importance scores,
            or None if not enough data.
        """
        try:
            return self.ax_client.get_feature_importance()
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
            return None

    def get_pareto_frontier(self) -> Optional[list[dict]]:
        """Get Pareto optimal configurations (if multi-objective).

        Returns:
            List of Pareto-optimal parameter configurations,
            or None if single-objective.
        """
        try:
            frontier = self.ax_client.get_pareto_optimal_parameters()
            return frontier
        except Exception as e:
            logger.debug(f"Pareto frontier not applicable: {e}")
            return None

    def save_experiment(self, path: Path) -> None:
        """Save experiment state to JSON file.

        Args:
            path: Path to save the experiment JSON
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ax_client.save_to_json_file(str(path))
        logger.info(f"Saved experiment to {path}")

    @classmethod
    def load_experiment(cls, path: Path) -> "BenchmarkAxClient":
        """Load experiment state from JSON file.

        Args:
            path: Path to the experiment JSON

        Returns:
            BenchmarkAxClient with loaded experiment
        """
        from ax.service.ax_client import AxClient

        ax_client = AxClient.load_from_json_file(str(path))

        # Create instance without running __init__
        instance = cls.__new__(cls)
        instance.ax_client = ax_client
        instance.experiment_name = ax_client.experiment.name
        instance.search_space_config = None
        instance.random_seed = None

        logger.info(f"Loaded experiment from {path}")
        return instance

    def get_num_completed_trials(self) -> int:
        """Get number of completed trials."""
        df = self.get_trials_dataframe()
        if df.empty:
            return 0
        return len(df[df["trial_status"] == "COMPLETED"])

    def get_num_trials(self) -> int:
        """Get total number of trials (including pending/failed)."""
        return len(self.get_trials_dataframe())


def load_search_space(path: Path) -> DictConfig:
    """Load search space configuration from YAML file.

    Args:
        path: Path to the search space YAML

    Returns:
        DictConfig with search space definition
    """
    return OmegaConf.load(path)


def create_default_search_space() -> DictConfig:
    """Create default search space configuration.

    Returns:
        DictConfig with default search space for 1.58-bit training
    """
    return OmegaConf.create(
        {
            "parameters": {
                "optimizer_type": {
                    "type": "choice",
                    "values": ["apollo", "apollo_mini", "muon", "adamw_8bit"],
                    "is_ordered": False,
                },
                "learning_rate": {
                    "type": "range",
                    "bounds": [1e-5, 1e-2],
                    "log_scale": True,
                    "value_type": "float",
                },
                "batch_size": {
                    "type": "choice",
                    "values": [4, 8, 16, 32, 64],
                    "is_ordered": True,
                },
                "gradient_accumulation_steps": {
                    "type": "choice",
                    "values": [1, 2, 4, 8],
                    "is_ordered": True,
                },
                "lambda_logits": {
                    "type": "range",
                    "bounds": [0.1, 50.0],
                    "log_scale": True,
                    "value_type": "float",
                },
                "gamma_attention": {
                    "type": "range",
                    "bounds": [1e-7, 1e-3],
                    "log_scale": True,
                    "value_type": "float",
                },
                "temperature": {
                    "type": "range",
                    "bounds": [1.0, 10.0],
                    "log_scale": False,
                    "value_type": "float",
                },
                "influence_enabled": {
                    "type": "choice",
                    "values": [True, False],
                    "is_ordered": False,
                },
                "influence_lambda_reg": {
                    "type": "range",
                    "bounds": [1e-6, 1e-2],
                    "log_scale": True,
                    "value_type": "float",
                },
                "influence_threshold": {
                    "type": "range",
                    "bounds": [-0.5, 0.5],
                    "log_scale": False,
                    "value_type": "float",
                },
            },
            "objectives": {
                "convergence_per_sec_per_gb": {"minimize": False},
            },
            "parameter_constraints": [],
            "outcome_constraints": [],
        }
    )
