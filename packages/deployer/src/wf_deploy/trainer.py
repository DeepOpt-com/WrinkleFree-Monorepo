"""Training job management with Modal (default) or SkyPilot backend.

Provides a unified interface for launching and managing training jobs.
Modal is the default backend for easier AI tool integration.

Example usage (Modal - default):
    from wf_deploy import TrainingConfig, Trainer

    config = TrainingConfig(
        name="qwen3-stage2",
        model="qwen3_4b",
        stage=2,
    )
    trainer = Trainer(config)
    run_id = trainer.launch()
    print(trainer.status(run_id))
    trainer.logs(run_id, follow=True)

Example usage (SkyPilot):
    config = TrainingConfig(
        name="qwen3-stage2",
        model="qwen3_4b",
        stage=2,
        backend="skypilot",
        checkpoint_bucket="my-checkpoints",
    )
    trainer = Trainer(config)
    job_id = trainer.launch()

For AI tools (simple JSON API):
    trainer = Trainer.from_json({
        "name": "test-run",
        "model": "qwen3_4b",
        "stage": 2,
        "max_steps": 1000,
    })
    run_id = trainer.launch()
"""

from typing import Any, Optional

from wf_deploy.config import TrainingConfig
from wf_deploy.credentials import Credentials


class Trainer:
    """Unified trainer interface supporting Modal and SkyPilot backends.

    Modal is the default backend, designed for easy AI tool control.
    SkyPilot is available for users who prefer it or need specific cloud features.
    """

    def __init__(
        self,
        config: TrainingConfig,
        credentials: Optional[Credentials] = None,
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration.
            credentials: Cloud credentials (required for SkyPilot, optional for Modal).
        """
        self.config = config
        self.credentials = credentials
        self._current_run_id: Optional[str] = None

        # Initialize appropriate backend
        if config.backend == "modal":
            self._init_modal_backend()
        else:
            self._init_skypilot_backend()

    def _init_modal_backend(self) -> None:
        """Initialize Modal backend."""
        from wf_deploy.modal_deployer import ModalTrainer
        self._modal_trainer = ModalTrainer()

    def _init_skypilot_backend(self) -> None:
        """Initialize SkyPilot backend."""
        if self.credentials is None:
            self.credentials = Credentials.from_env()
        self.credentials.apply_to_env()

        if not self.config.checkpoint_bucket:
            raise ValueError("checkpoint_bucket is required for SkyPilot backend")

    @classmethod
    def from_json(cls, config_dict: dict[str, Any]) -> "Trainer":
        """Create trainer from JSON config.

        Convenience method for AI tools.

        Args:
            config_dict: Dict with training configuration.

        Returns:
            Configured Trainer instance.
        """
        # Add default name if not provided
        if "name" not in config_dict:
            model = config_dict.get("model", "model")
            stage = config_dict.get("stage", 2)
            config_dict["name"] = f"{model}-s{stage}"

        config = TrainingConfig(**config_dict)
        return cls(config)

    def _get_envs(self) -> dict[str, str]:
        """Get environment variables for SkyPilot job."""
        return {
            "MODEL": self.config.model,
            "STAGE": str(self.config.stage),
            "WANDB_PROJECT": self.config.wandb_project,
            "CHECKPOINT_BUCKET": self.config.checkpoint_bucket or "",
            "CHECKPOINT_STORE": self.config.checkpoint_store,
        }

    def _get_job_name(self) -> str:
        """Generate job name from config."""
        return f"wrinklefree-train-{self.config.model}-stage{self.config.stage}"

    def launch(self, detach: bool = True) -> str:
        """Launch a training job.

        Args:
            detach: If True, return immediately after launching.
                   If False, wait for job to complete.

        Returns:
            Run ID (for Modal) or Job ID (for SkyPilot).
        """
        if self.config.backend == "modal":
            return self._launch_modal(detach)
        else:
            return self._launch_skypilot(detach)

    def _launch_modal(self, detach: bool) -> str:
        """Launch training on Modal."""
        result = self._modal_trainer.launch(
            model=self.config.model,
            stage=self.config.stage,
            data=self.config.data,
            max_steps=self.config.max_steps,
            max_tokens=self.config.max_tokens,
            wandb_enabled=self.config.wandb_enabled,
            wandb_project=self.config.wandb_project,
            hydra_overrides=self.config.hydra_overrides,
            detach=detach,
        )

        if detach:
            self._current_run_id = result
            return result
        else:
            self._current_run_id = result.get("run_id")
            return result

    def _launch_skypilot(self, detach: bool) -> str:
        """Launch training on SkyPilot."""
        try:
            import sky
        except ImportError:
            raise ImportError(
                "SkyPilot is required for skypilot backend. "
                "Install with: pip install 'skypilot[all]'"
            )

        # Create task
        task = sky.Task(
            name=self._get_job_name(),
            workdir=self.config.workdir,
            envs=self._get_envs(),
        )

        # Set resources
        resources_kwargs: dict[str, Any] = {
            "accelerators": self.config.accelerators,
            "use_spot": self.config.use_spot,
            "disk_tier": "best",
            "job_recovery": {"max_restarts_on_errors": 3},
        }

        if self.config.cloud:
            resources_kwargs["cloud"] = sky.clouds.CLOUD_REGISTRY.from_str(
                self.config.cloud
            )

        task.set_resources(sky.Resources(**resources_kwargs))

        # Mount checkpoint storage
        storage = sky.Storage(
            name=self.config.checkpoint_bucket,
            source=None,
        )
        task.set_storage_mounts({"/checkpoint": storage})

        # Launch as managed job
        request_id = sky.jobs.launch(task, name=self._get_job_name())

        if detach:
            self._current_run_id = str(request_id)
            return str(request_id)

        job_id, handle = sky.get(request_id)
        self._current_run_id = str(job_id)
        return str(job_id)

    def status(self, run_id: Optional[str] = None) -> dict[str, Any]:
        """Get job status.

        Args:
            run_id: Run ID to check. If None, uses current run.

        Returns:
            Dict with job status information.
        """
        run_id = run_id or self._current_run_id
        if not run_id:
            return {"error": "No run ID provided or available"}

        if self.config.backend == "modal":
            return self._modal_trainer.status(run_id)
        else:
            return self._status_skypilot()

    def _status_skypilot(self) -> dict[str, Any]:
        """Get SkyPilot job status."""
        try:
            import sky
        except ImportError:
            raise ImportError("SkyPilot is required. Install with: pip install 'skypilot[all]'")

        request_id = sky.jobs.queue_v2(refresh=True)
        jobs = sky.get(request_id)

        job_name = self._get_job_name()
        for job in jobs:
            if job.get("name") == job_name:
                return job

        return {"name": job_name, "status": "not_found"}

    def logs(self, run_id: Optional[str] = None, follow: bool = False) -> str:
        """Get job logs.

        Args:
            run_id: Run ID. If None, uses current run.
            follow: If True, stream logs continuously.

        Returns:
            Log output as string (if not following).
        """
        run_id = run_id or self._current_run_id
        if not run_id:
            return "No run ID provided or available"

        if self.config.backend == "modal":
            self._modal_trainer.logs(run_id, follow=follow)
            return ""
        else:
            return self._logs_skypilot(follow)

    def _logs_skypilot(self, follow: bool) -> str:
        """Get SkyPilot job logs."""
        try:
            import sky
        except ImportError:
            raise ImportError("SkyPilot is required. Install with: pip install 'skypilot[all]'")

        request_id = sky.jobs.logs(self._get_job_name(), follow=follow)

        if follow:
            sky.stream_and_get(request_id)
            return ""

        return sky.get(request_id)

    def cancel(self, run_id: Optional[str] = None) -> bool:
        """Cancel the training job.

        Args:
            run_id: Run ID to cancel. If None, uses current run.

        Returns:
            True if cancelled successfully.
        """
        run_id = run_id or self._current_run_id
        if not run_id:
            return False

        if self.config.backend == "modal":
            return self._modal_trainer.cancel(run_id)
        else:
            self._cancel_skypilot()
            return True

    def _cancel_skypilot(self) -> None:
        """Cancel SkyPilot job."""
        try:
            import sky
        except ImportError:
            raise ImportError("SkyPilot is required. Install with: pip install 'skypilot[all]'")

        request_id = sky.jobs.cancel(name=self._get_job_name())
        sky.get(request_id)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent training runs.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            List of run status dicts.
        """
        if self.config.backend == "modal":
            return self._modal_trainer.list_runs(limit=limit)
        else:
            return self._list_skypilot()

    def _list_skypilot(self) -> list[dict[str, Any]]:
        """List SkyPilot jobs."""
        try:
            import sky
        except ImportError:
            raise ImportError("SkyPilot is required. Install with: pip install 'skypilot[all]'")

        request_id = sky.jobs.queue_v2(refresh=True)
        return sky.get(request_id)

    def smoke_test(self, model: str = "smollm2_135m") -> dict[str, Any]:
        """Run a smoke test to verify the training pipeline.

        Args:
            model: Model to use for smoke test.

        Returns:
            Dict with test results.
        """
        if self.config.backend == "modal":
            return self._modal_trainer.smoke_test(model=model)
        else:
            raise NotImplementedError("Smoke test not implemented for SkyPilot backend")


# Convenience function for AI tools
def quick_launch(
    model: str = "qwen3_4b",
    stage: float = 2,
    max_steps: Optional[int] = None,
    **kwargs: Any,
) -> str:
    """Quick launch a training run with minimal config.

    Convenience function for AI tools.

    Args:
        model: Model config name
        stage: Training stage (1, 1.9, 2, or 3)
        max_steps: Maximum training steps
        **kwargs: Additional TrainingConfig fields

    Returns:
        Run ID
    """
    config = TrainingConfig(
        name=f"{model}-s{stage}",
        model=model,
        stage=stage,
        max_steps=max_steps,
        **kwargs,
    )
    trainer = Trainer(config)
    return trainer.launch()
