"""Service deployment using SkyPilot SkyServe.

Wraps the SkyPilot Python SDK to deploy inference services. Does NOT duplicate
backend logic - passes config as env vars to existing YAML shell scripts.

Example usage:
    from wf_deploy import ServiceConfig, Deployer, Credentials

    creds = Credentials.from_env_file(".env")
    config = ServiceConfig(
        name="my-llm-service",
        backend="bitnet",
        model_path="gs://my-bucket/model.gguf",
    )

    deployer = Deployer(config, creds)
    endpoint = deployer.up()
    print(f"Service ready at: {endpoint}")

    # Later
    deployer.down()
"""

from typing import Any, Optional

from wf_deploy.config import ServiceConfig
from wf_deploy.credentials import Credentials


class Deployer:
    """Deploy inference services using SkyPilot SkyServe."""

    def __init__(self, config: ServiceConfig, credentials: Optional[Credentials] = None):
        """Initialize the deployer.

        Args:
            config: Service configuration.
            credentials: Cloud credentials. If None, uses existing environment.
        """
        self.config = config
        self.credentials = credentials or Credentials.from_env()

        # Apply credentials to environment for SkyPilot
        self.credentials.apply_to_env()

    def _get_envs(self) -> dict[str, str]:
        """Get environment variables to pass to the service."""
        return {
            "BACKEND": self.config.backend,
            "MODEL_PATH": self.config.model_path,
            "CONTEXT_SIZE": str(self.config.context_size),
            "NUM_THREADS": str(self.config.num_threads),
            "HOST": self.config.host,
            "PORT": str(self.config.port),
            "MLOCK": "true",
        }

    def up(self, detach: bool = True) -> str:
        """Deploy the service.

        Args:
            detach: If True, return immediately after launching.
                   If False, wait for service to be ready.

        Returns:
            Service endpoint URL.
        """
        try:
            import sky
        except ImportError:
            raise ImportError(
                "SkyPilot is required for deployment. "
                "Install with: pip install 'skypilot[all]'"
            )

        # Create task with env vars
        task = sky.Task(
            name=self.config.name,
            envs=self._get_envs(),
        )

        # Set resources
        resources_kwargs: dict[str, Any] = {
            "cpus": self.config.resources.cpus,
            "memory": self.config.resources.memory,
            "use_spot": self.config.resources.use_spot,
            "disk_size": self.config.resources.disk_size,
            "ports": self.config.port,
        }

        if self.config.resources.cloud:
            resources_kwargs["cloud"] = sky.clouds.CLOUD_REGISTRY.from_str(
                self.config.resources.cloud
            )
        if self.config.resources.accelerators:
            resources_kwargs["accelerators"] = self.config.resources.accelerators
        if self.config.resources.region:
            resources_kwargs["region"] = self.config.resources.region

        task.set_resources(sky.Resources(**resources_kwargs))

        # Set service config for autoscaling
        service_config = {
            "readiness_probe": {
                "path": self.config.readiness_path,
                "initial_delay_seconds": self.config.initial_delay_seconds,
            },
            "replica_policy": {
                "min_replicas": self.config.min_replicas,
                "max_replicas": self.config.max_replicas,
                "target_qps_per_replica": self.config.target_qps,
            },
        }
        task.set_service(service_config)

        # Launch service
        request_id = sky.serve.up(task, service_name=self.config.name)

        if detach:
            # Return request ID for async tracking
            return f"Service '{self.config.name}' launching (request_id={request_id})"

        # Wait for completion and return endpoint
        service_name, endpoint = sky.get(request_id)
        return endpoint

    def down(self) -> None:
        """Tear down the service."""
        try:
            import sky
        except ImportError:
            raise ImportError("SkyPilot is required. Install with: pip install 'skypilot[all]'")

        request_id = sky.serve.down(self.config.name)
        sky.get(request_id)

    def status(self) -> dict[str, Any]:
        """Get service status.

        Returns:
            Dict with service status information.
        """
        try:
            import sky
        except ImportError:
            raise ImportError("SkyPilot is required. Install with: pip install 'skypilot[all]'")

        request_id = sky.serve.status(self.config.name)
        return sky.get(request_id)

    def logs(self, follow: bool = False) -> str:
        """Get service logs.

        Args:
            follow: If True, stream logs continuously.

        Returns:
            Log output as string.
        """
        try:
            import sky
        except ImportError:
            raise ImportError("SkyPilot is required. Install with: pip install 'skypilot[all]'")

        request_id = sky.serve.logs(self.config.name, follow=follow)

        if follow:
            sky.stream_and_get(request_id)
            return ""

        return sky.get(request_id)

    def update(
        self,
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
    ) -> None:
        """Update service configuration.

        Args:
            min_replicas: New minimum replicas.
            max_replicas: New maximum replicas.
        """
        try:
            import sky
        except ImportError:
            raise ImportError("SkyPilot is required. Install with: pip install 'skypilot[all]'")

        kwargs = {}
        if min_replicas is not None:
            kwargs["min_replicas"] = min_replicas
        if max_replicas is not None:
            kwargs["max_replicas"] = max_replicas

        if kwargs:
            request_id = sky.serve.update(self.config.name, **kwargs)
            sky.get(request_id)
