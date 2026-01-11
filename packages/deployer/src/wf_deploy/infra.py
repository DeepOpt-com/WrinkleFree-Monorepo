"""Infrastructure provisioning using Terraform.

Wraps the python-terraform library to provision cloud infrastructure
for Hetzner, AWS, and GCP.

Example usage:
    from wf_deploy import Infra, Credentials

    creds = Credentials.from_env_file(".env")

    # Provision Hetzner base layer
    hetzner = Infra("hetzner", creds, server_count=3)
    outputs = hetzner.provision(auto_approve=True)
    print(f"Servers: {outputs['server_ips']}")

    # Provision GCP burst layer
    gcp = Infra("gcp", creds, project_id="my-project")
    gcp.provision(auto_approve=True)
"""

from pathlib import Path
from typing import Any, Optional

from wf_deploy.credentials import Credentials


class InfraError(Exception):
    """Error during infrastructure provisioning."""

    pass


class Infra:
    """Manage infrastructure using Terraform."""

    # Default terraform directory relative to package
    TERRAFORM_DIR = Path(__file__).parent.parent.parent / "terraform"

    def __init__(
        self,
        provider: str,
        credentials: Optional[Credentials] = None,
        terraform_dir: Optional[Path] = None,
        **tfvars: Any,
    ):
        """Initialize infrastructure manager.

        Args:
            provider: Cloud provider ('hetzner', 'aws', 'gcp').
            credentials: Cloud credentials. If None, uses existing environment.
            terraform_dir: Custom terraform directory. Defaults to package terraform/.
            **tfvars: Terraform variables to pass.
        """
        self.provider = provider
        self.credentials = credentials or Credentials.from_env()
        self.tfvars = tfvars

        # Resolve terraform working directory
        base_dir = terraform_dir or self.TERRAFORM_DIR
        self.working_dir = base_dir / provider

        if not self.working_dir.exists():
            raise InfraError(
                f"Terraform directory not found: {self.working_dir}. "
                f"Available providers: {', '.join(self._list_providers(base_dir))}"
            )

        # Apply credentials to environment for Terraform
        self.credentials.apply_to_env()

        # Lazy-load terraform handler
        self._tf = None

    @staticmethod
    def _list_providers(base_dir: Path) -> list[str]:
        """List available terraform providers."""
        if not base_dir.exists():
            return []
        return [d.name for d in base_dir.iterdir() if d.is_dir() and (d / "main.tf").exists()]

    @property
    def tf(self):
        """Get or create Terraform handler."""
        if self._tf is None:
            try:
                from python_terraform import Terraform
            except ImportError:
                raise ImportError(
                    "python-terraform is required for infrastructure management. "
                    "Install with: pip install python-terraform"
                )
            self._tf = Terraform(working_dir=str(self.working_dir))
        return self._tf

    def _format_tfvars(self) -> dict[str, str]:
        """Format tfvars for terraform CLI."""
        result = {}
        for key, value in self.tfvars.items():
            if isinstance(value, bool):
                result[key] = "true" if value else "false"
            elif isinstance(value, (list, tuple)):
                result[key] = ",".join(str(v) for v in value)
            else:
                result[key] = str(value)
        return result

    def init(self, upgrade: bool = False) -> None:
        """Initialize Terraform working directory.

        Args:
            upgrade: If True, upgrade modules and plugins.
        """
        return_code, stdout, stderr = self.tf.init(upgrade=upgrade, capture_output=True)

        if return_code != 0:
            raise InfraError(f"Terraform init failed:\n{stderr}")

    def plan(self) -> str:
        """Run terraform plan (dry-run).

        Returns:
            Plan output as string.
        """
        self.init()

        return_code, stdout, stderr = self.tf.plan(
            var=self._format_tfvars(),
            capture_output=True,
            no_color=True,
        )

        if return_code not in (0, 2):  # 2 = changes pending
            raise InfraError(f"Terraform plan failed:\n{stderr}")

        return stdout

    def provision(self, auto_approve: bool = False) -> dict[str, Any]:
        """Provision infrastructure.

        Args:
            auto_approve: Skip interactive approval.

        Returns:
            Dict of terraform outputs.
        """
        self.init()

        return_code, stdout, stderr = self.tf.apply(
            var=self._format_tfvars(),
            skip_plan=auto_approve,
            auto_approve=auto_approve,
            capture_output=True,
            no_color=True,
        )

        if return_code != 0:
            raise InfraError(f"Terraform apply failed:\n{stderr}")

        return self.outputs()

    def destroy(self, auto_approve: bool = False) -> None:
        """Destroy infrastructure.

        Args:
            auto_approve: Skip interactive approval.
        """
        return_code, stdout, stderr = self.tf.destroy(
            var=self._format_tfvars(),
            auto_approve=auto_approve,
            capture_output=True,
            no_color=True,
        )

        if return_code != 0:
            raise InfraError(f"Terraform destroy failed:\n{stderr}")

    def outputs(self) -> dict[str, Any]:
        """Get terraform outputs.

        Returns:
            Dict of output names to values.
        """
        return_code, stdout, stderr = self.tf.output(json=True, capture_output=True)

        if return_code != 0:
            raise InfraError(f"Terraform output failed:\n{stderr}")

        # Parse JSON output
        import json

        try:
            raw = json.loads(stdout)
            # Flatten output structure: {"key": {"value": x}} -> {"key": x}
            return {k: v.get("value", v) for k, v in raw.items()}
        except json.JSONDecodeError:
            return {}

    def state(self) -> dict[str, Any]:
        """Get current terraform state summary.

        Returns:
            Dict with state information.
        """
        return_code, stdout, stderr = self.tf.cmd(
            "state",
            "list",
            capture_output=True,
            no_color=True,
        )

        if return_code != 0:
            if "No state file" in stderr or "empty state" in stderr.lower():
                return {"resources": [], "initialized": False}
            raise InfraError(f"Terraform state failed:\n{stderr}")

        resources = [line.strip() for line in stdout.split("\n") if line.strip()]
        return {"resources": resources, "initialized": True}

    @classmethod
    def list_providers(cls, terraform_dir: Optional[Path] = None) -> list[str]:
        """List available terraform providers.

        Args:
            terraform_dir: Custom terraform directory.

        Returns:
            List of provider names.
        """
        base_dir = terraform_dir or cls.TERRAFORM_DIR
        return cls._list_providers(base_dir)
