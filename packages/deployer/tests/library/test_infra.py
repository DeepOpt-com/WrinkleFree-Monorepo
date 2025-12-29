"""Tests for wf_deployer.infra module."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from wf_deployer.credentials import Credentials
from wf_deployer.infra import Infra, InfraError


@pytest.fixture
def mock_credentials():
    """Create mock credentials."""
    return Credentials(
        aws_access_key_id="AKIATEST",
        aws_secret_access_key="secret",
        hetzner_api_token="hetzner-token",
    )


@pytest.fixture
def mock_terraform_class():
    """Create mock Terraform class."""
    mock_tf = MagicMock()
    mock_tf.init.return_value = (0, "Initialized", "")
    mock_tf.plan.return_value = (0, "Plan: 3 to add", "")
    mock_tf.apply.return_value = (0, "Applied", "")
    mock_tf.destroy.return_value = (0, "Destroyed", "")
    mock_tf.output.return_value = (
        0,
        json.dumps({"server_ips": {"value": ["10.0.0.1", "10.0.0.2"]}}),
        "",
    )
    mock_tf.cmd.return_value = (0, "hetzner_server.node1\nhetzner_server.node2", "")

    mock_class = MagicMock(return_value=mock_tf)
    return mock_class, mock_tf


class TestInfra:
    """Tests for Infra class."""

    def test_init(self, mock_credentials, tmp_path, monkeypatch):
        """Test Infra initialization."""
        # Create mock terraform directory
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        monkeypatch.delenv("HETZNER_API_TOKEN", raising=False)

        infra = Infra(
            "hetzner",
            mock_credentials,
            terraform_dir=tmp_path,
            server_count=3,
        )

        assert infra.provider == "hetzner"
        assert infra.tfvars == {"server_count": 3}

    def test_init_invalid_provider(self, mock_credentials, tmp_path):
        """Test error on invalid provider."""
        with pytest.raises(InfraError, match="not found"):
            Infra("invalid", mock_credentials, terraform_dir=tmp_path)

    def test_list_providers(self, tmp_path):
        """Test listing available providers."""
        # Create mock terraform directories
        for provider in ["hetzner", "aws", "gcp"]:
            d = tmp_path / provider
            d.mkdir()
            (d / "main.tf").touch()

        # Create a directory without main.tf (should be excluded)
        (tmp_path / "incomplete").mkdir()

        providers = Infra.list_providers(terraform_dir=tmp_path)

        assert set(providers) == {"hetzner", "aws", "gcp"}

    def test_format_tfvars(self, mock_credentials, tmp_path):
        """Test formatting terraform variables."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        infra = Infra(
            "hetzner",
            mock_credentials,
            terraform_dir=tmp_path,
            server_count=3,
            enabled=True,
            servers=["a", "b", "c"],
        )

        formatted = infra._format_tfvars()

        assert formatted["server_count"] == "3"
        assert formatted["enabled"] == "true"
        assert formatted["servers"] == "a,b,c"

    def test_init_terraform(self, mock_credentials, tmp_path, mock_terraform_class):
        """Test terraform init."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_class, mock_tf = mock_terraform_class

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)
            infra.init()

            mock_tf.init.assert_called_once()

    def test_init_terraform_failure(self, mock_credentials, tmp_path):
        """Test terraform init failure."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_tf = MagicMock()
        mock_tf.init.return_value = (1, "", "Error: plugin not found")
        mock_class = MagicMock(return_value=mock_tf)

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)

            with pytest.raises(InfraError, match="init failed"):
                infra.init()

    def test_plan(self, mock_credentials, tmp_path, mock_terraform_class):
        """Test terraform plan."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_class, mock_tf = mock_terraform_class

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)
            plan = infra.plan()

            assert "3 to add" in plan

    def test_provision(self, mock_credentials, tmp_path, mock_terraform_class):
        """Test terraform apply."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_class, mock_tf = mock_terraform_class

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)
            outputs = infra.provision(auto_approve=True)

            assert outputs["server_ips"] == ["10.0.0.1", "10.0.0.2"]
            mock_tf.apply.assert_called_once()

    def test_provision_failure(self, mock_credentials, tmp_path):
        """Test terraform apply failure."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_tf = MagicMock()
        mock_tf.init.return_value = (0, "", "")
        mock_tf.apply.return_value = (1, "", "Error: quota exceeded")
        mock_class = MagicMock(return_value=mock_tf)

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)

            with pytest.raises(InfraError, match="apply failed"):
                infra.provision(auto_approve=True)

    def test_destroy(self, mock_credentials, tmp_path, mock_terraform_class):
        """Test terraform destroy."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_class, mock_tf = mock_terraform_class

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)
            infra.destroy(auto_approve=True)

            mock_tf.destroy.assert_called_once()

    def test_destroy_failure(self, mock_credentials, tmp_path):
        """Test terraform destroy failure."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_tf = MagicMock()
        mock_tf.destroy.return_value = (1, "", "Error: resource in use")
        mock_class = MagicMock(return_value=mock_tf)

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)

            with pytest.raises(InfraError, match="destroy failed"):
                infra.destroy(auto_approve=True)

    def test_outputs(self, mock_credentials, tmp_path, mock_terraform_class):
        """Test getting terraform outputs."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_class, mock_tf = mock_terraform_class

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)
            outputs = infra.outputs()

            assert outputs["server_ips"] == ["10.0.0.1", "10.0.0.2"]

    def test_outputs_empty(self, mock_credentials, tmp_path):
        """Test getting outputs when none exist."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_tf = MagicMock()
        mock_tf.output.return_value = (0, "invalid json", "")
        mock_class = MagicMock(return_value=mock_tf)

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)
            outputs = infra.outputs()

            assert outputs == {}

    def test_state(self, mock_credentials, tmp_path, mock_terraform_class):
        """Test getting terraform state."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_class, mock_tf = mock_terraform_class

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)
            state = infra.state()

            assert state["initialized"] is True
            assert len(state["resources"]) == 2
            assert "hetzner_server.node1" in state["resources"]

    def test_state_empty(self, mock_credentials, tmp_path):
        """Test getting state when no resources exist."""
        tf_dir = tmp_path / "hetzner"
        tf_dir.mkdir()
        (tf_dir / "main.tf").touch()

        mock_tf = MagicMock()
        mock_tf.cmd.return_value = (1, "", "No state file")
        mock_class = MagicMock(return_value=mock_tf)

        with patch.dict(
            sys.modules,
            {"python_terraform": MagicMock(Terraform=mock_class)},
        ):
            infra = Infra("hetzner", mock_credentials, terraform_dir=tmp_path)
            state = infra.state()

            assert state["initialized"] is False
            assert state["resources"] == []
