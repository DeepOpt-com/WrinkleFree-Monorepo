#!/usr/bin/env python3
"""Example: Provision infrastructure using the wf_deployer library.

This example shows how to:
1. Load credentials from a .env file
2. Provision Hetzner dedicated servers
3. Provision GCP/AWS cloud resources
4. Query infrastructure state

Usage:
    python examples/provision_infra.py
"""

from wf_deployer import Infra, Credentials


def main():
    # Load credentials from .env file (or environment)
    creds = Credentials.from_env_file(".env")

    # List available providers
    print("Available providers:", Infra.list_providers())

    # === Hetzner Base Layer ===
    print("\n=== Provisioning Hetzner Base Layer ===")

    hetzner = Infra(
        "hetzner",
        creds,
        server_count=3,
        server_type="ax102",
    )

    # Dry-run first
    print("Running terraform plan...")
    plan = hetzner.plan()
    print(f"Plan summary:\n{plan[:500]}...")

    # Actually provision (with auto-approve for automation)
    # WARNING: This will create real resources and incur costs!
    #
    # outputs = hetzner.provision(auto_approve=True)
    # print(f"Hetzner outputs: {outputs}")

    # === GCP Burst Layer ===
    print("\n=== Provisioning GCP Burst Layer ===")

    gcp = Infra(
        "gcp",
        creds,
        project_id=creds.gcp_project_id or "my-project",
        region="us-central1",
    )

    # Check current state
    state = gcp.state()
    print(f"GCP state: {state}")

    # To provision:
    # outputs = gcp.provision(auto_approve=True)

    # === AWS Storage ===
    print("\n=== Provisioning AWS Storage ===")

    aws = Infra(
        "aws",
        creds,
        bucket_name="wrinklefree-checkpoints",
    )

    # To provision:
    # outputs = aws.provision(auto_approve=True)

    # === Cleanup ===
    # To destroy infrastructure:
    # hetzner.destroy(auto_approve=True)
    # gcp.destroy(auto_approve=True)
    # aws.destroy(auto_approve=True)


if __name__ == "__main__":
    main()
