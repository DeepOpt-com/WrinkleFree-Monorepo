#!/usr/bin/env python3
"""Example: Deploy an inference service using the wf_deployer library.

This example shows how to:
1. Load credentials from a .env file
2. Configure a service deployment
3. Deploy and manage the service

Usage:
    python examples/deploy_service.py
"""

from wf_deployer import ServiceConfig, Deployer, Credentials, ResourcesConfig


def main():
    # Load credentials from .env file (or environment)
    creds = Credentials.from_env_file(".env")

    # Configure the service
    config = ServiceConfig(
        name="my-llm-service",
        backend="bitnet",  # or "vllm"
        model_path="gs://my-bucket/models/model.gguf",
        port=8080,
        context_size=4096,
        # Scaling
        min_replicas=1,
        max_replicas=10,
        target_qps=5.0,
        # Resources
        resources=ResourcesConfig(
            cpus="16+",
            memory="128+",
            use_spot=True,
        ),
    )

    print(f"Deploying service: {config.name}")
    print(f"Backend: {config.backend}")
    print(f"Model: {config.model_path}")

    # Create deployer
    deployer = Deployer(config, creds)

    # Deploy (async - returns immediately)
    result = deployer.up(detach=True)
    print(f"Launch result: {result}")

    # Wait and get status
    print("\nChecking status...")
    status = deployer.status()
    print(f"Status: {status}")

    # To tear down:
    # deployer.down()


if __name__ == "__main__":
    main()
