# WrinkleFree-Deployer Examples

This directory contains Python scripts demonstrating how to use the `wf_deploy` library to programmatically manage training and inference.

## Prerequisites

1.  **Install the library:**
    ```bash
    pip install -e .
    ```

2.  **Configure Environment:**
    Create a `.env` file in the project root with your cloud credentials.
    ```bash
    cp ../.env.example ../.env
    # Edit .env
    ```

## Scripts

### 1. Launch Training (`launch_training.py`)
Demonstrates how to configure and launch a training job on cloud GPUs.
*   **Usage:** `python launch_training.py`
*   **Key concepts:** `TrainingConfig`, `Trainer` class.

### 2. Deploy Service (`deploy_service.py`)
Demonstrates how to deploy a model for inference using SkyServe.
*   **Usage:** `python deploy_service.py`
*   **Key concepts:** `ServiceConfig`, `Deployer` class.

### 3. Provision Infrastructure (`provision_infra.py`)
Demonstrates how to provision underlying infrastructure (Terraform) programmatically.
*   **Usage:** `python provision_infra.py`
*   **Key concepts:** `Infra` class.
