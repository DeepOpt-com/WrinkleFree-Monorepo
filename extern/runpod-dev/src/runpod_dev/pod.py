"""RunPod instance management."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import runpod

from .config import (
    DEFAULT_CONTAINER_DISK,
    DEFAULT_GPU,
    DEFAULT_IMAGE,
    DEFAULT_VOLUME_SIZE,
    SSHInfo,
    get_api_key,
    get_gpu_type_id,
)


@dataclass
class PodInfo:
    """Information about a RunPod instance."""

    id: str
    name: str
    status: str
    gpu_type: str | None = None
    ssh_info: SSHInfo | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "gpu_type": self.gpu_type,
            "ssh": self.ssh_info.to_dict() if self.ssh_info else None,
        }


class PodManager:
    """Manage RunPod instances."""

    def __init__(self, api_key: str | None = None):
        """Initialize the pod manager.

        Args:
            api_key: RunPod API key. If None, loads from environment/files.
        """
        self.api_key = api_key or get_api_key()
        runpod.api_key = self.api_key

    def find_by_name(self, name: str) -> PodInfo | None:
        """Find an existing pod by name.

        Args:
            name: The pod name to search for.

        Returns:
            PodInfo if found, None otherwise.
        """
        pods = runpod.get_pods()
        for pod in pods:
            if pod.get("name") == name:
                return self._pod_to_info(pod)
        return None

    def list_pods(self) -> list[PodInfo]:
        """List all pods.

        Returns:
            List of PodInfo objects.
        """
        pods = runpod.get_pods()
        return [self._pod_to_info(pod) for pod in pods]

    def create(
        self,
        name: str,
        gpu: str = DEFAULT_GPU,
        gpu_count: int = 1,
        image: str = DEFAULT_IMAGE,
        volume_size: int = DEFAULT_VOLUME_SIZE,
        container_disk: int = DEFAULT_CONTAINER_DISK,
    ) -> PodInfo:
        """Create a new RunPod instance.

        Args:
            name: Name for the pod.
            gpu: GPU type (short name like "H100" or full ID).
            gpu_count: Number of GPUs (1-8, on single node).
            image: Docker image to use.
            volume_size: Persistent volume size in GB.
            container_disk: Container disk size in GB.

        Returns:
            PodInfo for the created pod.
        """
        gpu_type_id = get_gpu_type_id(gpu)

        pod = runpod.create_pod(
            name=name,
            image_name=image,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            volume_in_gb=volume_size,
            container_disk_in_gb=container_disk,
            support_public_ip=True,
            start_ssh=True,
            ports="22/tcp,8888/http",
            volume_mount_path="/workspace",
        )

        return PodInfo(
            id=pod["id"],
            name=name,
            status="PENDING",
            gpu_type=f"{gpu_count}x {gpu}" if gpu_count > 1 else gpu,
        )

    def wait_for_ready(
        self,
        pod_id: str,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> PodInfo:
        """Wait for a pod to be running with SSH available.

        Args:
            pod_id: The pod ID to wait for.
            timeout: Maximum wait time in seconds.
            poll_interval: Time between status checks.

        Returns:
            PodInfo with SSH info populated.

        Raises:
            TimeoutError: If pod doesn't become ready within timeout.
        """
        start = time.time()

        while time.time() - start < timeout:
            pod = runpod.get_pod(pod_id)

            if pod.get("desiredStatus") == "RUNNING":
                runtime = pod.get("runtime") or {}
                ports = runtime.get("ports") or []

                # Check if SSH port is available
                for port in ports:
                    if port.get("privatePort") == 22 and port.get("ip"):
                        info = self._pod_to_info(pod)
                        if info.ssh_info:
                            return info

            time.sleep(poll_interval)

        raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")

    def stop(self, name: str) -> bool:
        """Stop a pod (preserves data, can be resumed).

        Args:
            name: Pod name to stop.

        Returns:
            True if successful.

        Raises:
            ValueError: If pod not found.
        """
        pod = self.find_by_name(name)
        if not pod:
            raise ValueError(f"Pod '{name}' not found")

        runpod.stop_pod(pod.id)
        return True

    def terminate(self, name: str) -> bool:
        """Terminate a pod (deletes it permanently).

        Args:
            name: Pod name to terminate.

        Returns:
            True if successful.

        Raises:
            ValueError: If pod not found.
        """
        pod = self.find_by_name(name)
        if not pod:
            raise ValueError(f"Pod '{name}' not found")

        runpod.terminate_pod(pod.id)
        return True

    def resume(self, name: str, gpu_count: int = 1) -> PodInfo:
        """Resume a stopped pod.

        Args:
            name: Pod name to resume.
            gpu_count: Number of GPUs to resume with.

        Returns:
            PodInfo for the resumed pod.

        Raises:
            ValueError: If pod not found.
        """
        pod = self.find_by_name(name)
        if not pod:
            raise ValueError(f"Pod '{name}' not found")

        runpod.resume_pod(pod.id, gpu_count=gpu_count)
        return self.wait_for_ready(pod.id)

    def _pod_to_info(self, pod: dict[str, Any]) -> PodInfo:
        """Convert RunPod API response to PodInfo."""
        ssh_info = None

        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []

        for port in ports:
            if port.get("privatePort") == 22 and port.get("ip"):
                ssh_info = SSHInfo(
                    host=port["ip"],
                    port=port["publicPort"],
                )
                break

        return PodInfo(
            id=pod["id"],
            name=pod.get("name", "unknown"),
            status=pod.get("desiredStatus", "UNKNOWN"),
            gpu_type=pod.get("machine", {}).get("gpuDisplayName"),
            ssh_info=ssh_info,
        )
