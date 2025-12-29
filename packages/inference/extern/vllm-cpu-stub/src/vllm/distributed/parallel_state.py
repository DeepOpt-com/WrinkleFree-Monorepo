"""CPU stub for vllm distributed parallel state.

Provides no-op implementations for single-GPU/CPU scenarios.
"""

from typing import Optional
import torch.distributed as dist


class FakeProcessGroup:
    """Fake process group for non-distributed scenarios."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        self._rank = rank
        self._world_size = world_size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world_size


_fake_group = FakeProcessGroup()


def get_pp_group() -> FakeProcessGroup:
    """Get pipeline parallel group (stub returns fake group)."""
    return _fake_group


def get_tp_group() -> FakeProcessGroup:
    """Get tensor parallel group (stub returns fake group)."""
    return _fake_group


def get_world_group() -> FakeProcessGroup:
    """Get world group (stub returns fake group)."""
    return _fake_group


def get_tensor_model_parallel_world_size() -> int:
    """Get tensor model parallel world size."""
    return 1


def get_tensor_model_parallel_rank() -> int:
    """Get tensor model parallel rank."""
    return 0


def get_pipeline_model_parallel_world_size() -> int:
    """Get pipeline model parallel world size."""
    return 1


def get_pipeline_model_parallel_rank() -> int:
    """Get pipeline model parallel rank."""
    return 0


def is_initialized() -> bool:
    """Check if distributed is initialized."""
    return False


def init_distributed_environment(
    world_size: int = 1,
    rank: int = 0,
    distributed_init_method: Optional[str] = None,
    local_rank: int = 0,
    backend: str = "gloo",
) -> None:
    """Initialize distributed environment (no-op for CPU stub)."""
    pass
