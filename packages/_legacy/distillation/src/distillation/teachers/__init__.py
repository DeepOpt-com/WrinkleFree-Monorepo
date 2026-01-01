"""Teacher model wrappers for distillation."""

from distillation.teachers.base import BaseTeacher
from distillation.teachers.local_teacher import (
    HiddenStateTeacher,
    LocalTeacher,
)
from distillation.teachers.vllm_teacher import (
    VLLMConfig,
    VLLMTeacher,
    VLLMTeacherWithPrefetch,
    create_teacher,
    start_vllm_server,
)

__all__ = [
    # Protocol
    "BaseTeacher",
    # Local
    "LocalTeacher",
    "HiddenStateTeacher",
    # vLLM
    "VLLMTeacher",
    "VLLMTeacherWithPrefetch",
    "VLLMConfig",
    # Factory
    "create_teacher",
    "start_vllm_server",
]
