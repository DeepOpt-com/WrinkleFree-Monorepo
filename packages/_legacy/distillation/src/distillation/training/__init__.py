"""Distillation training components."""

from distillation.training.config import DistillationConfig, LossConfig, TeacherConfig
from distillation.training.trainer import DistillationTrainer

__all__ = [
    "DistillationConfig",
    "TeacherConfig",
    "LossConfig",
    "DistillationTrainer",
]
