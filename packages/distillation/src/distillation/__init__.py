"""Knowledge distillation for quantized LLMs."""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in _LOSSES:
        from distillation import losses
        return getattr(losses, name)
    elif name in _TEACHERS:
        from distillation import teachers
        return getattr(teachers, name)
    elif name in _TRAINING:
        from distillation import training
        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_LOSSES = {
    "LogitsDistillationLoss",
    "SoftTargetCrossEntropy",
    "AttentionDistillationLoss",
    "AttentionRelationDistillationLoss",
    "BitDistillAttentionRelationLoss",
    "HiddenStateDistillationLoss",
    "BitDistillLoss",
    "ClassificationDistillLoss",
}

_TEACHERS = {
    "BaseTeacher",
    "LocalTeacher",
    "HiddenStateTeacher",
    "VLLMTeacher",
    "VLLMTeacherWithPrefetch",
    "create_teacher",
}

_TRAINING = {
    "DistillationConfig",
    "TeacherConfig",
    "LossConfig",
    "DistillationTrainer",
}

__all__ = list(_LOSSES) + list(_TEACHERS) + list(_TRAINING)
