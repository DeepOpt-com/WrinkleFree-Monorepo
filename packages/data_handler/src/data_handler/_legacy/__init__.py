"""Legacy CheaperTraining components.

These modules are no longer actively maintained and have been moved to _legacy.
For new projects, use training.meta_optimization.odm instead of influence.

Submodules (import directly):
- data_handler._legacy.influence_datainf: DataInf-based influence (deprecated)
- data_handler._legacy.models: MobileLLM models (deprecated)
- data_handler._legacy.training: Legacy trainers (deprecated)

NOTE: This __init__.py intentionally does not import submodules to avoid
import errors when only some submodules are needed.
"""

# Lazy imports only - don't import anything eagerly to avoid circular imports
# and missing module errors when only some legacy submodules are needed.
