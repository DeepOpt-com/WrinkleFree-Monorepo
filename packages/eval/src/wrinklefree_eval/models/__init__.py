"""Model wrappers for WrinkleFree-Eval."""

from wrinklefree_eval.models.hf_model import HuggingFaceModel
from wrinklefree_eval.models.bitnet_model import BitNetModel
from wrinklefree_eval.models.fairy2_model import Fairy2Model

__all__ = ["HuggingFaceModel", "BitNetModel", "Fairy2Model"]
