"""J25 chromatic-wavelet compositional classification frontend."""

from .config import CWCFConfig
from .model import (
    CWCFDetectionModel,
    ChromaticWaveletDetectHead,
    ExplicitCompositionDetectHead,
    attribute_compatibility_logits,
    balanced_attribute_bce,
    build_cwcf_model,
    build_j25_attribute_matrix,
    load_cwcf_weights,
)
from .operator import chromatic_wavelet_cue, haar_decompose

__all__ = [
    "CWCFConfig",
    "CWCFDetectionModel",
    "ChromaticWaveletDetectHead",
    "ExplicitCompositionDetectHead",
    "attribute_compatibility_logits",
    "balanced_attribute_bce",
    "build_cwcf_model",
    "build_j25_attribute_matrix",
    "chromatic_wavelet_cue",
    "haar_decompose",
    "load_cwcf_weights",
]
