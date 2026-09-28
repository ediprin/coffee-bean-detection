"""Luminance-illumination preprocessing with raw-preserving score fusion."""

from .config import LIFRPFConfig
from .model import (
    LIFRPFDetectionModel,
    build_lif_rpf_model,
    load_lif_rpf_weights,
)
from .operator import luminance_illumination_preprocess, luminance_residual_cue

__all__ = [
    "LIFRPFConfig",
    "LIFRPFDetectionModel",
    "build_lif_rpf_model",
    "load_lif_rpf_weights",
    "luminance_illumination_preprocess",
    "luminance_residual_cue",
]
