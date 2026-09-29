"""Redundancy-aware selective texture fusion for DefectosCafeVerde."""

from .config import RATFConfig
from .model import (
    RATFDetectHead,
    RATFDetectionModel,
    build_ratf_model,
    load_ratf_weights,
)
from .operator import ratf_texture_cue

__all__ = [
    "RATFConfig",
    "RATFDetectHead",
    "RATFDetectionModel",
    "build_ratf_model",
    "load_ratf_weights",
    "ratf_texture_cue",
]
