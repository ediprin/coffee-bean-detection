"""AF2 feature-frequency classification adapter for YOLO26."""

from .model import (
    AF2FFAConfig,
    AF2FFADetectHead,
    AF2FFADetectionModel,
    FeatureFrequencyAdapter,
    load_af2_ffa_weights,
)
from .trainer import make_af2_ffa_trainer
from .audit import run_af2_ffa_static_audit

__all__ = [
    "AF2FFAConfig",
    "AF2FFADetectHead",
    "AF2FFADetectionModel",
    "FeatureFrequencyAdapter",
    "load_af2_ffa_weights",
    "make_af2_ffa_trainer",
    "run_af2_ffa_static_audit",
]
