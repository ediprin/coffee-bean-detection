"""Predicted-box geometry conditioning for YOLO26 classification logits."""

from .model import (
    GeometryConditionedDetectHead,
    GeometryConditionedDetectionModel,
    GeometryConditioningConfig,
    GeometryLogitAdapter,
    load_geometry_conditioned_weights,
)
from .trainer import make_geometry_conditioning_trainer

__all__ = [
    "GeometryConditionedDetectHead",
    "GeometryConditionedDetectionModel",
    "GeometryConditioningConfig",
    "GeometryLogitAdapter",
    "load_geometry_conditioned_weights",
    "make_geometry_conditioning_trainer",
]
