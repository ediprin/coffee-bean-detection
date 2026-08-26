"""Dense low-rank bilinear classification heads for fresh YOLO26 training."""

from .model import (
    DLRBCConfig,
    DLRBCDetectionModel,
    DLRBCDetectHead,
    LowRankClassResidual,
    load_dlrbc_weights,
)

__all__ = [
    "DLRBCConfig",
    "DLRBCDetectionModel",
    "DLRBCDetectHead",
    "LowRankClassResidual",
    "load_dlrbc_weights",
]
