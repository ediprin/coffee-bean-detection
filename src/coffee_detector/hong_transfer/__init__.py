"""Hong-to-YOLO26 architecture transfer.

The package implements the three mechanisms described by Hong et al. (2026)
without turning the detector into an ROI/two-stage pipeline.
"""

from .model import (
    DistributionShiftConv2d,
    HongSPPFAttention,
    HongTransferConfig,
    PartialConvBlock,
    inject_hong_transfer,
)
from .trainer import make_hong_transfer_trainer

__all__ = [
    "DistributionShiftConv2d",
    "HongSPPFAttention",
    "HongTransferConfig",
    "PartialConvBlock",
    "inject_hong_transfer",
    "make_hong_transfer_trainer",
]
