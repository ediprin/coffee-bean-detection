"""AF2 signal-preservation deep supervision.

The auxiliary decoders exist only during training.  They observe P3/P4/P5 but
never modify the tensors consumed by the native YOLO Detect head.
"""

from .config import ARMS, AF2SPDSConfig
from .loss import multilevel_reconstruction_loss, scheduled_auxiliary_gain
from .model import (
    AF2SPDSDetectionModel,
    AuxiliaryReconstructionDetectHead,
    load_af2_spds_weights,
    strip_auxiliary_head,
)
from .trainer import make_af2_spds_trainer

__all__ = [
    "ARMS",
    "AF2SPDSConfig",
    "AF2SPDSDetectionModel",
    "AuxiliaryReconstructionDetectHead",
    "load_af2_spds_weights",
    "strip_auxiliary_head",
    "make_af2_spds_trainer",
    "multilevel_reconstruction_loss",
    "scheduled_auxiliary_gain",
]
