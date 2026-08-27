"""Complementary mechanisms screened on top of the frozen AF2 detector."""

from .config import AF2ComplementConfig
from .loss import balanced_supervised_contrastive_loss
from .model import (
    AF2ComplementDetectionModel,
    SharedFeatureDetectHead,
    load_af2_complement_weights,
)
from .modules import FrequencySelectionResidual, SpaceFrequencySelectionResidual
from .trainer import make_af2_complement_trainer

__all__ = [
    "AF2ComplementConfig",
    "AF2ComplementDetectionModel",
    "FrequencySelectionResidual",
    "SharedFeatureDetectHead",
    "SpaceFrequencySelectionResidual",
    "balanced_supervised_contrastive_loss",
    "load_af2_complement_weights",
    "make_af2_complement_trainer",
]
