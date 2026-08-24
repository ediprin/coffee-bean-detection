"""Frequency-consistent STB distillation for one-stage coffee detection."""

from .loss import FCSTBE2ELoss, gt_bounded_logit_distillation
from .model import FCSTBConfig, FCSTBDetectionModel, load_fcstb_weights
from .task import FCSTBTaskModel
from .trainer import make_fcstb_trainer
from .audit import static_fcstb_audit, audit_fcstb_checkpoint_invariance
from .diagnostic import run_frequency_teacher_headroom

__all__ = [
    "FCSTBConfig",
    "FCSTBDetectionModel",
    "FCSTBTaskModel",
    "FCSTBE2ELoss",
    "gt_bounded_logit_distillation",
    "load_fcstb_weights",
    "make_fcstb_trainer",
    "static_fcstb_audit",
    "audit_fcstb_checkpoint_invariance",
    "run_frequency_teacher_headroom",
]
