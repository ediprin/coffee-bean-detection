"""Curriculum-gated SFS and privileged AF2 cue supervision."""

from .audit import run_af2_curriculum_sfs_static_audit
from .config import AF2CurriculumSFSConfig, CurriculumState, curriculum_state
from .model import (
    AF2CurriculumSFSDetectionModel,
    AF2CurriculumSFSHead,
    aligned_auxiliary_scale,
    load_af2_curriculum_sfs_weights,
)
from .trainer import make_af2_curriculum_sfs_trainer

__all__ = [
    "AF2CurriculumSFSConfig",
    "AF2CurriculumSFSDetectionModel",
    "AF2CurriculumSFSHead",
    "CurriculumState",
    "aligned_auxiliary_scale",
    "curriculum_state",
    "load_af2_curriculum_sfs_weights",
    "make_af2_curriculum_sfs_trainer",
    "run_af2_curriculum_sfs_static_audit",
]
