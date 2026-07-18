"""Visibility-Aware Dense Copy-Paste utilities.

The package is intentionally independent from Ultralytics internals.  It
materializes ordinary YOLO detection datasets while preserving richer amodal
metadata in a separate JSON file.
"""

from .compositor import CompositionSpec, compose_scene
from .profile import SceneCalibration, build_scene_calibration
from .types import Cutout, PlacedInstance, SyntheticScene, VisibilityBin

__all__ = [
    "CompositionSpec",
    "Cutout",
    "PlacedInstance",
    "SceneCalibration",
    "SyntheticScene",
    "VisibilityBin",
    "compose_scene",
    "build_scene_calibration",
]
