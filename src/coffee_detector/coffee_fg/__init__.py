"""Fine-grained detection extensions for YOLO26."""

from .model import (
    CoffeeFGDetectHead,
    CoffeeFGDetectionModel,
    MultiLevelROIRefiner,
    inject_coffee_fg,
)
from .trainer import make_coffee_fg_trainer

__all__ = [
    "CoffeeFGDetectHead",
    "CoffeeFGDetectionModel",
    "MultiLevelROIRefiner",
    "inject_coffee_fg",
    "make_coffee_fg_trainer",
]
