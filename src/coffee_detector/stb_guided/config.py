from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


MODES = ("crosskd", "crosskd_af2")


@dataclass(frozen=True)
class STBGuidedConfig:
    """Training-only guidance for a deployed WAV-L1 + native YOLO26 student.

    `crosskd` is the first authorized stage. `crosskd_af2` is implemented so the
    final framework is explicit, but its experiment must remain blocked until
    the frozen S2 CrossKD stage has passed its own validation protocol.
    """

    mode: str = "crosskd"
    temperature: float = 2.0
    distillation_weight: float = 0.50
    minimum_teacher_gt_probability: float = 0.10
    teacher_checkpoint: str | None = None

    # S3-only settings. They do not affect S2/crosskd training or inference.
    af2_detection_weight: float = 0.50
    consistency_weight: float = 0.25
    consistency_temperature: float = 1.0

    @classmethod
    def from_mapping(
        cls, payload: "STBGuidedConfig | Mapping[str, Any] | None"
    ) -> "STBGuidedConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.mode not in MODES:
            raise ValueError(f"mode harus salah satu {MODES}")
        if result.temperature <= 0.0 or result.consistency_temperature <= 0.0:
            raise ValueError("temperature harus positif")
        if result.distillation_weight < 0.0:
            raise ValueError("distillation_weight harus non-negatif")
        if not 0.0 <= result.minimum_teacher_gt_probability <= 1.0:
            raise ValueError("minimum_teacher_gt_probability harus di [0,1]")
        if not result.teacher_checkpoint:
            raise ValueError("STB-guided training memerlukan teacher_checkpoint STB1")
        if result.af2_detection_weight < 0.0 or result.consistency_weight < 0.0:
            raise ValueError("bobot AF2/consistency harus non-negatif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
