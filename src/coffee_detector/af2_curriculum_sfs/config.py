from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AF2CurriculumSFSConfig:
    """Frozen schedule for the AF2 curriculum-SFS seed-42 screen."""

    code: str = "AF2CURR1"
    feature_level: int = 0
    lowpass_kernel: int = 3
    auxiliary_gain: float = 0.10
    warmup_epochs: int = 5
    ramp_epochs: int = 10
    hold_epochs: int = 5
    decay_epochs: int = 10
    gradient_alignment: bool = True

    @classmethod
    def from_mapping(
        cls, payload: "AF2CurriculumSFSConfig | Mapping[str, Any] | None"
    ) -> "AF2CurriculumSFSConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.code != "AF2CURR1":
            raise ValueError("code harus AF2CURR1")
        if result.feature_level != 0:
            raise ValueError("feature_level dikunci pada shared P3")
        if result.lowpass_kernel != 3:
            raise ValueError("lowpass_kernel dikunci pada 3")
        if result.auxiliary_gain != 0.10:
            raise ValueError("auxiliary_gain dikunci pada 0.10")
        if (result.warmup_epochs, result.ramp_epochs, result.hold_epochs, result.decay_epochs) != (5, 10, 5, 10):
            raise ValueError("Jadwal curriculum dikunci pada 5/10/5/10 epoch")
        if result.gradient_alignment is not True:
            raise ValueError("gradient_alignment wajib aktif")
        return result

    @property
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.ramp_epochs + self.hold_epochs + self.decay_epochs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurriculumState:
    sfs_strength: float
    auxiliary_gain: float
    phase: str


def _cosine_ramp(progress: float) -> float:
    value = min(max(float(progress), 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * value)


def curriculum_state(
    config: AF2CurriculumSFSConfig | Mapping[str, Any], *, epoch: int, epochs: int
) -> CurriculumState:
    """Return the prospectively frozen strength/gain for a zero-based epoch."""

    frozen = AF2CurriculumSFSConfig.from_mapping(config)
    if int(epochs) != frozen.total_epochs:
        raise ValueError(f"epochs harus {frozen.total_epochs}")
    index = min(max(int(epoch), 0), int(epochs) - 1)
    warm_end = frozen.warmup_epochs
    ramp_end = warm_end + frozen.ramp_epochs
    hold_end = ramp_end + frozen.hold_epochs

    if index < warm_end:
        return CurriculumState(0.0, 0.0, "coffee_warmup")
    if index < ramp_end:
        progress = (index - warm_end) / max(frozen.ramp_epochs - 1, 1)
        strength = _cosine_ramp(progress)
        return CurriculumState(strength, frozen.auxiliary_gain * strength, "spectral_ramp")
    if index < hold_end:
        return CurriculumState(1.0, frozen.auxiliary_gain, "joint_hold")

    progress = (index - hold_end) / max(frozen.decay_epochs - 1, 1)
    gain = frozen.auxiliary_gain * (1.0 - _cosine_ramp(progress))
    return CurriculumState(1.0, gain, "auxiliary_release")
