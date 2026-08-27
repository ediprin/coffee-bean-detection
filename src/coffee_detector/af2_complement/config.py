from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


ARMS = ("AF2CTRL", "AF2FS1", "AF2SFS1", "AF2BHCL1")
MODES = ("control", "frequency_select", "space_frequency", "bhcl")


@dataclass(frozen=True)
class AF2ComplementConfig:
    """Frozen seed-screen settings for mechanisms added to legacy AF2."""

    arm: str = "AF2CTRL"
    mode: str = "control"
    feature_level: int = 0
    lowpass_kernel: int = 3
    contrastive_temperature: float = 0.10
    contrastive_gain: float = 0.05
    family_gain: float = 0.50

    @classmethod
    def from_mapping(
        cls, payload: "AF2ComplementConfig | Mapping[str, Any] | None"
    ) -> "AF2ComplementConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.arm not in ARMS:
            raise ValueError(f"arm harus salah satu {ARMS}")
        if result.mode not in MODES:
            raise ValueError(f"mode harus salah satu {MODES}")
        expected = {
            "AF2CTRL": "control",
            "AF2FS1": "frequency_select",
            "AF2SFS1": "space_frequency",
            "AF2BHCL1": "bhcl",
        }[result.arm]
        if result.mode != expected:
            raise ValueError(f"{result.arm} harus memakai mode={expected}")
        if result.feature_level != 0:
            raise ValueError("Screening dikunci pada shared P3 (feature_level=0)")
        if result.lowpass_kernel < 3 or result.lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel harus ganjil dan minimal 3")
        if result.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature harus positif")
        if not 0 <= result.contrastive_gain <= 1:
            raise ValueError("contrastive_gain harus dalam [0,1]")
        if not 0 <= result.family_gain <= 1:
            raise ValueError("family_gain harus dalam [0,1]")
        if result.mode != "bhcl" and result.contrastive_gain != cls.contrastive_gain:
            raise ValueError("Arm non-BHCL tidak boleh mengubah gain contrastive")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
