from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


ARMS = ("AF2BASE", "AF2RGBDS", "AF2SPDS")
TARGETS = ("none", "rgb", "af2_signal")


@dataclass(frozen=True)
class AF2SPDSConfig:
    """Frozen matched-continuation settings for the AF2-SPDS screen."""

    arm: str = "AF2BASE"
    target: str = "none"
    auxiliary_gain: float = 0.10
    decoder_channels: int = 3

    @classmethod
    def from_mapping(
        cls, payload: "AF2SPDSConfig | Mapping[str, Any] | None"
    ) -> "AF2SPDSConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.arm not in ARMS:
            raise ValueError(f"arm harus salah satu {ARMS}")
        expected = {
            "AF2BASE": "none",
            "AF2RGBDS": "rgb",
            "AF2SPDS": "af2_signal",
        }[result.arm]
        if result.target != expected:
            raise ValueError(f"{result.arm} harus memakai target={expected}")
        if result.auxiliary_gain != 0.10:
            raise ValueError("auxiliary_gain dibekukan pada 0.10")
        if result.decoder_channels != 3:
            raise ValueError("decoder_channels harus 3 untuk target RGB/sinyal AF2")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
