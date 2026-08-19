from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ARMS = ("WAV1_REF", "HP1", "WAV_L1", "WAV_L2", "WAV_RAWFUSE")
TRAIN_ARMS = ARMS[1:]


@dataclass(frozen=True)
class WAV1FactorizationConfig:
    """Frozen parameter-free cue variants for WAV1 mechanism factorization."""

    arm: str = "WAV1_REF"
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(
        cls, payload: "WAV1FactorizationConfig | dict[str, Any] | None"
    ) -> "WAV1FactorizationConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.arm not in ARMS:
            raise ValueError(f"arm harus salah satu {ARMS}")
        if result.eps <= 0.0:
            raise ValueError("eps harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frozen_arm_config(arm: str) -> WAV1FactorizationConfig:
    if arm not in ARMS:
        raise ValueError(f"arm harus salah satu {ARMS}")
    return WAV1FactorizationConfig(arm=arm)
