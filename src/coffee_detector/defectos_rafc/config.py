from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RAFCConfig:
    """Frozen controls for the training-only RAFC view and objective."""

    mix_strength: float = 0.20
    low_radius: float = 0.04
    high_radius: float = 0.22
    orientation_bins: int = 8
    angular_width_degrees: float = 45.0
    consistency_gain: float = 0.05
    temperature: float = 2.0
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(
        cls, value: "RAFCConfig | Mapping[str, object] | None"
    ) -> "RAFCConfig":
        result = value if isinstance(value, cls) else cls(**dict(value or {}))
        if not 0.0 < result.mix_strength <= 1.0:
            raise ValueError("mix_strength RAFC harus di (0,1]")
        if not 0.0 <= result.low_radius < result.high_radius <= 0.5:
            raise ValueError("Rentang radial RAFC tidak valid")
        if result.orientation_bins < 2:
            raise ValueError("orientation_bins RAFC minimal dua")
        if not 0.0 < result.angular_width_degrees <= 180.0:
            raise ValueError("angular_width_degrees RAFC tidak valid")
        if result.consistency_gain <= 0.0 or result.temperature <= 0.0:
            raise ValueError("Gain dan temperature RAFC harus positif")
        if result.eps <= 0.0:
            raise ValueError("eps RAFC harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
