from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DIDAAF2Config:
    """Frozen training-only controls for the AF2 DG x FG factorial study."""

    mode: str = "control"  # control | dg | fg | dgfg
    dg_gain: float = 0.10
    fg_gain: float = 0.05
    temperature: float = 2.0
    margin: float = 0.20
    topk: int = 3
    brightness: float = 0.08
    contrast_low: float = 0.90
    contrast_high: float = 1.10
    gamma_low: float = 0.90
    gamma_high: float = 1.10
    channel_gain_low: float = 0.97
    channel_gain_high: float = 1.03
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, payload: "DIDAAF2Config | dict[str, Any] | None") -> "DIDAAF2Config":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.mode not in {"control", "dg", "fg", "dgfg"}:
            raise ValueError(f"Mode DIDA-AF2 tidak dikenal: {result.mode}")
        if result.dg_gain < 0.0 or result.fg_gain < 0.0:
            raise ValueError("Gain auxiliary harus non-negatif")
        if result.temperature <= 0.0 or result.margin < 0.0 or result.topk <= 0:
            raise ValueError("temperature/margin/topk DIDA-AF2 tidak valid")
        if result.brightness < 0.0:
            raise ValueError("brightness harus non-negatif")
        for low, high, name in (
            (result.contrast_low, result.contrast_high, "contrast"),
            (result.gamma_low, result.gamma_high, "gamma"),
            (result.channel_gain_low, result.channel_gain_high, "channel_gain"),
        ):
            if low <= 0.0 or high < low:
                raise ValueError(f"Rentang {name} tidak valid")
        if result.eps <= 0.0:
            raise ValueError("eps harus positif")
        return result

    @property
    def dg_enabled(self) -> bool:
        return self.mode in {"dg", "dgfg"}

    @property
    def fg_enabled(self) -> bool:
        return self.mode in {"fg", "dgfg"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
