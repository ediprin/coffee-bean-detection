from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class CWCFConfig:
    """Frozen configuration for the first J25 CWCF screen."""

    cue_channels: int = 4
    attribute_gain: float = 0.15
    wavelet_levels: int = 2
    cue_clip: float = 4.0

    @classmethod
    def from_mapping(
        cls, value: "CWCFConfig | Mapping[str, object] | None"
    ) -> "CWCFConfig":
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            allowed = set(cls.__dataclass_fields__)
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"Kunci CWCF tidak dikenal: {sorted(unknown)}")
            result = cls(**value)
        else:
            raise TypeError("CWCF config harus mapping, CWCFConfig, atau None")
        if result.cue_channels != 4:
            raise ValueError("CWCF v1 dikunci pada empat cue: Cb, Cr, Haar-L1, Haar-L2")
        if result.wavelet_levels != 2:
            raise ValueError("CWCF v1 dikunci pada dua level Haar")
        if not 0.0 < result.attribute_gain <= 1.0:
            raise ValueError("attribute_gain harus berada di (0,1]")
        if result.cue_clip <= 0:
            raise ValueError("cue_clip harus positif")
        return result

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
