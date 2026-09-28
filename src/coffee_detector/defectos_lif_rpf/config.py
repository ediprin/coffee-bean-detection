from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class LIFRPFConfig:
    """Frozen seed-42 configuration for LIF-RPF.

    The preprocessing itself is parameter-free.  The detector receives its
    two cue maps only through a zero-initialized, bounded classification gate.
    """

    cue_channels: int = 2
    illumination_kernel: int = 31
    maximum_luminance_delta: float = 0.08
    cue_clip: float = 4.0
    maximum_feature_gain: float = 0.10

    @classmethod
    def from_mapping(
        cls, value: "LIFRPFConfig | Mapping[str, object] | None"
    ) -> "LIFRPFConfig":
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            unknown = set(value) - set(cls.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Kunci LIF-RPF tidak dikenal: {sorted(unknown)}")
            result = cls(**value)
        else:
            raise TypeError("Config LIF-RPF harus mapping, LIFRPFConfig, atau None")
        if result.cue_channels != 2:
            raise ValueError("LIF-RPF dikunci pada cue delta-luminance + reflectance")
        if result.illumination_kernel < 3 or result.illumination_kernel % 2 != 1:
            raise ValueError("illumination_kernel harus ganjil dan minimal 3")
        if not 0.0 < result.maximum_luminance_delta <= 0.25:
            raise ValueError("maximum_luminance_delta harus pada (0, 0.25]")
        if result.cue_clip <= 0.0:
            raise ValueError("cue_clip harus positif")
        if not 0.0 < result.maximum_feature_gain <= 0.25:
            raise ValueError("maximum_feature_gain harus pada (0, 0.25]")
        return result

    def to_dict(self) -> dict:
        return asdict(self)
