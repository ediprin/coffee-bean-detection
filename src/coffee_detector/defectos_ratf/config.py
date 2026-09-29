from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class RATFConfig:
    """Frozen RATF1 settings for the single seed-42 screen."""

    cue_channels: int = 3
    dwt_levels: int = 3
    gabor_orientations_degrees: tuple[float, float] = (114.0, 172.0)
    gabor_analysis_downsample: int = 2
    gabor_wavelength: float = 10.0
    gabor_sigma: float = 3.0
    gabor_gamma: float = 0.5
    gabor_kernel_size: int = 15
    cue_clip: float = 4.0
    residual_logit_bound: float = 2.0
    orthogonality_epsilon: float = 1e-6

    @classmethod
    def from_mapping(
        cls, value: "RATFConfig | Mapping[str, object] | None"
    ) -> "RATFConfig":
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            allowed = set(cls.__dataclass_fields__)
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"Kunci RATF tidak dikenal: {sorted(unknown)}")
            payload = dict(value)
            if "gabor_orientations_degrees" in payload:
                payload["gabor_orientations_degrees"] = tuple(
                    float(item) for item in payload["gabor_orientations_degrees"]
                )
            result = cls(**payload)
        else:
            raise TypeError("RATF config harus mapping, RATFConfig, atau None")
        if result.cue_channels != 3:
            raise ValueError("RATF1 dikunci pada satu DWT dan dua cue Gabor")
        if result.dwt_levels != 3:
            raise ValueError("RATF1 dikunci pada DWT level 3")
        if len(result.gabor_orientations_degrees) != 2:
            raise ValueError("RATF1 memerlukan tepat dua orientasi Gabor")
        if result.gabor_analysis_downsample != 2:
            raise ValueError("RATF1 mengunci analisis Gabor pada setengah resolusi")
        if result.gabor_kernel_size < 3 or result.gabor_kernel_size % 2 == 0:
            raise ValueError("gabor_kernel_size harus ganjil dan >= 3")
        if min(result.gabor_wavelength, result.gabor_sigma, result.gabor_gamma) <= 0:
            raise ValueError("Parameter Gabor harus positif")
        if result.cue_clip <= 0 or result.residual_logit_bound <= 0:
            raise ValueError("Batas cue dan logit harus positif")
        if result.orthogonality_epsilon <= 0:
            raise ValueError("orthogonality_epsilon harus positif")
        return result

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["gabor_orientations_degrees"] = list(
            self.gabor_orientations_degrees
        )
        return payload
