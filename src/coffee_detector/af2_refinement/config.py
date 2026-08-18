from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ARMS = ("AF2C", "AF2RAD", "AF2WAV", "AF2RADWAV")
TRAIN_ARMS = ARMS[1:]


@dataclass(frozen=True)
class AF2RefinementConfig:
    """Frozen parameter-free follow-up derived from AF2 factorization evidence."""

    arm: str = "AF2C"
    patch_size: int = 32
    overlap: float = 0.50
    gamma: float = 0.10
    angular_bins: int = 360
    radial_bands: int = 3
    chunk_size: int = 128
    eps: float = 1.0e-8
    wavelet_levels: int = 2
    fusion: str = "max"

    @classmethod
    def from_mapping(
        cls, payload: "AF2RefinementConfig | dict[str, Any] | None"
    ) -> "AF2RefinementConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.arm not in ARMS:
            raise ValueError(f"arm harus salah satu {ARMS}")
        if result.patch_size <= 1:
            raise ValueError("patch_size harus >1")
        if not 0.0 <= result.overlap < 1.0 or result.stride <= 0:
            raise ValueError("overlap tidak menghasilkan stride valid")
        if result.gamma <= 0.0 or result.angular_bins <= 1:
            raise ValueError("gamma/angular_bins tidak valid")
        if result.radial_bands <= 1:
            raise ValueError("radial_bands harus >1 untuk follow-up radial")
        if result.chunk_size <= 0 or result.eps <= 0.0:
            raise ValueError("chunk_size/eps tidak valid")
        if result.wavelet_levels <= 0:
            raise ValueError("wavelet_levels harus positif")
        if result.fusion != "max":
            raise ValueError("fusion follow-up dikunci pada max")
        return result

    @property
    def stride(self) -> int:
        return int(round(self.patch_size * (1.0 - self.overlap)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frozen_refinement_config(arm: str) -> AF2RefinementConfig:
    if arm not in ARMS:
        raise ValueError(f"arm harus salah satu {ARMS}")
    return AF2RefinementConfig.from_mapping({"arm": arm})
