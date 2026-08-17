from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ARMS = (
    "AF2C",
    "AF2WIN",
    "AF2ORI",
    "AF2POL",
    "AF2SOFT",
    "AF2LUM",
    "PCG1",
    "WAV1",
)


@dataclass(frozen=True)
class SpectralFrontendConfig:
    """Frozen, data-independent configuration for the AF2 factorization study."""

    arm: str = "AF2C"
    patch_size: int = 32
    overlap: float = 0.50
    gamma: float = 0.10
    angular_bins: int = 360
    radial_bands: int = 1
    soft_temperature: float = 0.02
    chunk_size: int = 128
    eps: float = 1.0e-8
    phase_scales: int = 4
    phase_orientations: int = 6
    phase_min_wavelength: float = 3.0
    phase_multiplier: float = 2.1
    phase_sigma_on_f: float = 0.65
    phase_dtheta_on_sigma: float = 1.5
    phase_lowpass_cutoff: float = 0.4
    phase_lowpass_order: int = 10
    phase_noise_k: float = 2.0
    wavelet_levels: int = 2

    @classmethod
    def from_mapping(
        cls, payload: "SpectralFrontendConfig | dict[str, Any] | None"
    ) -> "SpectralFrontendConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.arm not in ARMS:
            raise ValueError(f"arm harus salah satu {ARMS}")
        if result.patch_size <= 1:
            raise ValueError("patch_size harus >1")
        if not 0.0 <= result.overlap < 1.0 or result.stride <= 0:
            raise ValueError("overlap tidak menghasilkan stride valid")
        if result.gamma <= 0.0 or result.soft_temperature <= 0.0:
            raise ValueError("gamma/soft_temperature harus positif")
        if result.angular_bins <= 1 or result.radial_bands <= 0:
            raise ValueError("angular_bins/radial_bands tidak valid")
        if result.chunk_size <= 0 or result.eps <= 0.0:
            raise ValueError("chunk_size/eps tidak valid")
        if result.phase_scales <= 0 or result.phase_orientations <= 0:
            raise ValueError("bank phase-congruency tidak valid")
        if result.phase_min_wavelength < 2.0 or result.phase_multiplier <= 1.0:
            raise ValueError("wavelength/multiplier phase-congruency tidak valid")
        if not 0.0 < result.phase_sigma_on_f < 1.0:
            raise ValueError("phase_sigma_on_f harus berada di (0,1)")
        if not 0.0 < result.phase_lowpass_cutoff < 0.5:
            raise ValueError("phase_lowpass_cutoff harus berada di (0,0.5)")
        if result.wavelet_levels <= 0:
            raise ValueError("wavelet_levels harus positif")
        return result

    @property
    def stride(self) -> int:
        return int(round(self.patch_size * (1.0 - self.overlap)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frozen_arm_config(arm: str) -> SpectralFrontendConfig:
    """Return the pre-registered factorization for an arm, without tuning."""

    if arm not in ARMS:
        raise ValueError(f"arm harus salah satu {ARMS}")
    values: dict[str, Any] = {"arm": arm}
    if arm == "AF2ORI":
        values.update(angular_bins=16)
    elif arm in {"AF2POL", "AF2SOFT", "AF2LUM"}:
        values.update(angular_bins=16, radial_bands=3)
    return SpectralFrontendConfig.from_mapping(values)
