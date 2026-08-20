from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ARMS = ("AF2_BASE", "AF2_RADIAL", "AF2_ORIENT")


@dataclass(frozen=True)
class AF2IsolatedConfig:
    """Frozen AF2-centered factorization.

    AF2_BASE   : exact legacy AF2 geometry (360 deg, 360 bins, no radial split).
    AF2_RADIAL : only adds three radial bands; all other AF2 choices are retained.
    AF2_ORIENT : only folds direction into unsigned 180-degree orientation while
                 keeping approximately one-degree angular resolution (180 bins).

    The operator remains parameter-free and preserves AF2's independent RGB
    processing, hard entropy-conditioned suppression, patch size, overlap and
    residual min-max gate.
    """

    arm: str = "AF2_BASE"
    patch_size: int = 32
    overlap: float = 0.50
    gamma: float = 0.10
    angular_bins: int = 360
    orientation_period: float = 360.0
    radial_boundaries: tuple[float, ...] = ()
    chunk_size: int = 128
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(
        cls, payload: "AF2IsolatedConfig | dict[str, Any] | None"
    ) -> "AF2IsolatedConfig":
        if isinstance(payload, cls):
            result = payload
        else:
            values = dict(payload or {})
            if "radial_boundaries" in values:
                values["radial_boundaries"] = tuple(float(v) for v in values["radial_boundaries"])
            result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm harus salah satu {ARMS}")
        if self.patch_size <= 1:
            raise ValueError("patch_size harus >1")
        if not 0.0 <= self.overlap < 1.0 or self.stride <= 0:
            raise ValueError("overlap tidak menghasilkan stride valid")
        if self.gamma <= 0.0 or self.angular_bins <= 1:
            raise ValueError("gamma/angular_bins tidak valid")
        if self.orientation_period not in (180.0, 360.0):
            raise ValueError("orientation_period harus 180 atau 360")
        previous = 0.0
        for boundary in self.radial_boundaries:
            if not previous < boundary < 1.0:
                raise ValueError("radial_boundaries harus naik dan berada di (0,1)")
            previous = boundary
        if self.chunk_size <= 0 or self.eps <= 0.0:
            raise ValueError("chunk_size/eps tidak valid")

    @property
    def stride(self) -> int:
        return int(round(self.patch_size * (1.0 - self.overlap)))

    @property
    def radial_bands(self) -> int:
        return len(self.radial_boundaries) + 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["radial_boundaries"] = list(self.radial_boundaries)
        return payload


def frozen_arm_config(arm: str) -> AF2IsolatedConfig:
    """Return a pre-registered isolated arm without post-hoc tuning."""

    if arm not in ARMS:
        raise ValueError(f"arm harus salah satu {ARMS}")
    values: dict[str, Any] = {"arm": arm}
    if arm == "AF2_RADIAL":
        values.update(
            angular_bins=360,
            orientation_period=360.0,
            radial_boundaries=(1.0 / 3.0, 2.0 / 3.0),
        )
    elif arm == "AF2_ORIENT":
        values.update(
            angular_bins=180,
            orientation_period=180.0,
            radial_boundaries=(),
        )
    return AF2IsolatedConfig.from_mapping(values)
