from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VARIANTS = ("C1", "C2", "C3", "C4", "CAFR")


@dataclass(frozen=True)
class CAFRConfig:
    """Frozen configuration for Coffee-Adaptive Frequency Representation.

    The variants are a causal ladder rather than independent hyperparameter searches:
      C1   : luminance-guided/shared RGB gate, Xu-style angular hard selection.
      C2   : C1 + explicit radial x directional decomposition.
      C3   : C2 + entropy-conditioned soft selection.
      C4   : C3 + conjugate-symmetry-aware unsigned orientation bins.
      CAFR : C4 + coffee-scale-calibrated patch size.

    CAFR is parameter-free: every field below is a frozen signal-processing choice,
    not a learned parameter.
    """

    variant: str = "CAFR"
    patch_size: int = 32
    overlap: float = 0.50
    gamma: float = 0.10
    angular_bins: int = 16
    orientation_period: float = 180.0
    radial_boundaries: tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
    soft_selection: bool = True
    soft_temperature: float = 0.02
    chunk_size: int = 128
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(cls, payload: "CAFRConfig | dict[str, Any] | None") -> "CAFRConfig":
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
        if self.variant not in VARIANTS:
            raise ValueError(f"variant harus salah satu {VARIANTS}")
        if self.patch_size <= 1:
            raise ValueError("patch_size harus >1")
        if not 0.0 <= self.overlap < 1.0 or self.stride <= 0:
            raise ValueError("overlap tidak menghasilkan stride valid")
        if self.gamma <= 0.0 or self.soft_temperature <= 0.0:
            raise ValueError("gamma/soft_temperature harus positif")
        if self.angular_bins <= 1:
            raise ValueError("angular_bins harus >1")
        if self.orientation_period not in (180.0, 360.0):
            raise ValueError("orientation_period hanya 180 atau 360 derajat")
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


def frozen_variant_config(variant: str, *, patch_size: int = 32) -> CAFRConfig:
    """Return the pre-registered causal variant without post-hoc tuning."""

    if variant not in VARIANTS:
        raise ValueError(f"variant harus salah satu {VARIANTS}")
    values: dict[str, Any] = {"variant": variant, "patch_size": int(patch_size)}
    if variant == "C1":
        values.update(
            angular_bins=360,
            orientation_period=360.0,
            radial_boundaries=(),
            soft_selection=False,
        )
    elif variant == "C2":
        values.update(
            angular_bins=360,
            orientation_period=360.0,
            radial_boundaries=(1.0 / 3.0, 2.0 / 3.0),
            soft_selection=False,
        )
    elif variant == "C3":
        values.update(
            angular_bins=360,
            orientation_period=360.0,
            radial_boundaries=(1.0 / 3.0, 2.0 / 3.0),
            soft_selection=True,
        )
    elif variant in {"C4", "CAFR"}:
        values.update(
            angular_bins=16,
            orientation_period=180.0,
            radial_boundaries=(1.0 / 3.0, 2.0 / 3.0),
            soft_selection=True,
        )
    return CAFRConfig.from_mapping(values)
