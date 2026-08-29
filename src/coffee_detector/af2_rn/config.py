from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AF2RNConfig:
    """Frozen radially normalized AF2 frontend.

    The constants deliberately equal legacy AF2. ``annulus_width`` is fixed at
    one FFT-grid pixel and is expressed only to make the geometry auditable;
    it is not an experiment hyperparameter.
    """

    patch_size: int = 32
    overlap: float = 0.50
    gamma: float = 0.10
    angular_bins: int = 360
    annulus_width: float = 1.0
    chunk_size: int = 128
    eps: float = 1.0e-8

    @classmethod
    def from_mapping(
        cls, payload: "AF2RNConfig | dict[str, Any] | None" = None
    ) -> "AF2RNConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        result.validate()
        return result

    def validate(self) -> None:
        expected = type(self)()
        if self != expected:
            raise ValueError(
                "AF2RN dikunci pada konfigurasi legacy AF2 dan annulus satu piksel; "
                f"diterima={self.to_dict()}"
            )

    @property
    def stride(self) -> int:
        return int(round(self.patch_size * (1.0 - self.overlap)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
