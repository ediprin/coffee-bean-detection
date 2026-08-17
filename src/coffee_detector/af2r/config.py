from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AF2RConfig:
    """Frozen settings for raw-preserving adaptive AF2.

    ``zero`` is the parameter-matched control. ``illumination`` receives the
    same six photometric/recovery channels while keeping an identical module
    and state-dict schema.
    """

    conditioning: str = "illumination"  # zero | illumination
    hidden_channels: int = 8
    local_kernel: int = 15
    eps: float = 1.0e-6

    @classmethod
    def from_mapping(cls, payload: "AF2RConfig | dict[str, Any] | None") -> "AF2RConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.conditioning not in {"zero", "illumination"}:
            raise ValueError("conditioning AF2R harus zero atau illumination")
        if result.hidden_channels <= 0:
            raise ValueError("hidden_channels AF2R harus positif")
        if result.local_kernel <= 1 or result.local_kernel % 2 == 0:
            raise ValueError("local_kernel AF2R harus ganjil dan >1")
        if result.eps <= 0:
            raise ValueError("eps AF2R harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
