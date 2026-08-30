from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AF2SFSCUEConfig:
    """Frozen single-arm configuration for the direct AF2-SFS-CUE screen."""

    code: str = "AF2SFSCUE1"
    feature_level: int = 0
    lowpass_kernel: int = 3
    decoder_channels: int = 3
    auxiliary_gain: float = 0.10
    signal_mix: float = 0.50

    @classmethod
    def from_mapping(
        cls, payload: "AF2SFSCUEConfig | Mapping[str, Any] | None"
    ) -> "AF2SFSCUEConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.code != "AF2SFSCUE1":
            raise ValueError("code harus AF2SFSCUE1")
        if result.feature_level != 0:
            raise ValueError("SFS dikunci pada P3/feature_level=0")
        if result.lowpass_kernel != 3:
            raise ValueError("lowpass_kernel dikunci pada 3")
        if result.decoder_channels != 3:
            raise ValueError("decoder CUE harus menghasilkan tiga channel")
        if result.auxiliary_gain != 0.10:
            raise ValueError("auxiliary_gain dikunci pada 0.10")
        if result.signal_mix != 0.50:
            raise ValueError("signal_mix dikunci pada 0.50")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
