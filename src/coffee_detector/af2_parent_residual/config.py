from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from coffee_detector.igem.model import IGEMConfig
from coffee_detector.safpn_alignment.model import SAFPNAlignmentConfig


@dataclass(frozen=True)
class AF2ParentResidualConfig:
    """Frozen AF2 parent plus one classification-only residual family."""

    family: str = "saf"  # saf | igem
    conditioning: str = "feature"  # zero | feature
    saf: SAFPNAlignmentConfig = field(default_factory=SAFPNAlignmentConfig)
    igem: IGEMConfig = field(default_factory=IGEMConfig)

    @classmethod
    def from_mapping(
        cls, payload: "AF2ParentResidualConfig | Mapping[str, Any] | None"
    ) -> "AF2ParentResidualConfig":
        if isinstance(payload, cls):
            result = payload
        else:
            values = dict(payload or {})
            values["saf"] = SAFPNAlignmentConfig.from_mapping(values.get("saf"))
            values["igem"] = IGEMConfig.from_mapping(values.get("igem"))
            result = cls(**values)
        if result.family not in {"saf", "igem"}:
            raise ValueError("family harus saf atau igem")
        if result.conditioning not in {"zero", "feature"}:
            raise ValueError("conditioning harus zero atau feature")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
