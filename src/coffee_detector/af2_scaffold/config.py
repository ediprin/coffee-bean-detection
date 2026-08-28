from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, pi
from typing import Any, Mapping


@dataclass(frozen=True)
class AF2ScaffoldConfig:
    """Frozen seed-42 contract for a removable multilevel training scaffold."""

    arm: str = "AF2MTS1"
    feature_levels: tuple[int, ...] = (0, 1, 2)
    lowpass_kernel: int = 3
    full_strength_epochs: int = 18
    zero_start_epoch: int = 28
    total_epochs: int = 30

    @classmethod
    def from_mapping(
        cls, payload: "AF2ScaffoldConfig | Mapping[str, Any] | None"
    ) -> "AF2ScaffoldConfig":
        if isinstance(payload, cls):
            result = payload
        else:
            values = dict(payload or {})
            if "feature_levels" in values:
                values["feature_levels"] = tuple(int(v) for v in values["feature_levels"])
            result = cls(**values)
        if result.arm != "AF2MTS1":
            raise ValueError("Satu-satunya arm yang dibekukan adalah AF2MTS1")
        if result.feature_levels != (0, 1, 2):
            raise ValueError("AF2MTS1 wajib memasang scaffold pada P3/P4/P5")
        if result.lowpass_kernel < 3 or result.lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel harus ganjil dan minimal 3")
        if not 0 < result.full_strength_epochs < result.zero_start_epoch <= result.total_epochs:
            raise ValueError("Jadwal scaffold tidak valid")
        return result

    def strength(self, epoch_index: int) -> float:
        """Return the frozen train strength for a zero-based epoch index."""

        epoch_number = int(epoch_index) + 1
        if epoch_number <= self.full_strength_epochs:
            return 1.0
        if epoch_number >= self.zero_start_epoch:
            return 0.0
        progress = (epoch_number - self.full_strength_epochs) / (
            self.zero_start_epoch - self.full_strength_epochs
        )
        return float(0.5 * (1.0 + cos(pi * progress)))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_levels"] = list(self.feature_levels)
        return payload
