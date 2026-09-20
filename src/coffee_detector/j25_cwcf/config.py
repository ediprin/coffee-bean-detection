from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class CWCFConfig:
    """Frozen configuration for the first J25 CWCF screen."""

    cue_channels: int = 4
    attribute_gain: float = 0.15
    wavelet_levels: int = 2
    cue_clip: float = 4.0
    explicit_composition: bool = False
    composition_gain_max: float = 0.5
    class_balanced_attributes: bool = False
    attribute_loss: str = "bce"
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    asl_detach_focal_weight: bool = True
    conditional_confusion_gain: float = 0.0

    @classmethod
    def from_mapping(
        cls, value: "CWCFConfig | Mapping[str, object] | None"
    ) -> "CWCFConfig":
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            allowed = set(cls.__dataclass_fields__)
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f"Kunci CWCF tidak dikenal: {sorted(unknown)}")
            result = cls(**value)
        else:
            raise TypeError("CWCF config harus mapping, CWCFConfig, atau None")
        if result.cue_channels != 4:
            raise ValueError("CWCF v1 dikunci pada empat cue: Cb, Cr, Haar-L1, Haar-L2")
        if result.wavelet_levels != 2:
            raise ValueError("CWCF v1 dikunci pada dua level Haar")
        if not 0.0 < result.attribute_gain <= 1.0:
            raise ValueError("attribute_gain harus berada di (0,1]")
        if result.cue_clip <= 0:
            raise ValueError("cue_clip harus positif")
        if not 0.0 < result.composition_gain_max <= 1.0:
            raise ValueError("composition_gain_max harus berada di (0,1]")
        if result.attribute_loss not in {"bce", "asl"}:
            raise ValueError("attribute_loss harus 'bce' atau 'asl'")
        if result.asl_gamma_pos < 0.0 or result.asl_gamma_neg < 0.0:
            raise ValueError("gamma ASL harus non-negatif")
        if result.asl_gamma_neg < result.asl_gamma_pos:
            raise ValueError("ASL memerlukan gamma_neg >= gamma_pos")
        if not 0.0 <= result.asl_clip < 1.0:
            raise ValueError("asl_clip harus berada di [0,1)")
        if not 0.0 <= result.conditional_confusion_gain <= 1.0:
            raise ValueError("conditional_confusion_gain harus berada di [0,1]")
        return result

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
