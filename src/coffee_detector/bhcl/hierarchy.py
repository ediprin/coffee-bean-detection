from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from coffee_detector.sni21_ontology import SNI21_CLASSES, load_sni21_ontology


@dataclass(frozen=True)
class TwoLevelHierarchy:
    coarse_names: tuple[str, ...]
    leaf_names: tuple[str, ...]
    leaf_to_coarse: tuple[int, ...]

    @property
    def levels(self) -> int:
        return 2

    @property
    def coarse_count(self) -> int:
        return len(self.coarse_names)

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_names)

    def coarse_labels(self, leaf_labels: torch.Tensor) -> torch.Tensor:
        mapping = torch.tensor(self.leaf_to_coarse, device=leaf_labels.device, dtype=torch.long)
        return mapping[leaf_labels.long()]


def build_sni21_entity_family_hierarchy() -> TwoLevelHierarchy:
    ontology = load_sni21_ontology()
    coarse_names: list[str] = []
    coarse_lookup: dict[str, int] = {}
    leaf_to_coarse: list[int] = []
    for class_name in SNI21_CLASSES:
        family = str(ontology["classes"][class_name]["entity_family"])
        if family not in coarse_lookup:
            coarse_lookup[family] = len(coarse_names)
            coarse_names.append(family)
        leaf_to_coarse.append(coarse_lookup[family])
    if len(leaf_to_coarse) != 21 or len(set(range(21))) != 21:
        raise RuntimeError("SNI21 hierarchy leaf count tidak valid")
    if len(coarse_names) < 2:
        raise RuntimeError("entity_family tidak membentuk hierarchy nontrivial")
    return TwoLevelHierarchy(
        coarse_names=tuple(coarse_names),
        leaf_names=tuple(SNI21_CLASSES),
        leaf_to_coarse=tuple(leaf_to_coarse),
    )


def hierarchy_level_weights(levels: int = 2) -> torch.Tensor:
    """BHCL Eq. (7) hierarchy penalties, root excluded."""
    if levels <= 0:
        raise ValueError("levels harus positif")
    raw = torch.tensor(
        [math.exp(1.0 / float(levels + 1 - level)) for level in range(1, levels + 1)],
        dtype=torch.float64,
    )
    return raw / raw.sum()


def prototype_ema_factor(epsilon: float, *, levels: int, level: int) -> float:
    """BHCL Eq. (10): epsilon^(L-l)."""
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon harus di (0,1]")
    if not 1 <= level <= levels:
        raise ValueError("level di luar hierarchy")
    return float(epsilon ** (levels - level))
