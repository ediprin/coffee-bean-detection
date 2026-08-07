from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .hierarchy import (
    TwoLevelHierarchy,
    build_sni21_entity_family_hierarchy,
    hierarchy_level_weights,
    prototype_ema_factor,
)

if TYPE_CHECKING:
    from .model import BHCLConfig


class BalancedHierarchyPrototypeBank(nn.Module):
    """EMA prototype state for all non-root nodes in a two-level SNI21 tree.

    The BHCL paper does not specify prototype initialization. This transfer uses
    deterministic zeros. Unseen nodes therefore contribute exp(0)=1 to the
    class-balanced denominator until observed. Thereafter Eq. (10) is applied
    literally, including epsilon^0=1 for leaf prototypes.
    """

    def __init__(self, config: "BHCLConfig", hierarchy: TwoLevelHierarchy | None = None) -> None:
        super().__init__()
        self.config = config
        self.hierarchy = hierarchy or build_sni21_entity_family_hierarchy()
        d = config.embedding_dim
        self.register_buffer(
            "coarse_prototypes", torch.zeros(self.hierarchy.coarse_count, d), persistent=True
        )
        self.register_buffer(
            "leaf_prototypes", torch.zeros(self.hierarchy.leaf_count, d), persistent=True
        )
        self.register_buffer(
            "coarse_seen", torch.zeros(self.hierarchy.coarse_count, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "leaf_seen", torch.zeros(self.hierarchy.leaf_count, dtype=torch.bool), persistent=True
        )
        self.register_buffer(
            "level_weights",
            hierarchy_level_weights(self.hierarchy.levels).float(),
            persistent=True,
        )

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, leaf_labels: torch.Tensor) -> None:
        if not len(leaf_labels):
            return
        normalized = F.normalize(embeddings.detach(), dim=1, eps=1e-8)
        leaf_labels = leaf_labels.long()
        coarse_labels = self.hierarchy.coarse_labels(leaf_labels)

        coarse_alpha = prototype_ema_factor(
            self.config.epsilon, levels=self.hierarchy.levels, level=1
        )
        leaf_alpha = prototype_ema_factor(
            self.config.epsilon, levels=self.hierarchy.levels, level=2
        )
        for coarse_id in coarse_labels.unique().tolist():
            coarse_id = int(coarse_id)
            mean = normalized[coarse_labels == coarse_id].mean(dim=0)
            self.coarse_prototypes[coarse_id].mul_(1.0 - coarse_alpha).add_(mean, alpha=coarse_alpha)
            self.coarse_seen[coarse_id] = True
        for leaf_id in leaf_labels.unique().tolist():
            leaf_id = int(leaf_id)
            mean = normalized[leaf_labels == leaf_id].mean(dim=0)
            self.leaf_prototypes[leaf_id].mul_(1.0 - leaf_alpha).add_(mean, alpha=leaf_alpha)
            self.leaf_seen[leaf_id] = True

    def normalized(self, level: int) -> torch.Tensor:
        if level == 1:
            value = self.coarse_prototypes
        elif level == 2:
            value = self.leaf_prototypes
        else:
            raise ValueError("BHCL transfer hanya memiliki level 1 dan 2")
        return F.normalize(value.detach(), dim=1, eps=1e-8)
