from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch import nn


@dataclass(frozen=True)
class HierVIPConfig:
    embedding_dim: int = 128
    temperature: float = 0.2
    loss_weight: float = 0.001
    momentum_low: float = 0.5
    momentum_base: float = 0.8
    drift_alpha: float = 0.2

    @classmethod
    def from_mapping(cls, payload: "HierVIPConfig | Mapping[str, Any] | None") -> "HierVIPConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.embedding_dim <= 0:
            raise ValueError("embedding_dim harus positif")
        if result.temperature <= 0:
            raise ValueError("temperature harus positif")
        if result.loss_weight < 0:
            raise ValueError("loss_weight tidak boleh negatif")
        if not 0.0 <= result.momentum_low <= result.momentum_base <= 1.0:
            raise ValueError("momentum harus memenuhi 0 <= low <= base <= 1")
        if result.drift_alpha < 0:
            raise ValueError("drift_alpha tidak boleh negatif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HierarchySpec:
    """Four-level SNI tree: leaf -> primary_condition -> entity_family -> root."""

    class_names: tuple[str, ...]
    level_names: tuple[str, ...]
    level_categories: tuple[tuple[str, ...], ...]
    class_to_level: tuple[tuple[int, ...], ...]
    gamma: tuple[float, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_levels(self) -> int:
        return len(self.level_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_names": list(self.class_names),
            "level_names": list(self.level_names),
            "level_categories": [list(value) for value in self.level_categories],
            "class_to_level": [list(value) for value in self.class_to_level],
            "gamma": list(self.gamma),
        }


def _ordered_names(names: Mapping[int, str] | Sequence[str]) -> list[str]:
    if isinstance(names, Mapping):
        keys = sorted(int(key) for key in names)
        if keys != list(range(len(keys))):
            raise ValueError("Class ids harus kontigu mulai dari 0")
        return [str(names[key]) for key in keys]
    return [str(value) for value in names]


def build_sni_hierarchy(
    class_names: Mapping[int, str] | Sequence[str],
    ontology_path: str | Path,
) -> HierarchySpec:
    """Build a taxonomy solely from the frozen SNI ontology, never validation errors."""

    names = _ordered_names(class_names)
    ontology_path = Path(ontology_path).expanduser().resolve()
    payload = yaml.safe_load(ontology_path.read_text(encoding="utf-8")) or {}
    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("Ontology tidak memiliki mapping classes")

    # L0 is the fine class itself. L1/L2 come from frozen ontology fields.
    primary_values: list[str] = []
    family_values: list[str] = []
    for name in names:
        item = classes.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"Class tidak ada di ontology: {name}")
        primary = item.get("primary_condition")
        family = item.get("entity_family")
        if not isinstance(primary, str) or not primary:
            raise ValueError(f"primary_condition kosong: {name}")
        if not isinstance(family, str) or not family:
            raise ValueError(f"entity_family kosong: {name}")
        primary_values.append(primary)
        family_values.append(family)

    # Validate that each primary_condition has exactly one parent entity_family,
    # otherwise primary_condition would not define a valid tree level.
    parent_by_primary: dict[str, str] = {}
    for primary, family in zip(primary_values, family_values):
        previous = parent_by_primary.setdefault(primary, family)
        if previous != family:
            raise ValueError(
                f"primary_condition {primary!r} memiliki lebih dari satu entity_family"
            )

    def categories_in_first_occurrence(values: Sequence[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
        lookup: dict[str, int] = {}
        mapping: list[int] = []
        for value in values:
            if value not in lookup:
                lookup[value] = len(lookup)
            mapping.append(lookup[value])
        return tuple(lookup.keys()), tuple(mapping)

    primary_categories, primary_map = categories_in_first_occurrence(primary_values)
    family_categories, family_map = categories_in_first_occurrence(family_values)
    leaf_categories = tuple(names)
    leaf_map = tuple(range(len(names)))
    root_categories = ("coffee_quality_sample",)
    root_map = tuple(0 for _ in names)
    level_categories = (
        leaf_categories,
        primary_categories,
        family_categories,
        root_categories,
    )
    class_to_level = (leaf_map, primary_map, family_map, root_map)
    # ExpertDet Eq. (10): gamma_l = L-l-1; root therefore receives zero weight.
    L = len(level_categories)
    gamma = tuple(float(L - level - 1) for level in range(L))
    return HierarchySpec(
        class_names=tuple(names),
        level_names=("fine_class", "primary_condition", "entity_family", "root"),
        level_categories=level_categories,
        class_to_level=class_to_level,
        gamma=gamma,
    )


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class HierarchicalPrototypeTree(nn.Module):
    """ExpertDet HierVIP Eqs. (4)-(10) over a frozen hierarchy."""

    def __init__(self, hierarchy: HierarchySpec, config: HierVIPConfig) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.config = config
        for level, categories in enumerate(hierarchy.level_categories):
            self.register_buffer(
                f"prototypes_{level}",
                torch.zeros(len(categories), config.embedding_dim),
                persistent=True,
            )
            self.register_buffer(
                f"active_{level}",
                torch.zeros(len(categories), dtype=torch.bool),
                persistent=True,
            )
            self.register_buffer(
                f"class_to_level_{level}",
                torch.tensor(hierarchy.class_to_level[level], dtype=torch.long),
                persistent=True,
            )

    def prototypes(self, level: int) -> torch.Tensor:
        return getattr(self, f"prototypes_{level}")

    def active(self, level: int) -> torch.Tensor:
        return getattr(self, f"active_{level}")

    def class_to_level(self, level: int) -> torch.Tensor:
        return getattr(self, f"class_to_level_{level}")

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, fine_labels: torch.Tensor) -> dict[str, float]:
        """Sequential adaptive-momentum update following ExpertDet Eqs. (4)-(7)."""

        z = F.normalize(embeddings.detach(), dim=1, eps=1e-8)
        labels = fine_labels.to(device=z.device, dtype=torch.long).reshape(-1)
        drift_values: list[float] = []
        momentum_values: list[float] = []
        for row in range(len(labels)):
            feature = z[row]
            fine = int(labels[row])
            for level in range(self.hierarchy.num_levels):
                category = int(self.class_to_level(level)[fine])
                prototypes = self.prototypes(level)
                active = self.active(level)
                if not bool(active[category]):
                    prototypes[category].copy_(feature)
                    active[category] = True
                    continue
                current = F.normalize(prototypes[category], dim=0, eps=1e-8)
                drift = float((1.0 - torch.dot(current, feature)).clamp(min=0.0, max=2.0))
                momentum = max(
                    self.config.momentum_low,
                    min(
                        self.config.momentum_base,
                        self.config.momentum_base - self.config.drift_alpha * drift,
                    ),
                )
                updated = momentum * current + (1.0 - momentum) * feature
                prototypes[category].copy_(F.normalize(updated, dim=0, eps=1e-8))
                drift_values.append(drift)
                momentum_values.append(momentum)
        return {
            "mean_drift": sum(drift_values) / max(len(drift_values), 1),
            "mean_momentum": sum(momentum_values) / max(len(momentum_values), 1),
        }

    def loss(self, embeddings: torch.Tensor, fine_labels: torch.Tensor) -> torch.Tensor:
        """ExpertDet HSC Eq. (10), restricted to activated prototypes per Eq. (9)."""

        if embeddings.ndim != 2 or embeddings.shape[1] != self.config.embedding_dim:
            raise ValueError(
                f"embeddings harus [N,{self.config.embedding_dim}], diterima {tuple(embeddings.shape)}"
            )
        labels = fine_labels.to(device=embeddings.device, dtype=torch.long).reshape(-1)
        if len(labels) != len(embeddings):
            raise ValueError("Jumlah embedding dan label tidak sama")
        if not len(labels):
            return embeddings.sum() * 0.0
        if int(labels.min()) < 0 or int(labels.max()) >= self.hierarchy.num_classes:
            raise ValueError("Label HierVIP di luar rentang kelas")

        z = F.normalize(embeddings, dim=1, eps=1e-8)
        numerator = z.new_zeros(())
        gamma_sum = sum(self.hierarchy.gamma)
        if gamma_sum <= 0:
            raise RuntimeError("Jumlah gamma HierVIP harus positif")
        for level, gamma in enumerate(self.hierarchy.gamma):
            if gamma <= 0:
                continue
            active = self.active(level)
            active_indices = torch.where(active)[0]
            if not len(active_indices):
                continue
            prototypes = F.normalize(
                self.prototypes(level).detach()[active_indices], dim=1, eps=1e-8
            )
            logits = (z @ prototypes.t()) / self.config.temperature
            category_targets = self.class_to_level(level)[labels]
            # Convert absolute category id -> column among activated prototypes.
            remap = torch.full(
                (len(self.hierarchy.level_categories[level]),),
                -1,
                device=z.device,
                dtype=torch.long,
            )
            remap[active_indices] = torch.arange(len(active_indices), device=z.device)
            targets = remap[category_targets]
            valid = targets >= 0
            if bool(valid.any()):
                numerator = numerator + float(gamma) * F.cross_entropy(
                    logits[valid], targets[valid], reduction="mean"
                )
        return numerator / float(gamma_sum)

    def update_and_loss(self, embeddings: torch.Tensor, fine_labels: torch.Tensor) -> torch.Tensor:
        self.update(embeddings, fine_labels)
        return self.loss(embeddings, fine_labels)


class HierVIPProjectionHead(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int],
        hierarchy: HierarchySpec,
        config: HierVIPConfig,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("HierVIP memerlukan P3/P4/P5")
        self.config = config
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(channel), config.embedding_dim, 1, bias=False),
                    nn.BatchNorm2d(config.embedding_dim),
                    nn.SiLU(inplace=True),
                )
                for channel in channels
            ]
        )
        self.prototype_tree = HierarchicalPrototypeTree(hierarchy, config)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError("HierVIP memerlukan tepat tiga feature levels")
        rows = []
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            batch = value.shape[0]
            rows.append(value.view(batch, self.config.embedding_dim, -1).transpose(1, 2))
        return torch.cat(rows, dim=1)


class HierVIPDetectHead(nn.Module):
    """Native YOLO26 Detect with training-only ExpertDet HierVIP supervision."""

    def __init__(self, base_head: nn.Module, hierarchy: HierarchySpec, config: HierVIPConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("HierVIP memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("HierVIP dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("HierVIP memerlukan tiga level P3/P4/P5")
        if int(base_head.nc) != hierarchy.num_classes:
            raise ValueError("Hierarchy dan jumlah fine classes tidak cocok")
        self.base_head = base_head
        self.hierarchy = hierarchy
        self.config = config
        self.hiervip = HierVIPProjectionHead(channels, hierarchy, config)
        for name in ("i", "f", "type", "np"):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))
        for name in (
            "nc", "nl", "reg_max", "stride", "end2end", "max_det", "export",
            "format", "dynamic", "agnostic_nms",
        ):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))

    @property
    def one2many(self):
        return self.base_head.one2many

    @property
    def one2one(self):
        return self.base_head.one2one

    def _sync_runtime_attributes(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    def _forward_branch(self, features: list[torch.Tensor], branch: dict[str, nn.Module], *, include_hierarchy: bool) -> dict[str, torch.Tensor]:
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](features[index]))
        batch = features[0].shape[0]
        output = {
            "boxes": torch.cat([value.view(batch, 4 * self.reg_max, -1) for value in boxes], dim=-1),
            "scores": torch.cat([value.view(batch, self.nc, -1) for value in logits], dim=-1),
            "feats": features,
        }
        if include_hierarchy:
            output["hiervip_embeddings"] = self.hiervip(features)
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many, include_hierarchy=True),
                "one2one": self._forward_branch(
                    [value.detach() for value in features], self.one2one, include_hierarchy=False
                ),
            }
        # ExpertDet discards HierVIP at inference; native YOLO path is exact.
        return self.base_head(features)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_hiervip_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, HierVIPDetectHead):
        raise TypeError("Target bukan HierVIPDetectHead")
    if isinstance(source_head, HierVIPDetectHead):
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}
