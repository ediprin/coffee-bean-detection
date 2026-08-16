from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from coffee_detector.geometry_conditioning.model import _ordered_names


FAMILIES = ("kulit_kopi", "kulit_tanduk", "tanah_batu_ranting")
SIZE_ORDER = ("kecil", "sedang", "besar")


@dataclass(frozen=True)
class GeometryFactorizationConfig:
    shared_hidden_dim: int = 60
    family_hidden_dim: int = 35
    max_normalized_side: float = 2.0
    max_aspect_ratio: float = 10.0

    @classmethod
    def from_mapping(
        cls, payload: "GeometryFactorizationConfig | dict[str, Any] | None"
    ) -> "GeometryFactorizationConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.shared_hidden_dim <= 0 or result.family_hidden_dim <= 0:
            raise ValueError("hidden dim harus positif")
        if result.max_normalized_side <= 0:
            raise ValueError("max_normalized_side harus positif")
        if result.max_aspect_ratio <= 1.0:
            raise ValueError("max_aspect_ratio harus > 1")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _family_and_size(name: str) -> tuple[str, str] | None:
    for family in FAMILIES:
        prefix = family + "_"
        if not name.startswith(prefix):
            continue
        for size in SIZE_ORDER:
            if name.endswith("_" + size):
                return family, size
    return None


def family_class_indices(
    class_names: Mapping[int, str] | Sequence[str] | None,
    nc: int,
) -> dict[str, tuple[int, int, int]]:
    names = _ordered_names(class_names, nc)
    by_family: dict[str, dict[str, int]] = {family: {} for family in FAMILIES}
    for index, name in enumerate(names):
        parsed = _family_and_size(name)
        if parsed is None:
            continue
        family, size = parsed
        by_family[family][size] = index
    result: dict[str, tuple[int, int, int]] = {}
    for family in FAMILIES:
        missing = [size for size in SIZE_ORDER if size not in by_family[family]]
        if missing:
            raise ValueError(f"Family {family} kehilangan size classes: {missing}")
        result[family] = tuple(by_family[family][size] for size in SIZE_ORDER)
    flat = [index for family in FAMILIES for index in result[family]]
    if len(flat) != 9 or len(set(flat)) != 9:
        raise RuntimeError("Target family-size harus tepat sembilan kelas unik")
    return result


class Shared60GeometryAdapter(nn.Module):
    """Exact 849-parameter shared geometry mapping: 4 -> 60 -> 9."""

    def __init__(
        self,
        nc: int,
        config: GeometryFactorizationConfig,
        class_names: Mapping[int, str] | Sequence[str] | None,
    ) -> None:
        super().__init__()
        self.nc = int(nc)
        self.family_indices = family_class_indices(class_names, self.nc)
        target = [index for family in FAMILIES for index in self.family_indices[family]]
        self.register_buffer(
            "target_indices", torch.tensor(target, dtype=torch.long), persistent=True
        )
        self.network = nn.Sequential(
            nn.Linear(4, config.shared_hidden_dim, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(config.shared_hidden_dim, 9, bias=True),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        compact = self.network(geometry.transpose(1, 2)).transpose(1, 2)
        full = compact.new_zeros(compact.shape[0], self.nc, compact.shape[2])
        return full.index_copy(1, self.target_indices, compact)


class Family35x3GeometryAdapter(nn.Module):
    """Exact 849-parameter family factorization: 3 x (4 -> 35 -> 3)."""

    def __init__(
        self,
        nc: int,
        config: GeometryFactorizationConfig,
        class_names: Mapping[int, str] | Sequence[str] | None,
    ) -> None:
        super().__init__()
        self.nc = int(nc)
        self.family_indices = family_class_indices(class_names, self.nc)
        self.networks = nn.ModuleDict()
        for family in FAMILIES:
            network = nn.Sequential(
                nn.Linear(4, config.family_hidden_dim, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(config.family_hidden_dim, 3, bias=True),
            )
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
            self.networks[family] = network
            self.register_buffer(
                f"indices_{family}",
                torch.tensor(self.family_indices[family], dtype=torch.long),
                persistent=True,
            )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        batch, _, count = geometry.shape
        full = geometry.new_zeros(batch, self.nc, count)
        features = geometry.transpose(1, 2)
        for family in FAMILIES:
            compact = self.networks[family](features).transpose(1, 2)
            indices = getattr(self, f"indices_{family}")
            full = full.index_copy(1, indices, compact)
        return full


def _make_adapter(
    mode: str,
    nc: int,
    config: GeometryFactorizationConfig,
    class_names: Mapping[int, str] | Sequence[str] | None,
) -> nn.Module:
    if mode == "shared60":
        return Shared60GeometryAdapter(nc, config, class_names)
    if mode == "family35x3":
        return Family35x3GeometryAdapter(nc, config, class_names)
    raise ValueError("mode harus shared60 atau family35x3")


class GeometryFactorizedDetectHead(nn.Module):
    """Native YOLO26 Detect plus an exact-capacity geometry residual."""

    def __init__(
        self,
        base_head: nn.Module,
        config: GeometryFactorizationConfig,
        *,
        mode: str,
        class_names: Mapping[int, str] | Sequence[str] | None,
    ) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("Geometry factorization memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("Geometry factorization dikunci untuk YOLO26 end-to-end")
        if not hasattr(base_head, "_get_decode_boxes"):
            raise TypeError("Ultralytics Detect tidak mengekspos _get_decode_boxes")
        self.base_head = base_head
        self.config = config
        self.mode = mode
        self.adapter = _make_adapter(mode, int(base_head.nc), config, class_names)
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

    def _geometry(self, boxes: torch.Tensor, features: list[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            decoded = self.base_head._get_decode_boxes(
                {"boxes": boxes.detach(), "feats": features}
            )
            x1, y1, x2, y2 = decoded.unbind(dim=1)
            eps = torch.finfo(decoded.dtype).eps
            width = (x2 - x1).clamp_min(eps)
            height = (y2 - y1).clamp_min(eps)
            stride0 = self.base_head.stride[0].to(
                device=decoded.device, dtype=decoded.dtype
            )
            image_width = decoded.new_tensor(float(features[0].shape[-1])) * stride0
            image_height = decoded.new_tensor(float(features[0].shape[-2])) * stride0
            w = (width / image_width).clamp(eps, self.config.max_normalized_side)
            h = (height / image_height).clamp(eps, self.config.max_normalized_side)
            area = (w * h).clamp(eps, self.config.max_normalized_side**2)
            aspect = (torch.maximum(w, h) / torch.minimum(w, h)).clamp(
                1.0, self.config.max_aspect_ratio
            )
            return torch.stack((w, h, area, aspect), dim=1)

    def _forward_branch(
        self, features: list[torch.Tensor], branch: dict[str, nn.Module]
    ) -> dict[str, torch.Tensor]:
        boxes_by_level, logits_by_level = [], []
        for index in range(self.nl):
            boxes_by_level.append(branch["box_head"][index](features[index]))
            logits_by_level.append(branch["cls_head"][index](features[index]))
        batch = features[0].shape[0]
        boxes = torch.cat(
            [value.view(batch, 4 * self.reg_max, -1) for value in boxes_by_level],
            dim=-1,
        )
        native_scores = torch.cat(
            [value.view(batch, self.nc, -1) for value in logits_by_level], dim=-1
        )
        residual = self.adapter(self._geometry(boxes, features))
        return {"boxes": boxes, "scores": native_scores + residual, "feats": features}

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many),
                "one2one": self._forward_branch(
                    [value.detach() for value in features], self.one2one
                ),
            }
        one2many = (
            self._forward_branch(features, self.one2many)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch([value.detach() for value in features], self.one2one)
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_geometry_factorized_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target_model = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    if not isinstance(target_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target_model[-1]
    if not isinstance(target_head, GeometryFactorizedDetectHead):
        raise TypeError("Target bukan GeometryFactorizedDetectHead")
    if isinstance(source_head, GeometryFactorizedDetectHead):
        if source_head.mode != target_head.mode:
            raise RuntimeError("Resume silang shared/family dilarang")
        result = target_head.load_state_dict(source_head.state_dict(), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Resume geometry factorization tidak lengkap")
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Arm harus dimulai dari native D0, bukan {type(source_head).__name__}")
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


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore[assignment,misc]


class GeometryFactorizedDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        geometry_factorization: GeometryFactorizationConfig | dict[str, Any] | None = None,
        mode: str = "shared60",
        class_names: Mapping[int, str] | Sequence[str] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.geometry_factorization_config = GeometryFactorizationConfig.from_mapping(
            geometry_factorization
        )
        self.model[-1] = GeometryFactorizedDetectHead(
            self.model[-1],
            self.geometry_factorization_config,
            mode=mode,
            class_names=class_names,
        )


def config_dict(head: GeometryFactorizedDetectHead) -> dict[str, Any]:
    return asdict(head.config)
