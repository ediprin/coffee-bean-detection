from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class GeometryConditioningConfig:
    hidden_dim: int = 32
    max_normalized_side: float = 2.0
    max_aspect_ratio: float = 10.0
    target_size_classes_only: bool = True

    @classmethod
    def from_mapping(
        cls, payload: "GeometryConditioningConfig | dict[str, Any] | None"
    ) -> "GeometryConditioningConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.hidden_dim <= 0:
            raise ValueError("hidden_dim harus positif")
        if result.max_normalized_side <= 0:
            raise ValueError("max_normalized_side harus positif")
        if result.max_aspect_ratio <= 1.0:
            raise ValueError("max_aspect_ratio harus > 1")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ordered_names(
    names: Mapping[int, str] | Sequence[str] | None, nc: int
) -> tuple[str, ...]:
    if names is None:
        return tuple(str(index) for index in range(nc))
    if isinstance(names, Mapping):
        return tuple(str(names[index]) for index in range(nc))
    values = tuple(str(value) for value in names)
    if len(values) != nc:
        raise ValueError(f"Jumlah nama kelas {len(values)} != nc={nc}")
    return values


def _is_size_class(name: str) -> bool:
    return name.endswith(("_kecil", "_sedang", "_besar")) or any(
        marker in name
        for marker in ("_ukuran_kecil", "_ukuran_sedang", "_ukuran_besar")
    )


class GeometryLogitAdapter(nn.Module):
    """Small residual classifier driven by detached predicted-box geometry.

    The final projection is zero-initialized so GEO1 and GEO-C0 both start
    exactly from the native D0 function. GEO-C0 owns the identical parameters
    but receives a zero-information tensor instead of local predicted geometry.
    """

    def __init__(
        self,
        nc: int,
        config: GeometryConditioningConfig,
        class_names: Mapping[int, str] | Sequence[str] | None,
    ) -> None:
        super().__init__()
        self.nc = int(nc)
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(4, config.hidden_dim, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_dim, self.nc, bias=True),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        names = _ordered_names(class_names, self.nc)
        if config.target_size_classes_only:
            mask = torch.tensor(
                [1.0 if _is_size_class(name) else 0.0 for name in names],
                dtype=torch.float32,
            )
            if not bool(mask.any()):
                raise ValueError("Tidak menemukan kelas size-defined untuk GEO1")
        else:
            mask = torch.ones(self.nc, dtype=torch.float32)
        self.register_buffer("class_mask", mask.view(1, self.nc, 1), persistent=True)
        self.class_names = names

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if geometry.ndim != 3 or geometry.shape[1] != 4:
            raise ValueError(f"geometry harus Bx4xN, got={tuple(geometry.shape)}")
        residual = self.network(geometry.transpose(1, 2)).transpose(1, 2)
        return residual * self.class_mask.to(
            dtype=residual.dtype, device=residual.device
        )


class GeometryConditionedDetectHead(nn.Module):
    """Native YOLO26 box path plus a geometry-conditioned class-logit residual.

    Geometry is decoded from each branch's own predicted boxes. Raw box logits
    are detached before decoding, so the added classification path has no direct
    gradient into box regression. The native box tensor is returned unchanged.
    `signal_mode='zero'` is the parameter-matched zero-information GEO-C0 arm.
    """

    def __init__(
        self,
        base_head: nn.Module,
        config: GeometryConditioningConfig,
        *,
        signal_mode: str,
        class_names: Mapping[int, str] | Sequence[str] | None,
    ) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("Geometry conditioning memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("Geometry conditioning dikunci untuk YOLO26 end-to-end")
        if signal_mode not in {"geometry", "zero"}:
            raise ValueError("signal_mode harus 'geometry' atau 'zero'")
        if not hasattr(base_head, "_get_decode_boxes"):
            raise TypeError("Ultralytics Detect tidak mengekspos _get_decode_boxes")
        self.base_head = base_head
        self.config = config
        self.signal_mode = signal_mode
        self.adapter = GeometryLogitAdapter(int(base_head.nc), config, class_names)
        for name in ("i", "f", "type", "np"):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))
        for name in (
            "nc",
            "nl",
            "reg_max",
            "stride",
            "end2end",
            "max_det",
            "export",
            "format",
            "dynamic",
            "agnostic_nms",
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

    def _geometry(
        self, boxes: torch.Tensor, features: list[torch.Tensor]
    ) -> torch.Tensor:
        # YOLO26 end-to-end decode yields xyxy boxes. The raw branch output is
        # detached before decode, which blocks classification gradients to box.
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
            w = (width / image_width).clamp(
                eps, self.config.max_normalized_side
            )
            h = (height / image_height).clamp(
                eps, self.config.max_normalized_side
            )
            area = (w * h).clamp(
                eps, self.config.max_normalized_side**2
            )
            aspect = (
                torch.maximum(w, h) / torch.minimum(w, h)
            ).clamp(1.0, self.config.max_aspect_ratio)
            geometry = torch.stack((w, h, area, aspect), dim=1)
            if self.signal_mode == "zero":
                geometry = torch.zeros_like(geometry)
        return geometry

    def _forward_branch(
        self, features: list[torch.Tensor], branch: dict[str, nn.Module]
    ) -> dict[str, torch.Tensor]:
        boxes_by_level, logits_by_level = [], []
        for index in range(self.nl):
            boxes_by_level.append(branch["box_head"][index](features[index]))
            logits_by_level.append(branch["cls_head"][index](features[index]))
        batch = features[0].shape[0]
        boxes = torch.cat(
            [
                value.view(batch, 4 * self.reg_max, -1)
                for value in boxes_by_level
            ],
            dim=-1,
        )
        native_scores = torch.cat(
            [value.view(batch, self.nc, -1) for value in logits_by_level],
            dim=-1,
        )
        residual = self.adapter(self._geometry(boxes, features))
        return {
            "boxes": boxes,
            "scores": native_scores + residual,
            "feats": features,
        }

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
        one2one = self._forward_branch(
            [value.detach() for value in features], self.one2one
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_geometry_conditioned_weights(
    model: nn.Module, weights: Any
) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target_model = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    if not isinstance(target_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target_model[-1]
    if not isinstance(target_head, GeometryConditionedDetectHead):
        raise TypeError("Target bukan GeometryConditionedDetectHead")
    if isinstance(source_head, GeometryConditionedDetectHead):
        if source_head.signal_mode != target_head.signal_mode:
            raise RuntimeError("Resume GEO-C0/GEO1 silang mode dilarang")
        result = target_head.load_state_dict(source_head.state_dict(), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Resume geometry head tidak lengkap")
        return {
            "native_head_items": len(target_head.base_head.state_dict()),
            "resume": 1,
        }
    if type(source_head).__name__ != "Detect":
        raise TypeError(
            f"Geometry arm harus dimulai dari native D0, bukan {type(source_head).__name__}"
        )
    result = target_head.base_head.load_state_dict(
        source_head.state_dict(), strict=True
    )
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect ke geometry wrapper tidak lengkap")
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


class GeometryConditionedDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        geometry_conditioning: GeometryConditioningConfig
        | dict[str, Any]
        | None = None,
        signal_mode: str = "geometry",
        class_names: Mapping[int, str] | Sequence[str] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.geometry_conditioning_config = (
            GeometryConditioningConfig.from_mapping(geometry_conditioning)
        )
        self.model[-1] = GeometryConditionedDetectHead(
            self.model[-1],
            self.geometry_conditioning_config,
            signal_mode=signal_mode,
            class_names=class_names,
        )


def config_dict(head: GeometryConditionedDetectHead) -> dict[str, Any]:
    return asdict(head.config)
