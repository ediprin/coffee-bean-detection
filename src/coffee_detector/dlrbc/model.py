from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class DLRBCConfig:
    """Frozen dense transfer of Kong-Fowlkes low-rank bilinear scoring.

    ``linear`` is the capacity-matched first-order control. ``quadratic``
    splits the total rank evenly into positive and negative factors and uses
    their signed energy difference. Both modes retain the native YOLO class
    logit and add a residual computed from the native class tower feature.
    """

    mode: str = "quadratic"
    rank: int = 8
    projection_ratio: float = 0.5
    minimum_projection: int = 16
    residual_scale: float = 0.1
    signed_sqrt: bool = True
    eps: float = 1.0e-6

    @classmethod
    def from_mapping(
        cls, payload: "DLRBCConfig | Mapping[str, Any] | None"
    ) -> "DLRBCConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.mode not in {"linear", "quadratic"}:
            raise ValueError(f"mode DLRBC tidak dikenal: {result.mode}")
        if result.rank <= 0 or result.rank % 2:
            raise ValueError("rank harus positif dan genap")
        if not 0.0 < result.projection_ratio < 1.0:
            raise ValueError("projection_ratio harus berada di (0, 1)")
        if result.minimum_projection < result.rank:
            raise ValueError("minimum_projection tidak boleh lebih kecil dari rank")
        if result.residual_scale <= 0.0:
            raise ValueError("residual_scale harus positif")
        if result.eps <= 0.0:
            raise ValueError("eps harus positif")
        return result

    def projection_channels(self, input_channels: int) -> int:
        input_channels = int(input_channels)
        projected = max(
            int(self.minimum_projection),
            int(round(input_channels * float(self.projection_ratio))),
        )
        projected = min(projected, input_channels - 1)
        if projected < self.rank:
            raise ValueError(
                f"Head terlalu sempit untuk low-rank reduction: c={input_channels}, "
                f"m={projected}, r={self.rank}"
            )
        return projected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LowRankClassResidual(nn.Module):
    """Matched linear/quadratic residual over a dense class-tower feature map."""

    def __init__(self, channels: int, num_classes: int, config: DLRBCConfig) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.config = config
        self.projection_channels = config.projection_channels(self.channels)
        self.projection = nn.Conv2d(
            self.channels, self.projection_channels, kernel_size=1, bias=False
        )
        self.factors = nn.Conv2d(
            self.projection_channels,
            self.num_classes * int(config.rank),
            kernel_size=1,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(self.num_classes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.projection.weight.flatten(1))
        nn.init.xavier_uniform_(self.factors.weight)
        nn.init.zeros_(self.bias)

    def normalized_feature(self, value: torch.Tensor) -> torch.Tensor:
        if not self.config.signed_sqrt:
            return value
        # Sign-sqrt is the feature-map normalization used by the implicit
        # low-rank configuration in Kong & Fowlkes. Clamp-free abs+eps keeps
        # the derivative finite around zero for AMP training.
        return value.sign() * torch.sqrt(value.abs() + float(self.config.eps))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.channels:
            raise ValueError(
                f"DLRBC membutuhkan [B,{self.channels},H,W], diterima {tuple(value.shape)}"
            )
        projected = self.projection(self.normalized_feature(value))
        factor = self.factors(projected)
        batch, _, height, width = factor.shape
        factor = factor.view(
            batch, self.num_classes, int(self.config.rank), height, width
        )
        if self.config.mode == "linear":
            score = factor.sum(dim=2) / math.sqrt(float(self.config.rank))
        else:
            half = int(self.config.rank) // 2
            positive = factor[:, :, :half].square().sum(dim=2)
            negative = factor[:, :, half:].square().sum(dim=2)
            score = (positive - negative) / math.sqrt(float(half))
        score = score + self.bias.view(1, -1, 1, 1)
        return float(self.config.residual_scale) * score


def _class_tower_feature(branch: nn.Module, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(branch, nn.Sequential) or len(branch) < 2:
        raise TypeError("Cabang klasifikasi YOLO harus nn.Sequential")
    tower = value
    for layer in list(branch.children())[:-1]:
        tower = layer(tower)
    classifier = list(branch.children())[-1]
    if not isinstance(classifier, nn.Conv2d):
        raise TypeError("Layer terakhir cabang klasifikasi harus Conv2d")
    return tower, classifier(tower)


def _classifier_channels(branch: nn.Module) -> int:
    if not isinstance(branch, nn.Sequential) or not branch:
        raise TypeError("Cabang klasifikasi tidak sesuai kontrak Detect")
    classifier = list(branch.children())[-1]
    if not isinstance(classifier, nn.Conv2d):
        raise TypeError("Classifier akhir bukan Conv2d")
    return int(classifier.in_channels)


class DLRBCDetectHead(nn.Module):
    """Native YOLO26 boxes/classes plus matched low-rank class residuals."""

    def __init__(self, base_head: nn.Module, config: DLRBCConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("DLRBC memerlukan native YOLO26 end-to-end Detect")
        if len(base_head.cv3) != 3 or len(base_head.one2one_cv3) != 3:
            raise ValueError("DLRBC dikunci untuk P3/P4/P5")
        self.base_head = base_head
        self.config = config
        channels = tuple(_classifier_channels(branch) for branch in base_head.cv3)
        one2one_channels = tuple(
            _classifier_channels(branch) for branch in base_head.one2one_cv3
        )
        if channels != one2one_channels or len(set(channels)) != 1:
            raise ValueError(
                f"Lebar one-to-many/one-to-one tidak cocok: {channels} vs {one2one_channels}"
            )
        self.class_tower_channels = channels
        self.one2many_residuals = nn.ModuleList(
            [LowRankClassResidual(c, int(base_head.nc), config) for c in channels]
        )
        self.one2one_residuals = nn.ModuleList(
            [LowRankClassResidual(c, int(base_head.nc), config) for c in channels]
        )
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

    def _forward_branch(
        self,
        features: list[torch.Tensor],
        branch: dict[str, nn.Module],
        residuals: nn.ModuleList,
    ) -> dict[str, torch.Tensor]:
        boxes, scores = [], []
        for index in range(self.nl):
            feature = features[index]
            boxes.append(branch["box_head"][index](feature))
            tower, native = _class_tower_feature(branch["cls_head"][index], feature)
            scores.append(native + residuals[index](tower))
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat(
                [value.view(batch, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [value.view(batch, self.nc, -1) for value in scores], dim=-1
            ),
            "feats": features,
        }

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return {
                "one2many": self._forward_branch(
                    features, self.one2many, self.one2many_residuals
                ),
                "one2one": self._forward_branch(
                    [value.detach() for value in features],
                    self.one2one,
                    self.one2one_residuals,
                ),
            }
        one2many = (
            self._forward_branch(features, self.one2many, self.one2many_residuals)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [value.detach() for value in features],
            self.one2one,
            self.one2one_residuals,
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def _shape_compatible_state(target: nn.Module, source: nn.Module) -> dict[str, torch.Tensor]:
    target_state = target.state_dict()
    return {
        key: value
        for key, value in source.state_dict().items()
        if key in target_state and target_state[key].shape == value.shape
    }


def load_dlrbc_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load official/native or resumable DLRBC weights without Coffee parents."""

    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    source_model = getattr(source, "model", None)
    target_model = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    if not isinstance(target_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos daftar layer model")
    target_head = target_model[-1]
    if not isinstance(target_head, DLRBCDetectHead):
        raise TypeError("Target bukan DLRBCDetectHead")

    source_head = source_model[-1]
    if isinstance(source_head, DLRBCDetectHead):
        model.load(weights)
        return {
            "source_items": len(source.state_dict()),
            "shape_compatible_items": len(_shape_compatible_state(model, source)),
            "resume": 1,
        }
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")

    # BaseModel.load transfers every shape-compatible backbone/neck tensor.
    model.load(weights)
    compatible = _shape_compatible_state(target_head.base_head, source_head)
    result = target_head.base_head.load_state_dict(compatible, strict=False)
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {
        "source_items": len(source.state_dict()),
        "native_head_compatible_items": len(compatible),
        "native_head_missing_items": len(result.missing_keys),
        "resume": 0,
    }


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class DLRBCDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, dlrbc=None):
        self.dlrbc_config = DLRBCConfig.from_mapping(dlrbc)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = DLRBCDetectHead(self.model[-1], self.dlrbc_config)
