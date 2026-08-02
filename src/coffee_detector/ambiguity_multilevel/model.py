"""Native one-stage P3--P5 classification correction for YOLO26."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AmbiguityMultilevelConfig:
    hidden_dim: int = 64
    context_kernel: int = 3
    correction_scale: float = 1.0
    ambiguity_floor: float = 0.0

    @classmethod
    def from_mapping(
        cls, payload: "AmbiguityMultilevelConfig" | dict[str, Any] | None
    ) -> "AmbiguityMultilevelConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.hidden_dim <= 0:
            raise ValueError("hidden_dim harus positif")
        if result.context_kernel not in {3, 5}:
            raise ValueError("context_kernel harus 3 atau 5")
        if result.correction_scale <= 0:
            raise ValueError("correction_scale harus positif")
        if not 0.0 <= result.ambiguity_floor < 1.0:
            raise ValueError("ambiguity_floor harus berada pada [0,1)")
        return result


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class _DepthwiseContext(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=False
        )
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.activation(self.norm(self.depthwise(value))))


class AmbiguityConditionedFusion(nn.Module):
    """Select P3/P4/P5 evidence separately for every classification grid cell."""

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
        config: AmbiguityMultilevelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        self.projections = nn.ModuleList(
            [nn.Conv2d(channel, config.hidden_dim, 1, bias=False) for channel in channels]
        )
        # Concatenated P3/P4/P5 descriptors plus detached normalized leaf entropy.
        self.level_selectors = nn.ModuleList(
            [nn.Conv2d(3 * config.hidden_dim + 1, 3, 1) for _ in channels]
        )
        self.contexts = nn.ModuleList(
            [_DepthwiseContext(config.hidden_dim, config.context_kernel) for _ in channels]
        )
        self.class_corrections = nn.ModuleList(
            [nn.Conv2d(config.hidden_dim, self.num_classes, 1) for _ in channels]
        )
        # Zero correction makes the injected model exactly D0 before learning.
        for layer in self.class_corrections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _ambiguity(self, logits: torch.Tensor) -> torch.Tensor:
        probability = logits.detach().softmax(dim=1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(max(self.num_classes, 2))
        if self.config.ambiguity_floor:
            entropy = (entropy - self.config.ambiguity_floor).clamp_min(0.0)
            entropy = entropy / (1.0 - self.config.ambiguity_floor)
        return entropy

    def forward(
        self, features: list[torch.Tensor], level_logits: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        if len(features) != 3 or len(level_logits) != 3:
            raise ValueError("ACMC memerlukan tepat P3, P4, dan P5")
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        corrections: list[torch.Tensor] = []
        for target, logits in enumerate(level_logits):
            size = projected[target].shape[-2:]
            aligned = [
                value if value.shape[-2:] == size else F.interpolate(value, size=size, mode="nearest")
                for value in projected
            ]
            ambiguity = self._ambiguity(logits)
            weights = self.level_selectors[target](torch.cat((*aligned, ambiguity), dim=1)).softmax(dim=1)
            fused = sum(weights[:, index : index + 1] * value for index, value in enumerate(aligned))
            correction = self.class_corrections[target](self.contexts[target](fused))
            corrections.append(float(self.config.correction_scale) * ambiguity * correction)
        return corrections


class AmbiguityMultilevelDetectHead(nn.Module):
    """Detect wrapper that changes class scores only; no ROI or decoded boxes."""

    def __init__(self, base_head: nn.Module, config: AmbiguityMultilevelConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("ACMC memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("ACMC dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("ACMC memerlukan tiga level P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.correction = AmbiguityConditionedFusion(channels, int(base_head.nc), config)
        for name in ("i", "f", "type", "np"):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))
        for name in (
            "nc", "nl", "reg_max", "stride", "end2end", "max_det", "export", "format", "dynamic", "agnostic_nms"
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

    def _forward_branch(self, features: list[torch.Tensor], branch: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](features[index]))
        corrections = self.correction(features, logits)
        batch_size = features[0].shape[0]
        return {
            "boxes": torch.cat([value.view(batch_size, 4 * self.reg_max, -1) for value in boxes], dim=-1),
            "scores": torch.cat(
                [(logit + correction).view(batch_size, self.nc, -1) for logit, correction in zip(logits, corrections)],
                dim=-1,
            ),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        one2many = self._forward_branch(features, self.one2many)
        one2one = self._forward_branch([value.detach() for value in features], self.one2one)
        predictions = {"one2many": one2many, "one2one": one2one}
        if self.training:
            return predictions
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_ambiguity_multilevel_head(
    model: nn.Module, config: AmbiguityMultilevelConfig | dict[str, Any] | None
) -> int:
    frozen = AmbiguityMultilevelConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], AmbiguityMultilevelDetectHead):
        return 0
    detector[-1] = AmbiguityMultilevelDetectHead(detector[-1], frozen)
    return 1


def load_ambiguity_multilevel_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly map a native D0 Detect state into the wrapper namespace."""
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, AmbiguityMultilevelDetectHead):
        raise TypeError("Target bukan AmbiguityMultilevelDetectHead")
    if isinstance(source_head, AmbiguityMultilevelDetectHead):
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


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore[assignment,misc]


class AmbiguityMultilevelDetectionModel(DetectionModel):
    """YOLO26 model with an end-to-end field-level classification correction."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        ambiguity_multilevel: AmbiguityMultilevelConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.ambiguity_multilevel_config = AmbiguityMultilevelConfig.from_mapping(ambiguity_multilevel)
        inject_ambiguity_multilevel_head(self, self.ambiguity_multilevel_config)


def config_dict(head: AmbiguityMultilevelDetectHead) -> dict[str, Any]:
    return asdict(head.config)
