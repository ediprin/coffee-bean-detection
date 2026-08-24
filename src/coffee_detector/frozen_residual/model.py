from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from coffee_detector.multilevel_head.model import (
    MultilevelHeadConfig,
    MultilevelResidualDetectHead,
    _expand_and_clip_boxes,
)


@dataclass(frozen=True)
class FrozenResidualConfig:
    descriptor_dim: int = 512
    roi_size: int = 3
    topk: int = 500
    training_topk: int = 500
    inference_weight: float = 1.0
    box_expand: float = 1.0
    positive_iou: float = 0.5
    preservation_weight: float = 0.25
    gate_init_probability: float = 0.01

    @classmethod
    def from_mapping(
        cls, payload: "FrozenResidualConfig" | dict[str, Any] | None
    ) -> "FrozenResidualConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.descriptor_dim != 512:
            raise ValueError("FRM1 mengunci descriptor_dim=512")
        if result.roi_size <= 0 or result.topk <= 0 or result.training_topk <= 0:
            raise ValueError("roi_size dan top-k harus positif")
        if result.inference_weight <= 0 or result.preservation_weight < 0:
            raise ValueError("Bobot FRM1 tidak valid")
        if result.box_expand < 1.0 or not 0.0 < result.positive_iou <= 1.0:
            raise ValueError("box_expand/positive_iou FRM1 tidak valid")
        if not 0.0 < result.gate_init_probability < 0.5:
            raise ValueError("gate_init_probability harus berada pada (0,0.5)")
        return result


class ConfidenceResidualGate(nn.Module):
    """Gate residuals using frozen D0 confidence, entropy, and top-2 margin."""

    def __init__(self, num_classes: int, initial_probability: float) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.linear = nn.Linear(3, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(
            self.linear.bias,
            math.log(initial_probability / (1.0 - initial_probability)),
        )

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        probability = base_logits.detach().softmax(dim=-1)
        top2 = probability.topk(min(2, self.num_classes), dim=-1).values
        confidence = top2[..., 0]
        margin = (
            top2[..., 0] - top2[..., 1]
            if self.num_classes > 1
            else top2[..., 0]
        )
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(max(self.num_classes, 2))
        features = torch.stack((1.0 - confidence, entropy, 1.0 - margin), dim=-1)
        return self.linear(features).sigmoid()


class FrozenResidualDetectHead(MultilevelResidualDetectHead):
    """D0-preserving P3-P5 residual classifier with a learned uncertainty gate."""

    def __init__(self, base_head: nn.Module, config: FrozenResidualConfig) -> None:
        multilevel = MultilevelHeadConfig(
            mode="pyramid_fusion",
            descriptor_dim=512,
            roi_size=config.roi_size,
            topk=config.topk,
            inference_weight=config.inference_weight,
            box_expand=config.box_expand,
            training_topk=config.training_topk,
            auxiliary_weight=1.0,
            candidate_source="one2one",
            positive_iou=config.positive_iou,
            predicted_start_epoch=0,
            predicted_full_epoch=1,
        )
        super().__init__(base_head, multilevel)
        self.frozen_config = config
        self.gate = ConfidenceResidualGate(
            self.nc, config.gate_init_probability
        )
        nn.init.zeros_(self.refiner.classifier.weight)
        nn.init.zeros_(self.refiner.classifier.bias)

    def apply_residual(
        self, base_logits: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = self.gate(base_logits)
        correction = (
            float(self.frozen_config.inference_weight) * gate * residual
        )
        return base_logits + correction, gate, correction

    def _refine_scores(
        self,
        predictions: dict[str, torch.Tensor],
        features: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        scores = predictions["scores"]
        confidence = scores.detach().amax(dim=1)
        count = min(int(self.frozen_config.topk), int(confidence.shape[1]))
        indices = confidence.topk(count, dim=1).indices
        boxes = self.base_head._get_decode_boxes(predictions).transpose(1, 2)
        boxes = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        image_height = int(features[0].shape[-2] * float(self.stride[0]))
        image_width = int(features[0].shape[-1] * float(self.stride[0]))
        boxes = _expand_and_clip_boxes(
            boxes,
            image_height=image_height,
            image_width=image_width,
            factor=self.frozen_config.box_expand,
        )
        rois = []
        for batch_index in range(int(boxes.shape[0])):
            column = boxes.new_full((count, 1), float(batch_index))
            rois.append(torch.cat((column, boxes[batch_index]), dim=1))
        residual = self.refiner(
            features,
            torch.cat(rois, dim=0),
            tuple(float(value) for value in self.stride),
        ).view(scores.shape[0], count, self.nc)
        by_anchor = scores.transpose(1, 2)
        selected = by_anchor.gather(
            1, indices[..., None].expand(-1, -1, self.nc)
        )
        selected, gate, _ = self.apply_residual(selected, residual)
        by_anchor = by_anchor.scatter(
            1, indices[..., None].expand(-1, -1, self.nc), selected
        )
        output = dict(predictions)
        output["scores"] = by_anchor.transpose(1, 2)
        output["frozen_residual_indices"] = indices
        output["frozen_residual_gate"] = gate
        return output


def inject_frozen_residual_head(
    model: nn.Module,
    config: FrozenResidualConfig | dict[str, Any] | None,
) -> int:
    frozen = FrozenResidualConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], FrozenResidualDetectHead):
        return 0
    detector[-1] = FrozenResidualDetectHead(detector[-1], frozen)
    return 1


def freeze_native_detector(model: nn.Module) -> dict[str, int]:
    detector = getattr(model, "model", model)
    if not isinstance(detector[-1], FrozenResidualDetectHead):
        raise TypeError("freeze_native_detector memerlukan FrozenResidualDetectHead")
    head = detector[-1]
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.refiner.parameters():
        parameter.requires_grad = True
    for parameter in head.gate.parameters():
        parameter.requires_grad = True
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def load_frozen_d0_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load matching backbone weights and explicitly preserve the native D0 head."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint D0 tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = model.model[-1]
    if not isinstance(target_head, FrozenResidualDetectHead):
        raise TypeError("Target bukan FrozenResidualDetectHead")
    if isinstance(source_head, FrozenResidualDetectHead):
        target_head.stride = source_head.stride.detach().clone()
        target_head.base_head.stride = target_head.stride
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(source_head, name):
                value = getattr(source_head, name)
                setattr(target_head, name, value)
                setattr(target_head.base_head, name, value)
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native D0 head tidak lengkap")
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


class FrozenResidualDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        frozen_residual: FrozenResidualConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.frozen_residual_config = FrozenResidualConfig.from_mapping(
            frozen_residual
        )
        inject_frozen_residual_head(self, self.frozen_residual_config)
        freeze_native_detector(self)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            detector = self.model
            head = detector[-1]
            native_modules = list(detector[:-1].modules()) + list(
                head.base_head.modules()
            )
            for module in native_modules:
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            head.refiner.train(True)
            head.gate.train(True)
        return self

    def init_criterion(self):
        from .loss import FrozenResidualLoss

        return FrozenResidualLoss(self)


def config_dict(head: FrozenResidualDetectHead) -> dict[str, Any]:
    return asdict(head.frozen_config)
