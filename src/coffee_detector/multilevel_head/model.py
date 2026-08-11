from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torchvision.ops import roi_align


@dataclass(frozen=True)
class MultilevelHeadConfig:
    mode: str = "pyramid_fusion"
    descriptor_dim: int = 512
    roi_size: int = 3
    topk: int = 500
    inference_weight: float = 0.0
    box_expand: float = 1.0
    training_topk: int = 500
    auxiliary_weight: float = 0.5
    candidate_source: str = "one2one"
    positive_iou: float = 0.5
    predicted_start_epoch: int = 10
    predicted_full_epoch: int = 25

    @classmethod
    def from_mapping(
        cls, payload: "MultilevelHeadConfig" | dict[str, Any] | None
    ) -> "MultilevelHeadConfig":
        if isinstance(payload, cls):
            return payload
        result = cls(**dict(payload or {}))
        if result.mode not in {"p5_control", "pyramid_fusion"}:
            raise ValueError("mode harus p5_control atau pyramid_fusion")
        if result.descriptor_dim != 512:
            raise ValueError("Protokol CM512 mengunci descriptor_dim=512")
        if result.roi_size <= 0 or result.topk <= 0 or result.training_topk <= 0:
            raise ValueError("roi_size, topk, dan training_topk harus positif")
        if result.inference_weight < 0 or result.auxiliary_weight < 0:
            raise ValueError("Bobot multilevel head tidak boleh negatif")
        if result.box_expand < 1.0:
            raise ValueError("box_expand minimal 1.0")
        if result.candidate_source not in {"one2one", "one2many"}:
            raise ValueError("candidate_source harus one2one atau one2many")
        if not 0.0 < result.positive_iou <= 1.0:
            raise ValueError("positive_iou harus berada pada (0,1]")
        if (
            result.predicted_start_epoch < 0
            or result.predicted_full_epoch <= result.predicted_start_epoch
        ):
            raise ValueError(
                "predicted_full_epoch harus lebih besar dari predicted_start_epoch"
            )
        return result


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


def _expand_and_clip_boxes(
    boxes: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    factor: float,
) -> torch.Tensor:
    boxes = boxes.detach().clone()
    centres = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0) * float(factor)
    boxes[..., :2] = centres - sizes * 0.5
    boxes[..., 2:] = centres + sizes * 0.5
    boxes[..., 0::2].clamp_(0, max(image_width - 1, 0))
    boxes[..., 1::2].clamp_(0, max(image_height - 1, 0))
    boxes[..., 2] = torch.maximum(boxes[..., 2], boxes[..., 0] + 1.0)
    boxes[..., 3] = torch.maximum(boxes[..., 3], boxes[..., 1] + 1.0)
    boxes[..., 2].clamp_(max=float(image_width))
    boxes[..., 3].clamp_(max=float(image_height))
    return boxes


class CapacityMatchedROIClassifier(nn.Module):
    """P5 control and P3-P5 fusion with an identical parameter schema."""

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
        config: MultilevelHeadConfig,
    ) -> None:
        super().__init__()
        if tuple(channels) != (64, 128, 256):
            raise ValueError(
                f"CM512 protocol dikunci untuk kanal D0 (64,128,256), diterima {channels}"
            )
        self.config = config
        descriptor_dims = tuple(2 * channel for channel in channels)
        if descriptor_dims != (128, 256, 512):
            raise AssertionError(descriptor_dims)
        self.level_norms = nn.ModuleList(
            [nn.LayerNorm(dimension) for dimension in descriptor_dims]
        )
        self.p5_side = nn.Linear(512, 384, bias=False)
        self.projection = nn.Linear(896, config.descriptor_dim, bias=False)
        self.hidden_norm = nn.LayerNorm(config.descriptor_dim)
        self.activation = nn.SiLU(inplace=False)
        self.classifier = nn.Linear(config.descriptor_dim, int(num_classes))

    def _descriptor(
        self,
        feature: torch.Tensor,
        rois: torch.Tensor,
        stride: float,
    ) -> torch.Tensor:
        aligned = roi_align(
            feature,
            rois.to(device=feature.device, dtype=feature.dtype),
            output_size=(self.config.roi_size, self.config.roi_size),
            spatial_scale=1.0 / float(stride),
            sampling_ratio=2,
            aligned=True,
        )
        return torch.cat(
            (aligned.mean(dim=(-2, -1)), aligned.amax(dim=(-2, -1))), dim=1
        )

    def forward(
        self,
        features: list[torch.Tensor],
        rois: torch.Tensor,
        strides: tuple[float, float, float],
    ) -> torch.Tensor:
        if len(features) != 3 or len(strides) != 3:
            raise ValueError("CM512 memerlukan tepat tiga feature level")
        if rois.ndim != 2 or rois.shape[1] != 5:
            raise ValueError(f"ROI harus [N,5], diterima {tuple(rois.shape)}")
        if not len(rois):
            return features[0].new_zeros((0, self.classifier.out_features))
        descriptors = [
            normalization(self._descriptor(feature, rois, stride))
            for normalization, feature, stride in zip(
                self.level_norms, features, strides
            )
        ]
        p3_p4 = torch.cat(descriptors[:2], dim=1)
        p5 = descriptors[2]
        side = self.p5_side(p5)
        if self.config.mode == "pyramid_fusion":
            side = side + p3_p4
        combined = torch.cat((side, p5), dim=1)
        hidden = self.activation(self.hidden_norm(self.projection(combined)))
        return self.classifier(hidden)


class MultilevelResidualDetectHead(nn.Module):
    """Wrap YOLO26 Detect while preserving all native localization branches."""

    def __init__(self, base_head: nn.Module, config: MultilevelHeadConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("Multilevel head memerlukan native Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("Multilevel head dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("Multilevel head memerlukan P3-P5")
        self.base_head = base_head
        self.config = config
        self.refiner = CapacityMatchedROIClassifier(
            channels, int(base_head.nc), config
        )
        self.proposal_mix = 0.0
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

    def _refine_scores(
        self,
        predictions: dict[str, torch.Tensor],
        features: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        scores = predictions["scores"]
        confidence = scores.detach().amax(dim=1)
        count = min(int(self.config.topk), int(confidence.shape[1]))
        indices = confidence.topk(count, dim=1).indices
        boxes = self.base_head._get_decode_boxes(predictions).transpose(1, 2)
        selected_boxes = boxes.gather(1, indices[..., None].expand(-1, -1, 4))
        image_height = int(features[0].shape[-2] * float(self.stride[0]))
        image_width = int(features[0].shape[-1] * float(self.stride[0]))
        selected_boxes = _expand_and_clip_boxes(
            selected_boxes,
            image_height=image_height,
            image_width=image_width,
            factor=self.config.box_expand,
        )
        rois = []
        for batch_index in range(selected_boxes.shape[0]):
            column = selected_boxes.new_full((count, 1), float(batch_index))
            rois.append(torch.cat((column, selected_boxes[batch_index]), dim=1))
        residual = self.refiner(
            features,
            torch.cat(rois, dim=0),
            tuple(float(value) for value in self.stride),
        ).view(scores.shape[0], count, self.nc)
        by_anchor = scores.transpose(1, 2)
        selected = by_anchor.gather(
            1, indices[..., None].expand(-1, -1, self.nc)
        )
        selected = selected + float(self.config.inference_weight) * residual
        by_anchor = by_anchor.scatter(
            1, indices[..., None].expand(-1, -1, self.nc), selected
        )
        output = dict(predictions)
        output["scores"] = by_anchor.transpose(1, 2)
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return self.base_head(features)
        one2many = self.base_head.forward_head(features, **self.base_head.one2many)
        detached = [feature.detach() for feature in features]
        one2one = self.base_head.forward_head(detached, **self.base_head.one2one)
        one2one = self._refine_scores(one2one, features)
        predictions = {"one2many": one2many, "one2one": one2one}
        output = self.base_head._inference(one2one)
        output = self.base_head.postprocess(output.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_multilevel_head(
    model: nn.Module,
    config: MultilevelHeadConfig | dict[str, Any] | None,
) -> int:
    frozen = MultilevelHeadConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], MultilevelResidualDetectHead):
        return 0
    detector[-1] = MultilevelResidualDetectHead(detector[-1], frozen)
    return 1


def config_dict(head: MultilevelResidualDetectHead) -> dict[str, Any]:
    return asdict(head.config)


def load_multilevel_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load a native YOLO checkpoint into a wrapped multilevel detector exactly.

    Ultralytics' generic partial loader cannot map ``model.N.cv*`` from a
    native Detect head to ``model.N.base_head.cv*`` after the wrapper changes
    the state-dict namespace.  Explicitly remapping the native head prevents a
    silently reinitialized detector head in future multilevel runs.
    """

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    if not isinstance(target, (nn.Sequential, nn.ModuleList)) or not len(target):
        raise TypeError("Target tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, MultilevelResidualDetectHead):
        raise TypeError("Target bukan MultilevelResidualDetectHead")
    if isinstance(source_head, MultilevelResidualDetectHead):
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


class MultilevelHeadDetectionModel(DetectionModel):
    """YOLO26 DetectionModel with the serializable capacity-matched refiner."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        multilevel_head: MultilevelHeadConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.multilevel_head_config = MultilevelHeadConfig.from_mapping(
            multilevel_head
        )
        inject_multilevel_head(self, self.multilevel_head_config)

    def init_criterion(self):
        from .loss import MultilevelHeadLoss

        return MultilevelHeadLoss(self)
