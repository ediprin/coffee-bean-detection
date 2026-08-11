from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align


@dataclass(frozen=True)
class CoffeeFGConfig:
    """Configuration frozen into a CoffeeFG checkpoint."""

    mode: str = "bilinear"
    rank: int = 64
    roi_size: int = 7
    topk: int = 128
    training_topk: int = 128
    feature_levels: tuple[int, int] = (0, 1)
    auxiliary_weight: float = 0.5
    inference_weight: float = 0.5
    box_expand: float = 1.10
    candidate_source: str = "one2one"
    positive_iou: float = 0.50
    predicted_start_epoch: int = 10
    predicted_full_epoch: int = 25

    @classmethod
    def from_mapping(
        cls, payload: "CoffeeFGConfig" | dict[str, Any] | None
    ) -> "CoffeeFGConfig":
        if isinstance(payload, cls):
            return payload
        payload = dict(payload or {})
        if "feature_levels" in payload:
            payload["feature_levels"] = tuple(int(value) for value in payload["feature_levels"])
        result = cls(**payload)
        if result.mode not in {"first_order", "bilinear"}:
            raise ValueError("coffee_fg.mode harus first_order atau bilinear")
        if (
            result.rank <= 0
            or result.roi_size <= 0
            or result.topk <= 0
            or result.training_topk <= 0
        ):
            raise ValueError(
                "rank, roi_size, topk, dan training_topk CoffeeFG harus positif"
            )
        if len(result.feature_levels) != 2 or result.feature_levels[0] == result.feature_levels[1]:
            raise ValueError("feature_levels harus memuat dua level yang berbeda")
        if result.auxiliary_weight < 0 or result.inference_weight < 0:
            raise ValueError("Bobot CoffeeFG tidak boleh negatif")
        if result.box_expand < 1.0:
            raise ValueError("box_expand CoffeeFG minimal 1.0")
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
    raise TypeError(f"Tidak menemukan Conv2d pada branch {type(module).__name__}")


def _expand_and_clip_boxes(
    boxes: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    factor: float,
) -> torch.Tensor:
    """Expand XYXY boxes around their centres and make every ROI valid."""

    boxes = boxes.detach().clone()
    centres = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0) * float(factor)
    boxes[..., :2] = centres - sizes * 0.5
    boxes[..., 2:] = centres + sizes * 0.5
    boxes[..., 0::2].clamp_(0, max(int(image_width) - 1, 0))
    boxes[..., 1::2].clamp_(0, max(int(image_height) - 1, 0))
    boxes[..., 2] = torch.maximum(boxes[..., 2], boxes[..., 0] + 1.0)
    boxes[..., 3] = torch.maximum(boxes[..., 3], boxes[..., 1] + 1.0)
    boxes[..., 2].clamp_(max=float(image_width))
    boxes[..., 3].clamp_(max=float(image_height))
    return boxes


class MultiLevelROIRefiner(nn.Module):
    """Classify an object ROI using two aligned pyramid levels.

    ``first_order`` concatenates the spatial means from both levels.
    ``bilinear`` multiplies aligned projected feature maps before pooling.
    Both modes contain exactly two rank-by-rank fusion matrices, so their
    parameter counts are identical for the same input channels and rank.
    """

    def __init__(
        self,
        channels: tuple[int, ...],
        num_classes: int,
        config: CoffeeFGConfig,
    ) -> None:
        super().__init__()
        first, second = config.feature_levels
        if min(first, second) < 0 or max(first, second) >= len(channels):
            raise ValueError(
                f"feature_levels={config.feature_levels} di luar {len(channels)} feature maps"
            )
        self.config = config
        self.num_classes = int(num_classes)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[index], config.rank, 1, bias=False),
                    nn.BatchNorm2d(config.rank),
                    nn.SiLU(inplace=True),
                )
                for index in config.feature_levels
            ]
        )
        if config.mode == "first_order":
            self.fusion = nn.Linear(config.rank * 2, config.rank, bias=False)
            self.capacity_match = nn.Identity()
        else:
            self.fusion = nn.Linear(config.rank, config.rank, bias=False)
            self.capacity_match = nn.Linear(config.rank, config.rank, bias=False)
        self.activation = nn.SiLU(inplace=True)
        self.classifier = nn.Linear(config.rank, self.num_classes)

    def forward(
        self,
        features: list[torch.Tensor],
        rois: torch.Tensor,
        strides: tuple[float, ...],
    ) -> torch.Tensor:
        if rois.ndim != 2 or rois.shape[1] != 5:
            raise ValueError(f"ROI harus berbentuk [N,5], diterima {tuple(rois.shape)}")
        if not len(rois):
            return features[0].new_zeros((0, self.num_classes))

        aligned = []
        for projection, level in zip(self.projections, self.config.feature_levels):
            projected = projection(features[level])
            aligned.append(
                roi_align(
                    projected,
                    rois.to(device=projected.device, dtype=projected.dtype),
                    output_size=(self.config.roi_size, self.config.roi_size),
                    spatial_scale=1.0 / float(strides[level]),
                    sampling_ratio=2,
                    aligned=True,
                )
            )

        if self.config.mode == "first_order":
            descriptor = torch.cat([item.mean(dim=(-2, -1)) for item in aligned], dim=1)
        else:
            descriptor = (aligned[0] * aligned[1]).mean(dim=(-2, -1))
            descriptor = torch.sign(descriptor) * torch.sqrt(torch.abs(descriptor) + 1e-8)
            descriptor = F.normalize(descriptor, p=2, dim=1, eps=1e-8)

        descriptor = self.activation(self.fusion(descriptor))
        descriptor = self.activation(self.capacity_match(descriptor))
        return self.classifier(descriptor)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CoffeeFGDetectHead(nn.Module):
    """Wrap an Ultralytics Detect head while leaving box regression untouched."""

    def __init__(self, base_head: nn.Module, config: CoffeeFGConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError(
                "CoffeeFG v1 memerlukan head Detect. Segment/mask-guided adalah tahap terpisah."
            )
        if not getattr(base_head, "end2end", False):
            raise ValueError("CoffeeFG v1 dikunci untuk YOLO26 end-to-end")

        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.base_head = base_head
        self.config = config
        self.refiner = MultiLevelROIRefiner(channels, int(base_head.nc), config)
        self.proposal_mix = 0.0

        # Metadata consumed by Ultralytics BaseModel, criterion, and validators.
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
    def one2many(self) -> dict[str, nn.Module]:
        return self.base_head.one2many

    @property
    def one2one(self) -> dict[str, nn.Module]:
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
        anchor_scores = scores.detach().amax(dim=1)
        count = min(int(self.config.topk), int(anchor_scores.shape[1]))
        indices = anchor_scores.topk(count, dim=1).indices

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
        roi_tensor = torch.cat(rois, dim=0)
        residual = self.refiner(
            features,
            roi_tensor,
            tuple(float(value) for value in self.stride),
        ).view(scores.shape[0], count, self.nc)

        scores_by_anchor = scores.transpose(1, 2)
        selected_scores = scores_by_anchor.gather(
            1, indices[..., None].expand(-1, -1, self.nc)
        )
        refined_scores = selected_scores + float(self.config.inference_weight) * residual
        scores_by_anchor = scores_by_anchor.scatter(
            1,
            indices[..., None].expand(-1, -1, self.nc),
            refined_scores,
        )
        result = dict(predictions)
        result["scores"] = scores_by_anchor.transpose(1, 2)
        result["coffee_fg_indices"] = indices
        return result

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


def inject_coffee_fg(model: nn.Module, config: CoffeeFGConfig | dict[str, Any]) -> int:
    """Replace the final Detect head with a CoffeeFG wrapper exactly once."""

    config = config if isinstance(config, CoffeeFGConfig) else CoffeeFGConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    current = detector[-1]
    if isinstance(current, CoffeeFGDetectHead):
        return 0
    detector[-1] = CoffeeFGDetectHead(current, config)
    return 1


def config_dict(head: CoffeeFGDetectHead) -> dict[str, Any]:
    return asdict(head.config)


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover - runtime dependency is pinned by pyproject
    DetectionModel = nn.Module  # type: ignore[assignment,misc]


class CoffeeFGDetectionModel(DetectionModel):
    """Ultralytics DetectionModel with a serializable CoffeeFG head."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        coffee_fg: CoffeeFGConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.coffee_fg_config = CoffeeFGConfig.from_mapping(coffee_fg)
        inject_coffee_fg(self, self.coffee_fg_config)

    def init_criterion(self):
        from .loss import CoffeeFGLoss

        return CoffeeFGLoss(self)
