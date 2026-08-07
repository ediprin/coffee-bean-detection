from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class FSCECPEConfig:
    embedding_dim: int = 128
    temperature: float = 0.2
    iou_threshold: float = 0.7
    loss_weight: float = 0.5

    @classmethod
    def from_mapping(cls, payload: "FSCECPEConfig | dict[str, Any] | None") -> "FSCECPEConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.embedding_dim <= 0:
            raise ValueError("embedding_dim harus positif")
        if result.temperature <= 0:
            raise ValueError("temperature harus positif")
        if not 0.0 <= result.iou_threshold < 1.0:
            raise ValueError("iou_threshold harus berada pada [0,1)")
        if result.loss_weight < 0:
            raise ValueError("loss_weight tidak boleh negatif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class CPEProjectionHead(nn.Module):
    """Training-only dense transfer of FSCE's one-layer contrastive MLP.

    FSCE projects each RoI feature through a one-layer MLP into D_C=128.
    YOLO26 does not have pooled RoI vectors, so each P3/P4/P5 spatial token is
    projected by an independent 1x1 convolution, which is the dense linear
    analogue of that one-layer MLP. No activation or prototype state is added.
    """

    def __init__(self, channels: tuple[int, int, int], config: FSCECPEConfig) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("CPE memerlukan tepat P3/P4/P5")
        self.config = config
        self.projections = nn.ModuleList(
            [nn.Conv2d(int(channel), config.embedding_dim, 1, bias=True) for channel in channels]
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError("CPE memerlukan tepat P3/P4/P5")
        embeddings = []
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            batch = value.shape[0]
            value = value.view(batch, self.config.embedding_dim, -1).transpose(1, 2)
            embeddings.append(value)
        return torch.cat(embeddings, dim=1)


class FSCECPEDetectHead(nn.Module):
    """Native YOLO26 Detect plus training-only CPE embeddings.

    Native box and classification outputs remain untouched. The projection is
    executed only for the one-to-many training branch and is completely skipped
    at evaluation/inference, so the candidate has zero inference overhead.
    """

    def __init__(self, base_head: nn.Module, config: FSCECPEConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("CPE memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("CPE dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("CPE memerlukan tiga level P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.cpe_projection = CPEProjectionHead(channels, config)
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
        *,
        include_cpe: bool,
    ) -> dict[str, torch.Tensor]:
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](features[index]))
        batch_size = features[0].shape[0]
        output = {
            "boxes": torch.cat(
                [value.view(batch_size, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [value.view(batch_size, self.nc, -1) for value in logits], dim=-1
            ),
            "feats": features,
        }
        if include_cpe:
            output["cpe_embeddings"] = self.cpe_projection(features)
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            one2many = self._forward_branch(features, self.one2many, include_cpe=True)
            one2one = self._forward_branch(
                [value.detach() for value in features], self.one2one, include_cpe=False
            )
            return {"one2many": one2many, "one2one": one2one}
        return self.base_head(features)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_fsce_cpe(model: nn.Module, config: FSCECPEConfig | dict[str, Any] | None) -> int:
    frozen = FSCECPEConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], FSCECPEDetectHead):
        return 0
    detector[-1] = FSCECPEDetectHead(detector[-1], frozen)
    return 1


def load_fsce_cpe_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly transfer the native YOLO26 Detect state into the CPE wrapper."""
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, FSCECPEDetectHead):
        raise TypeError("Target bukan FSCECPEDetectHead")
    if isinstance(source_head, FSCECPEDetectHead):
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect ke FSCE-CPE tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}
