from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .hierarchy import build_sni21_entity_family_hierarchy


@dataclass(frozen=True)
class BHCLConfig:
    embedding_dim: int = 128
    temperature: float = 0.1
    loss_weight: float = 0.6
    epsilon: float = 0.1
    anchor_chunk_size: int = 256

    @classmethod
    def from_mapping(cls, payload: "BHCLConfig | dict[str, Any] | None") -> "BHCLConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.embedding_dim <= 0:
            raise ValueError("embedding_dim harus positif")
        if result.temperature <= 0 or result.loss_weight < 0:
            raise ValueError("temperature harus positif dan loss_weight tidak negatif")
        if not 0.0 < result.epsilon <= 1.0:
            raise ValueError("epsilon harus di (0,1]")
        if result.anchor_chunk_size <= 0:
            raise ValueError("anchor_chunk_size harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class BHCLProjectionHead(nn.Module):
    """APCL-capacity-matched P3/P4/P5 projection for BHCL."""

    def __init__(self, channels: tuple[int, int, int], config: BHCLConfig) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("BHCL memerlukan tepat P3/P4/P5")
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

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError("BHCL memerlukan tepat P3/P4/P5")
        rows = []
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            batch = value.shape[0]
            value = value.view(batch, self.config.embedding_dim, -1).transpose(1, 2)
            rows.append(value)
        return torch.cat(rows, dim=1)


class BHCLDetectHead(nn.Module):
    """Native YOLO26 Detect with training-only BHCL embeddings and EMA state."""

    def __init__(self, base_head: nn.Module, config: BHCLConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("BHCL dikunci untuk native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("BHCL memerlukan tiga level P3/P4/P5")
        hierarchy = build_sni21_entity_family_hierarchy()
        if int(base_head.nc) != hierarchy.leaf_count:
            raise ValueError(
                f"BHCL SNI21 memerlukan {hierarchy.leaf_count} kelas, diterima {base_head.nc}"
            )
        self.base_head = base_head
        self.config = config
        self.bhcl_projection = BHCLProjectionHead(channels, config)
        # Import here avoids a model<->state runtime import cycle while keeping
        # prototype buffers inside the checkpointed model state.
        from .state import BalancedHierarchyPrototypeBank
        self.bhcl_prototypes = BalancedHierarchyPrototypeBank(config, hierarchy)
        for name in ("i", "f", "type", "np", "nc", "nl", "reg_max", "stride", "end2end", "max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))

    @property
    def one2many(self):
        return self.base_head.one2many

    @property
    def one2one(self):
        return self.base_head.one2one

    def _sync(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    def _forward_branch(self, features, branch, *, include_bhcl: bool):
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](features[index]))
        batch = features[0].shape[0]
        output = {
            "boxes": torch.cat([v.view(batch, 4 * self.reg_max, -1) for v in boxes], dim=-1),
            "scores": torch.cat([v.view(batch, self.nc, -1) for v in logits], dim=-1),
            "feats": features,
        }
        if include_bhcl:
            output["bhcl_embeddings"] = self.bhcl_projection(features)
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many, include_bhcl=True),
                "one2one": self._forward_branch(
                    [value.detach() for value in features], self.one2one, include_bhcl=False
                ),
            }
        return self.base_head(features)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_bhcl_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, BHCLDetectHead):
        raise TypeError("Target bukan BHCLDetectHead")
    if isinstance(source_head, BHCLDetectHead):
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class BHCLDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, bhcl=None):
        self.bhcl_config = BHCLConfig.from_mapping(bhcl)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = BHCLDetectHead(self.model[-1], self.bhcl_config)

    def init_criterion(self):
        from .loss import BHCLDetectionLoss
        return BHCLDetectionLoss(self)
