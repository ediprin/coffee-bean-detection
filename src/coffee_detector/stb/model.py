from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torchvision.models.swin_transformer import SwinTransformerBlock


@dataclass(frozen=True)
class STBConfig:
    window_size: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0

    @classmethod
    def from_mapping(cls, payload: "STBConfig | dict[str, Any] | None") -> "STBConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.window_size <= 0 or result.num_heads <= 0 or result.mlp_ratio <= 0:
            raise ValueError("parameter STB harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class ClassificationSTB(nn.Module):
    """Zhao et al. STB: W-MSA block followed by shifted W-MSA block.

    Torchvision's SwinTransformerBlock implements LN->window attention->residual
    and LN->MLP->residual. We stack one unshifted and one half-window-shifted
    block exactly in the Eq.(4) order. A learnable residual gate is initialized
    at zero only to preserve an exact native-YOLO starting function; the paper
    itself replaces the classification convolutions rather than using this gate.
    """

    def __init__(self, channels: int, config: STBConfig) -> None:
        super().__init__()
        if channels % config.num_heads:
            raise ValueError(f"channels={channels} tidak habis dibagi heads={config.num_heads}")
        window = [config.window_size, config.window_size]
        shift = config.window_size // 2
        self.wmsa = SwinTransformerBlock(
            dim=channels,
            num_heads=config.num_heads,
            window_size=window,
            shift_size=[0, 0],
            mlp_ratio=config.mlp_ratio,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=0.0,
        )
        self.swmsa = SwinTransformerBlock(
            dim=channels,
            num_heads=config.num_heads,
            window_size=window,
            shift_size=[shift, shift],
            mlp_ratio=config.mlp_ratio,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=0.0,
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        nhwc = value.permute(0, 2, 3, 1).contiguous()
        transformed = self.swmsa(self.wmsa(nhwc)).permute(0, 3, 1, 2).contiguous()
        return value + self.gate * (transformed - value)


class STBDetectHead(nn.Module):
    """STB only on classification path; native YOLO26 localization stays untouched."""

    def __init__(self, base_head: nn.Module, config: STBConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("STB transfer memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("STB transfer memerlukan P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.blocks = nn.ModuleList([ClassificationSTB(channel, config) for channel in channels])
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

    def _forward_branch(self, features: list[torch.Tensor], branch: dict[str, nn.Module]):
        enhanced = [block(feature) for block, feature in zip(self.blocks, features)]
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](enhanced[index]))
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat([x.view(batch, 4 * self.reg_max, -1) for x in boxes], dim=-1),
            "scores": torch.cat([x.view(batch, self.nc, -1) for x in logits], dim=-1),
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
                "one2one": self._forward_branch([value.detach() for value in features], self.one2one),
            }
        one2many = self._forward_branch(features, self.one2many) if self._has_heads(self.one2many) else None
        one2one = self._forward_branch([value.detach() for value in features], self.one2one)
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_stb_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, STBDetectHead):
        raise TypeError("Target bukan STBDetectHead")
    if isinstance(source_head, STBDetectHead):
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
    DetectionModel = nn.Module  # type: ignore


class STBDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, stb=None):
        self.stb_config = STBConfig.from_mapping(stb)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = STBDetectHead(self.model[-1], self.stb_config)
