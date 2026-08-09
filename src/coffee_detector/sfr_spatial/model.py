from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import math
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SFRSpatialConfig:
    hidden_dim: int = 64
    num_heads: int = 4
    window_size: int = 7
    mlp_ratio: float = 2.0
    correction_scale: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "SFRSpatialConfig | dict[str, Any] | None") -> "SFRSpatialConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.hidden_dim <= 0 or result.hidden_dim % result.num_heads:
            raise ValueError("hidden_dim harus positif dan habis dibagi num_heads")
        if result.window_size <= 1:
            raise ValueError("window_size harus >1")
        if result.mlp_ratio <= 0 or result.correction_scale <= 0:
            raise ValueError("mlp_ratio/correction_scale harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


def sinusoidal_position(length: int, dim: int, *, device, dtype) -> torch.Tensor:
    """Fixed sinusoidal PE as used by SFRNet's spatial transformer."""
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    half = torch.arange(0, dim, 2, device=device, dtype=dtype)
    div = torch.exp(-math.log(10000.0) * half / float(dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe


class WindowSpatialFormer(nn.Module):
    """One-stage transfer of SFRNet S-Former.

    Original SFRNet applies MSA to 7x7 RoI features. Dense YOLO P3/P4/P5 maps
    would make global attention quadratic and semantically unlike RoI attention.
    This transfer therefore uses non-overlapping local windows of the same 7x7
    token extent. It preserves the paper's core operator (sinusoidal PE + MSA +
    LN/MLP residuals) while explicitly being a one-stage transfer hypothesis.
    """

    def __init__(self, in_channels: int, config: SFRSpatialConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.project = nn.Conv2d(in_channels, d, 1, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, config.num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        hidden = max(d, int(round(d * config.mlp_ratio)))
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        s = self.config.window_size
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        windows = (
            x.view(b, c, hp // s, s, wp // s, s)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, s * s, c)
        )
        return windows, (h, w, hp, wp)

    def _restore(self, windows: torch.Tensor, shape: tuple[int, int, int, int], batch: int) -> torch.Tensor:
        h, w, hp, wp = shape
        s = self.config.window_size
        c = windows.shape[-1]
        x = (
            windows.view(batch, hp // s, wp // s, s, s, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, c, hp, wp)
        )
        return x[:, :, :h, :w]

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.project(feature)
        batch = x.shape[0]
        windows, shape = self._partition(x)
        pe = sinusoidal_position(
            windows.shape[1], windows.shape[2], device=windows.device, dtype=windows.dtype
        ).unsqueeze(0)
        q = self.norm1(windows + pe)
        attended, _ = self.attn(q, q, q, need_weights=False)
        windows = windows + attended
        windows = windows + self.mlp(self.norm2(windows))
        return self._restore(windows, shape, batch)


class SFRSpatialCorrection(nn.Module):
    def __init__(self, channels: tuple[int, int, int], num_classes: int, config: SFRSpatialConfig) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SFR spatial transfer memerlukan P3/P4/P5")
        self.blocks = nn.ModuleList([WindowSpatialFormer(c, config) for c in channels])
        self.classifiers = nn.ModuleList(
            [nn.Conv2d(config.hidden_dim, num_classes, 1) for _ in channels]
        )
        for layer in self.classifiers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [classifier(block(feature)) for block, classifier, feature in zip(self.blocks, self.classifiers, features)]


class SFRSpatialDetectHead(nn.Module):
    def __init__(self, base_head: nn.Module, config: SFRSpatialConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("SFR spatial transfer memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.base_head = base_head
        self.config = config
        self.spatial = SFRSpatialCorrection(channels, int(base_head.nc), config)
        for name in ("i", "f", "type", "np", "nc", "nl", "reg_max", "stride", "end2end", "max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))

    @property
    def one2many(self):
        return self.base_head.one2many

    @property
    def one2one(self):
        return self.base_head.one2one

    def _sync(self):
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    def _forward_branch(self, features, branch):
        corrections = self.spatial(features)
        boxes, scores = [], []
        for i in range(self.nl):
            boxes.append(branch["box_head"][i](features[i]))
            native = branch["cls_head"][i](features[i])
            scores.append(native + float(self.config.correction_scale) * corrections[i])
        b = features[0].shape[0]
        return {
            "boxes": torch.cat([v.view(b, 4 * self.reg_max, -1) for v in boxes], dim=-1),
            "scores": torch.cat([v.view(b, self.nc, -1) for v in scores], dim=-1),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        self._sync()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many),
                "one2one": self._forward_branch([x.detach() for x in features], self.one2one),
            }
        one2many = self._forward_branch(features, self.one2many)
        one2one = self._forward_branch([x.detach() for x in features], self.one2one)
        predictions = {"one2many": one2many, "one2one": one2one}
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_sfr_spatial_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, SFRSpatialDetectHead):
        raise TypeError("Target bukan SFRSpatialDetectHead")
    if isinstance(source_head, SFRSpatialDetectHead):
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


class SFRSpatialDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, sfr_spatial=None):
        self.sfr_spatial_config = SFRSpatialConfig.from_mapping(sfr_spatial)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = SFRSpatialDetectHead(self.model[-1], self.sfr_spatial_config)
