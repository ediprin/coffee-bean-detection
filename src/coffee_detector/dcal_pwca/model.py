from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DCALPWCAConfig:
    mode: str = "pwca"  # sa | pwca
    hidden_dim: int = 64
    num_heads: int = 4
    mlp_ratio: float = 2.0
    correction_scale: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "DCALPWCAConfig | dict[str, Any] | None") -> "DCALPWCAConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.mode not in {"sa", "pwca"}:
            raise ValueError("mode DCAL-PWCA harus sa atau pwca")
        if result.hidden_dim <= 0 or result.hidden_dim % result.num_heads:
            raise ValueError("hidden_dim harus positif dan habis dibagi num_heads")
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


class P5CrossAttentionRegularizer(nn.Module):
    """Training-path transfer of DCAL PWCA Eq. (4) on YOLO26 P5.

    Paper mechanism retained:
      * target-image queries;
      * self key/value for SA control;
      * concatenated target + paired-image key/value for PWCA;
      * same classification target as the native branch;
      * PWCA removed at inference.

    Transfer choices:
      * P5 only, one attention block;
      * hidden_dim=64 and 4 heads by default;
      * random cyclic in-batch pairing;
      * zero-initialized residual leaf-logit correction.
    """

    def __init__(self, in_channels: int, num_classes: int, config: DCALPWCAConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.project = nn.Conv2d(int(in_channels), d, 1, bias=False)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, config.num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d)
        hidden = max(d, int(round(d * config.mlp_ratio)))
        self.ffn = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.classifier = nn.Conv2d(d, int(num_classes), 1, bias=True)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.last_query_tokens: int | None = None
        self.last_kv_tokens: int | None = None
        self.last_pair_offset: int | None = None

    def _pair_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch = tokens.shape[0]
        if batch <= 1:
            return tokens, 0
        # Random cyclic pairing is a transfer choice. It guarantees every image
        # receives another image and remains deterministic under the frozen torch seed.
        offset = int(torch.randint(1, batch, (1,), device=tokens.device).item())
        return torch.roll(tokens, shifts=offset, dims=0), offset

    def forward_features(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.project(feature)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B,N,D]
        query = self.norm_q(tokens)
        if self.config.mode == "pwca":
            paired, offset = self._pair_tokens(tokens)
            kv_tokens = torch.cat((tokens, paired), dim=1) if offset else tokens
            self.last_pair_offset = offset
        else:
            kv_tokens = tokens
            self.last_pair_offset = 0
        kv = self.norm_kv(kv_tokens)
        attended, _ = self.attention(query, kv, kv, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        self.last_query_tokens = int(query.shape[1])
        self.last_kv_tokens = int(kv.shape[1])
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(feature))


class DCALPWCADetectHead(nn.Module):
    """Native YOLO26 Detect with training-only P5 SA/PWCA classification correction."""

    def __init__(self, base_head: nn.Module, config: DCALPWCAConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("DCAL-PWCA transfer memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("DCAL-PWCA transfer dikunci untuk P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.regularizer = P5CrossAttentionRegularizer(channels[2], int(base_head.nc), config)
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

    def _native_branch(self, features: list[torch.Tensor], branch: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
        boxes, scores = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            scores.append(branch["cls_head"][index](features[index]))
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat([value.view(batch, 4 * self.reg_max, -1) for value in boxes], dim=-1),
            "scores": torch.cat([value.view(batch, self.nc, -1) for value in scores], dim=-1),
            "feats": features,
        }

    def _one2many_with_regularizer(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        boxes, scores = [], []
        correction = float(self.config.correction_scale) * self.regularizer(features[2])
        for index in range(self.nl):
            boxes.append(self.one2many["box_head"][index](features[index]))
            native = self.one2many["cls_head"][index](features[index])
            scores.append(native + correction if index == 2 else native)
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat([value.view(batch, 4 * self.reg_max, -1) for value in boxes], dim=-1),
            "scores": torch.cat([value.view(batch, self.nc, -1) for value in scores], dim=-1),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        # PWCA is explicitly absent from inference. Delegating to base_head makes
        # deployment path exactly native YOLO26 rather than merely numerically close.
        if not self.training:
            return self.base_head(features)
        one2many = self._one2many_with_regularizer(features)
        one2one = self._native_branch([value.detach() for value in features], self.one2one)
        return {"one2many": one2many, "one2one": one2one}

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_dcal_pwca(model: nn.Module, config: DCALPWCAConfig | dict[str, Any] | None) -> int:
    frozen = DCALPWCAConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], DCALPWCADetectHead):
        return 0
    detector[-1] = DCALPWCADetectHead(detector[-1], frozen)
    return 1


def load_dcal_pwca_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly transfer native D0 state into the DCAL-PWCA wrapper."""
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, DCALPWCADetectHead):
        raise TypeError("Target bukan DCALPWCADetectHead")
    if isinstance(source_head, DCALPWCADetectHead):
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect ke DCAL-PWCA tidak lengkap")
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


class DCALPWCADetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, dcal_pwca=None):
        self.dcal_pwca_config = DCALPWCAConfig.from_mapping(dcal_pwca)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        inject_dcal_pwca(self, self.dcal_pwca_config)
