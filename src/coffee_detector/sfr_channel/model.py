from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.sfr_spatial.model import SFRSpatialConfig, WindowSpatialFormer


@dataclass(frozen=True)
class SFRChannelConfig:
    hidden_dim: int = 64
    spatial_heads: int = 4
    window_size: int = 7
    bucket_count: int = 4
    mlp_ratio: float = 2.0
    correction_scale: float = 1.0
    hash_seed: int = 2023
    mode: str = "channel"  # channel | spatial_channel

    @classmethod
    def from_mapping(cls, payload: "SFRChannelConfig | dict[str, Any] | None") -> "SFRChannelConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.hidden_dim <= 0 or result.hidden_dim % result.bucket_count:
            raise ValueError("hidden_dim harus positif dan habis dibagi bucket_count")
        if result.hidden_dim % result.spatial_heads:
            raise ValueError("hidden_dim harus habis dibagi spatial_heads")
        if result.window_size <= 1 or result.bucket_count <= 1:
            raise ValueError("window_size/bucket_count tidak valid")
        if result.mlp_ratio <= 0 or result.correction_scale <= 0:
            raise ValueError("mlp_ratio/correction_scale harus positif")
        if result.mode not in {"channel", "spatial_channel"}:
            raise ValueError("mode harus channel atau spatial_channel")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class LSHChannelFormer(nn.Module):
    """SFRNet C-Former transfer on local 7x7 dense-field windows.

    Paper-faithful elements retained: channel tokens are spatially flattened,
    Q and K share one linear projection, random-projection LSH groups similar
    channels into B=4 buckets, and self-attention is restricted to each bucket.
    The original paper operates on 7x7 RoI tensors with C=256. Here dense YOLO
    feature maps are partitioned into 7x7 windows and projected to a lightweight
    hidden dimension, so this is explicitly a one-stage transfer hypothesis.
    """

    def __init__(self, in_channels: int, config: SFRChannelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        token_dim = config.window_size * config.window_size
        self.project = nn.Conv2d(in_channels, d, 1, bias=False)
        self.norm1 = nn.LayerNorm(token_dim)
        self.shared_qk = nn.Linear(token_dim, token_dim, bias=False)
        self.value = nn.Linear(token_dim, token_dim, bias=False)
        self.out = nn.Linear(token_dim, token_dim, bias=False)
        self.norm2 = nn.LayerNorm(token_dim)
        hidden = max(token_dim, int(round(token_dim * config.mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.GELU(), nn.Linear(hidden, token_dim)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.hash_seed))
        projection = torch.randn(token_dim, generator=generator)
        projection = projection / projection.norm().clamp_min(1e-12)
        self.register_buffer("hash_projection", projection, persistent=True)

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

    def bucket_order(self, channel_tokens: torch.Tensor) -> torch.Tensor:
        """Return deterministic LSH order; adjacent equal chunks form buckets."""
        qk = self.shared_qk(self.norm1(channel_tokens))
        projection = self.hash_projection.to(device=qk.device, dtype=qk.dtype)
        hash_values = torch.matmul(qk, projection)
        return torch.argsort(hash_values, dim=1, stable=True)

    @staticmethod
    def _gather_tokens(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.project(feature)
        batch = x.shape[0]
        windows, shape = self._partition(x)  # [NW, 49, C]
        channels = windows.transpose(1, 2).contiguous()  # [NW, C, 49]
        normed = self.norm1(channels)
        qk = self.shared_qk(normed)
        value = self.value(normed)
        projection = self.hash_projection.to(device=qk.device, dtype=qk.dtype)
        order = torch.argsort(torch.matmul(qk, projection), dim=1, stable=True)
        sorted_tokens = self._gather_tokens(channels, order)
        sorted_qk = self._gather_tokens(qk, order)
        sorted_v = self._gather_tokens(value, order)

        n, c, token_dim = sorted_qk.shape
        bcount = self.config.bucket_count
        per_bucket = c // bcount
        q = sorted_qk.view(n, bcount, per_bucket, token_dim)
        v = sorted_v.view(n, bcount, per_bucket, token_dim)
        logits = torch.matmul(q, q.transpose(-1, -2)) / math.sqrt(float(token_dim))
        attended = torch.softmax(logits, dim=-1) @ v
        attended = self.out(attended.reshape(n, c, token_dim))
        sorted_tokens = sorted_tokens + attended
        sorted_tokens = sorted_tokens + self.mlp(self.norm2(sorted_tokens))

        inverse = torch.argsort(order, dim=1, stable=True)
        channels = self._gather_tokens(sorted_tokens, inverse)
        windows = channels.transpose(1, 2).contiguous()
        return self._restore(windows, shape, batch)


class SFRChannelCorrection(nn.Module):
    def __init__(self, channels: tuple[int, int, int], num_classes: int, config: SFRChannelConfig) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SFR channel transfer memerlukan P3/P4/P5")
        self.config = config
        self.channel_blocks = nn.ModuleList([LSHChannelFormer(c, config) for c in channels])
        if config.mode == "spatial_channel":
            spatial_cfg = SFRSpatialConfig(
                hidden_dim=config.hidden_dim,
                num_heads=config.spatial_heads,
                window_size=config.window_size,
                mlp_ratio=config.mlp_ratio,
                correction_scale=config.correction_scale,
            )
            self.spatial_blocks = nn.ModuleList([WindowSpatialFormer(c, spatial_cfg) for c in channels])
        else:
            self.spatial_blocks = None
        self.classifiers = nn.ModuleList(
            [nn.Conv2d(config.hidden_dim, num_classes, 1) for _ in channels]
        )
        for layer in self.classifiers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        output = []
        for index, (block, classifier, feature) in enumerate(
            zip(self.channel_blocks, self.classifiers, features)
        ):
            refined = block(feature)
            if self.spatial_blocks is not None:
                refined = refined + self.spatial_blocks[index](feature)
            output.append(classifier(refined))
        return output


class SFRChannelDetectHead(nn.Module):
    def __init__(self, base_head: nn.Module, config: SFRChannelConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("SFR channel transfer memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.base_head = base_head
        self.config = config
        self.correction = SFRChannelCorrection(channels, int(base_head.nc), config)
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
        corrections = self.correction(features)
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
        one2one = self._forward_branch([x.detach() for x in features], self.one2one)
        predictions = {"one2one": one2one}
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_sfr_channel_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, SFRChannelDetectHead):
        raise TypeError("Target bukan SFRChannelDetectHead")
    if isinstance(source_head, SFRChannelDetectHead):
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


class SFRChannelDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, sfr_channel=None):
        self.sfr_channel_config = SFRChannelConfig.from_mapping(sfr_channel)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = SFRChannelDetectHead(self.model[-1], self.sfr_channel_config)
