from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import math
import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.sfr_spatial.model import SFRSpatialConfig, WindowSpatialFormer


@dataclass(frozen=True)
class SFRSCConfig:
    """Frozen one-stage transfer settings for SFRNet SC-Former."""

    hidden_dim: int = 64
    num_heads: int = 4
    window_size: int = 7
    mlp_ratio: float = 2.0
    hash_buckets: int = 4
    hash_seed: int = 2023
    correction_scale: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "SFRSCConfig | dict[str, Any] | None") -> "SFRSCConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.hidden_dim <= 0 or result.hidden_dim % result.num_heads:
            raise ValueError("hidden_dim harus positif dan habis dibagi num_heads")
        if result.window_size <= 1:
            raise ValueError("window_size harus >1")
        if result.mlp_ratio <= 0 or result.correction_scale <= 0:
            raise ValueError("mlp_ratio/correction_scale harus positif")
        if result.hash_buckets < 2 or result.hash_buckets % 2:
            raise ValueError("hash_buckets harus genap dan >=2 sesuai h(x)=[xR;-xR]")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def spatial_config(self) -> SFRSpatialConfig:
        return SFRSpatialConfig(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            correction_scale=self.correction_scale,
        )


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class WindowChannelLSHFormer(nn.Module):
    """Windowed transfer of SFRNet C-Former.

    SFRNet defines channel tokens from a C x (H*W) RoI representation, uses a
    shared linear projection for Q and K, hashes tokens with
    h(x)=argmax([xR; -xR]), and restricts attention to matching hash buckets.

    Original SFRNet operates on 7x7 RoI features. For dense YOLO fields this
    module first projects channels, then partitions the map into non-overlapping
    7x7 windows. Each projected channel becomes one token with 49 components.
    The windowing is therefore an explicit one-stage transfer hypothesis.
    """

    def __init__(self, in_channels: int, config: SFRSCConfig) -> None:
        super().__init__()
        self.config = config
        self.token_dim = config.window_size * config.window_size
        self.project = nn.Conv2d(in_channels, config.hidden_dim, 1, bias=False)
        self.norm1 = nn.LayerNorm(self.token_dim)
        # One shared layer implements W_C^Q = W_C^K rather than two tied copies.
        self.qk = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.value = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.norm2 = nn.LayerNorm(self.token_dim)
        hidden = max(self.token_dim, int(round(self.token_dim * config.mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(self.token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.token_dim),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.hash_seed))
        random_projection = torch.randn(
            self.token_dim,
            config.hash_buckets // 2,
            generator=generator,
            dtype=torch.float32,
        )
        self.register_buffer("hash_projection", random_projection, persistent=True)

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        s = self.config.window_size
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        # [B,C,H,W] -> [B*nwin,C,s*s], i.e. channels are sequence tokens.
        windows = (
            x.view(b, c, hp // s, s, wp // s, s)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, c, s * s)
        )
        return windows, (h, w, hp, wp)

    def _restore(
        self,
        windows: torch.Tensor,
        shape: tuple[int, int, int, int],
        batch: int,
    ) -> torch.Tensor:
        h, w, hp, wp = shape
        s = self.config.window_size
        c = windows.shape[1]
        x = (
            windows.view(batch, hp // s, wp // s, c, s, s)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, c, hp, wp)
        )
        return x[:, :, :h, :w]

    def hash_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return SFRNet-style LSH bucket IDs in [0, B-1]."""
        if tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Dimensi token harus {self.token_dim}")
        projection = self.hash_projection.to(device=tokens.device, dtype=tokens.dtype)
        signed = tokens @ projection
        signed = torch.cat((signed, -signed), dim=-1)
        return signed.argmax(dim=-1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        x = self.project(feature)
        batch = x.shape[0]
        windows, shape = self._partition(x)
        normalized = self.norm1(windows)
        # Shared Q/K linear layer as required by the paper.
        q = self.qk(normalized)
        k = q
        v = self.value(normalized)
        q_bucket = self.hash_tokens(q)
        k_bucket = self.hash_tokens(k)
        allowed = q_bucket.unsqueeze(-1).eq(k_bucket.unsqueeze(-2))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.token_dim))
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attention, v)
        windows = windows + attended
        windows = windows + self.mlp(self.norm2(windows))
        return self._restore(windows, shape, batch)


class SFRSCCorrection(nn.Module):
    """Parallel S-Former + C-Former classification correction.

    The paper establishes parallel spatial/channel refinement. The dense YOLO26
    transfer uses a frozen arithmetic mean to combine the two refined fields;
    this mean is an adaptation boundary, not claimed as a literal paper fusion
    equation. A zero-initialized 1x1 classifier preserves the native detector at
    initialization.
    """

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
        config: SFRSCConfig,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SC-Former transfer memerlukan P3/P4/P5")
        self.config = config
        spatial_config = config.spatial_config()
        self.spatial_blocks = nn.ModuleList(
            [WindowSpatialFormer(channel, spatial_config) for channel in channels]
        )
        self.channel_blocks = nn.ModuleList(
            [WindowChannelLSHFormer(channel, config) for channel in channels]
        )
        self.classifiers = nn.ModuleList(
            [nn.Conv2d(config.hidden_dim, num_classes, 1) for _ in channels]
        )
        for layer in self.classifiers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        corrections: list[torch.Tensor] = []
        for feature, spatial, channel, classifier in zip(
            features, self.spatial_blocks, self.channel_blocks, self.classifiers
        ):
            spatial_feature = spatial(feature)
            channel_feature = channel(feature)
            fused = 0.5 * (spatial_feature + channel_feature)
            corrections.append(classifier(fused))
        return corrections


class SFRSCDetectHead(nn.Module):
    def __init__(self, base_head: nn.Module, config: SFRSCConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("SC-Former transfer memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.base_head = base_head
        self.config = config
        self.sc = SFRSCCorrection(channels, int(base_head.nc), config)
        for name in (
            "i", "f", "type", "np", "nc", "nl", "reg_max", "stride", "end2end",
            "max_det", "export", "format", "dynamic", "agnostic_nms",
        ):
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

    def _forward_branch(self, features, branch):
        corrections = self.sc(features)
        boxes, scores = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            native = branch["cls_head"][index](features[index])
            scores.append(
                native + float(self.config.correction_scale) * corrections[index]
            )
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat(
                [value.view(batch, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [value.view(batch, self.nc, -1) for value in scores], dim=-1
            ),
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


def load_sfr_sc_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load native YOLO26 D0 weights while keeping SC correction zero-initialized."""
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, SFRSCDetectHead):
        raise TypeError("Target bukan SFRSCDetectHead")
    if isinstance(source_head, SFRSCDetectHead):
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


class SFRSCDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, sfr_sc=None):
        self.sfr_sc_config = SFRSCConfig.from_mapping(sfr_sc)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = SFRSCDetectHead(self.model[-1], self.sfr_sc_config)
