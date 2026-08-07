from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SAFPNAlignmentConfig:
    """Configuration for a SAFM-inspired classification-only correction path.

    The alignment operator follows Li et al. (TGRS 2025), Eqs. (3)-(8):
    adjacent features are channel-aligned, the deep feature is upsampled,
    two 2-D offset maps and one spatial weight map are predicted, both inputs
    are bilinearly warped, and weighted aligned features are added together
    with the original unwarped features as priors.

    This repository adaptation deliberately leaves YOLO26 box regression
    untouched and injects the aligned representation only as a residual class
    correction. The final class-correction convolutions are zero initialized,
    so a freshly injected model is exactly D0 before learning.
    """

    correction_scale: float = 1.0
    offset_init_zero: bool = True
    sampling_ratio_note: str = "grid_sample_bilinear"

    @classmethod
    def from_mapping(
        cls, payload: "SAFPNAlignmentConfig | dict[str, Any] | None"
    ) -> "SAFPNAlignmentConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.correction_scale <= 0:
            raise ValueError("correction_scale harus positif")
        if result.sampling_ratio_note != "grid_sample_bilinear":
            raise ValueError("sampling_ratio_note dikunci ke grid_sample_bilinear")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


def _pixel_base_grid(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((yy, xx), dim=0).unsqueeze(0)


def _bilinear_align(feature: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Eq. (5) implemented with differentiable bilinear grid sampling.

    Offset channel 0 is the vertical displacement (delta_h) and channel 1 is
    the horizontal displacement (delta_w), both expressed in feature pixels.
    """

    if feature.ndim != 4 or offset.ndim != 4 or offset.shape[1] != 2:
        raise ValueError("feature harus [B,C,H,W] dan offset harus [B,2,H,W]")
    batch, _, height, width = feature.shape
    if tuple(offset.shape[-2:]) != (height, width):
        raise ValueError("Resolusi offset harus sama dengan feature yang di-warp")
    base = _pixel_base_grid(height, width, device=feature.device, dtype=feature.dtype)
    coords = base + offset.to(dtype=feature.dtype)
    y = coords[:, 0]
    x = coords[:, 1]
    if height > 1:
        y = 2.0 * y / float(height - 1) - 1.0
    else:
        y = torch.zeros_like(y)
    if width > 1:
        x = 2.0 * x / float(width - 1) - 1.0
    else:
        x = torch.zeros_like(x)
    grid = torch.stack((x, y), dim=-1)
    if grid.shape[0] == 1 and batch > 1:
        grid = grid.expand(batch, -1, -1, -1)
    return F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class SpatialAwareAlignmentFusion(nn.Module):
    """SAFM operator for one adjacent deep/shallow pair.

    The shallow resolution/channel count defines the output. The deep feature
    is first projected to that channel count and bilinearly upsampled. This
    lateral projection is required in YOLO26 because P3/P4/P5 have unequal
    channels, whereas the paper formulates SAFM after FPN channel alignment.
    """

    def __init__(
        self,
        shallow_channels: int,
        deep_channels: int,
        *,
        offset_init_zero: bool = True,
    ) -> None:
        super().__init__()
        self.deep_projection = (
            nn.Identity()
            if int(deep_channels) == int(shallow_channels)
            else nn.Conv2d(int(deep_channels), int(shallow_channels), 1, bias=False)
        )
        joined_channels = 2 * int(shallow_channels)
        self.shallow_offset = nn.Conv2d(joined_channels, 2, 1)
        self.deep_offset = nn.Conv2d(joined_channels, 2, 1)
        self.spatial_weight = nn.Conv2d(2, 1, 1)
        if offset_init_zero:
            nn.init.zeros_(self.shallow_offset.weight)
            nn.init.zeros_(self.shallow_offset.bias)
            nn.init.zeros_(self.deep_offset.weight)
            nn.init.zeros_(self.deep_offset.bias)

    def forward(
        self, shallow: torch.Tensor, deep: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        deep = self.deep_projection(deep)
        deep_up = F.interpolate(
            deep,
            size=shallow.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        joined = torch.cat((deep_up, shallow), dim=1)
        shallow_offset = self.shallow_offset(joined)
        deep_offset = self.deep_offset(joined)

        pooled = torch.cat(
            (joined.mean(dim=1, keepdim=True), joined.amax(dim=1, keepdim=True)),
            dim=1,
        )
        weight = torch.sigmoid(self.spatial_weight(pooled))
        aligned_shallow = _bilinear_align(shallow, shallow_offset)
        aligned_deep = _bilinear_align(deep_up, deep_offset)
        fused = weight * (aligned_deep + aligned_shallow) + deep_up + shallow
        diagnostics = {
            "weight": weight,
            "shallow_offset": shallow_offset,
            "deep_offset": deep_offset,
        }
        return fused, diagnostics


class SAFPNClassificationCorrection(nn.Module):
    """Top-down P5->P4->P3 SAFM representation for class-score correction."""

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
        config: SAFPNAlignmentConfig,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SAFPN alignment memerlukan tepat P3/P4/P5")
        c3, c4, c5 = (int(value) for value in channels)
        self.config = config
        self.p5_to_p4 = SpatialAwareAlignmentFusion(
            c4, c5, offset_init_zero=config.offset_init_zero
        )
        self.p4_to_p3 = SpatialAwareAlignmentFusion(
            c3, c4, offset_init_zero=config.offset_init_zero
        )
        self.class_corrections = nn.ModuleList(
            [nn.Conv2d(c3, num_classes, 1), nn.Conv2d(c4, num_classes, 1)]
        )
        for layer in self.class_corrections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        if len(features) != 3:
            raise ValueError("SAFPN alignment memerlukan tepat P3/P4/P5")
        p3, p4, p5 = features
        aligned_p4, diag45 = self.p5_to_p4(p4, p5)
        aligned_p3, diag34 = self.p4_to_p3(p3, aligned_p4)
        scale = float(self.config.correction_scale)
        corrections = [
            scale * self.class_corrections[0](aligned_p3),
            scale * self.class_corrections[1](aligned_p4),
            p5.new_zeros((p5.shape[0], self.class_corrections[1].out_channels, *p5.shape[-2:])),
        ]
        diagnostics = {
            "p4_weight": diag45["weight"],
            "p4_shallow_offset": diag45["shallow_offset"],
            "p4_deep_offset": diag45["deep_offset"],
            "p3_weight": diag34["weight"],
            "p3_shallow_offset": diag34["shallow_offset"],
            "p3_deep_offset": diag34["deep_offset"],
        }
        return corrections, diagnostics


class SAFPNAlignmentDetectHead(nn.Module):
    """YOLO26 Detect wrapper: native boxes, SAFM-derived class correction only."""

    def __init__(self, base_head: nn.Module, config: SAFPNAlignmentConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("SAFPN alignment memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("SAFPN alignment dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("SAFPN alignment memerlukan tiga level P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.alignment = SAFPNClassificationCorrection(channels, int(base_head.nc), config)
        self.last_alignment_diagnostics: dict[str, torch.Tensor] = {}
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

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def _forward_branch(
        self, features: list[torch.Tensor], branch: dict[str, nn.Module]
    ) -> dict[str, torch.Tensor]:
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](features[index]))
        corrections, diagnostics = self.alignment(features)
        self.last_alignment_diagnostics = diagnostics
        batch_size = features[0].shape[0]
        return {
            "boxes": torch.cat(
                [value.view(batch_size, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [
                    (logit + correction).view(batch_size, self.nc, -1)
                    for logit, correction in zip(logits, corrections)
                ],
                dim=-1,
            ),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            one2many = self._forward_branch(features, self.one2many)
            one2one = self._forward_branch(
                [value.detach() for value in features], self.one2one
            )
            return {"one2many": one2many, "one2one": one2one}
        one2many = (
            self._forward_branch(features, self.one2many)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [value.detach() for value in features], self.one2one
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_safpn_alignment(
    model: nn.Module, config: SAFPNAlignmentConfig | dict[str, Any] | None
) -> int:
    frozen = SAFPNAlignmentConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], SAFPNAlignmentDetectHead):
        return 0
    detector[-1] = SAFPNAlignmentDetectHead(detector[-1], frozen)
    return 1


def load_safpn_alignment_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly map a native D0 Detect state into the SAFPN wrapper namespace."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, SAFPNAlignmentDetectHead):
        raise TypeError("Target bukan SAFPNAlignmentDetectHead")
    if isinstance(source_head, SAFPNAlignmentDetectHead):
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


class SAFPNAlignmentDetectionModel(DetectionModel):
    """YOLO26 model with classification-only SAFM residual alignment."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        safpn_alignment: SAFPNAlignmentConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.safpn_alignment_config = SAFPNAlignmentConfig.from_mapping(safpn_alignment)
        inject_safpn_alignment(self, self.safpn_alignment_config)
