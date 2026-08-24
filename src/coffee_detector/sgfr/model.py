"""Spatial-Geometry-Frequency Frozen Residual synthesis for YOLO26."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer
from coffee_detector.igem.model import ClassAwareReferenceLevel, IGEMConfig
from coffee_detector.stb.model import ClassificationSTB, STBConfig, STBDetectHead


@dataclass(frozen=True)
class SGFRConfig:
    """Configuration shared by the optimizer control and residual stages."""

    stage: str = "geometry"  # control | geometry | frequency
    window_size: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    reference_depth: int = 3
    mask_loss_weight: float = 0.05
    kernel_size: int = 3
    attention_heads: int = 4
    channel_reduction: int = 4
    frequency_hidden: int = 32
    af2_patch_size: int = 32
    af2_overlap: float = 0.50
    af2_radius_ratio: float = 0.05
    af2_gamma: float = 0.10

    @classmethod
    def from_mapping(
        cls, payload: "SGFRConfig | Mapping[str, Any] | None"
    ) -> "SGFRConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.stage not in {"control", "geometry", "frequency"}:
            raise ValueError("stage SGFR harus control, geometry, atau frequency")
        if result.frequency_hidden <= 0:
            raise ValueError("frequency_hidden harus positif")
        result.stb_config()
        result.igem_config()
        result.af2_config()
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stb_config(self) -> STBConfig:
        return STBConfig(
            window_size=self.window_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        )

    def igem_config(self) -> IGEMConfig:
        return IGEMConfig(
            reference_depth=self.reference_depth,
            mask_loss_weight=self.mask_loss_weight,
            kernel_size=self.kernel_size,
            attention_heads=self.attention_heads,
            channel_reduction=self.channel_reduction,
            correction_scale=1.0,
        )

    def af2_config(self) -> AFABConfig:
        return AFABConfig(
            mode="af2",
            patch_size=self.af2_patch_size,
            overlap=self.af2_overlap,
            radius_ratio=self.af2_radius_ratio,
            gamma=self.af2_gamma,
        )


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class FrequencyResidualLevel(nn.Module):
    """Lightweight AF2 residual encoder with an exact zero-output start."""

    def __init__(self, num_classes: int, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.class_correction = nn.Conv2d(hidden, int(num_classes), 1)
        nn.init.zeros_(self.class_correction.weight)
        nn.init.zeros_(self.class_correction.bias)

    def forward(self, residual: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        resized = F.interpolate(residual, size=size, mode="bilinear", align_corners=False)
        return self.class_correction(self.encoder(resized))


class SGFRDetectHead(nn.Module):
    """Frozen STB detector with parallel geometry and frequency class residuals."""

    def __init__(self, base_head: nn.Module, config: SGFRConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("SGFR memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("SGFR memerlukan P3/P4/P5")
        self.base_head = base_head
        self.config = config
        # Keep the source STB key schema so a completed STB1 checkpoint can be
        # transferred and verified strictly rather than approximately.
        self.blocks = nn.ModuleList(
            [ClassificationSTB(channel, config.stb_config()) for channel in channels]
        )
        self.geometry_levels = nn.ModuleList(
            [
                ClassAwareReferenceLevel(channel, int(base_head.nc), config.igem_config())
                for channel in channels
            ]
        )
        self.frequency_levels = nn.ModuleList(
            [
                FrequencyResidualLevel(int(base_head.nc), int(config.frequency_hidden))
                for _ in channels
            ]
        )
        self.af2 = AFABInputEnhancer(config.af2_config())
        self._frequency_residual: torch.Tensor | None = None
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

    def prepare_frequency(self, image: torch.Tensor) -> None:
        self._frequency_residual = self.af2(image) - image

    def clear_frequency(self) -> None:
        self._frequency_residual = None

    def _sync_runtime_attributes(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def _forward_branch(
        self,
        features: list[torch.Tensor],
        branch: dict[str, nn.Module],
        *,
        expose_masks: bool,
    ) -> dict[str, Any]:
        enhanced = [block(feature) for block, feature in zip(self.blocks, features)]
        boxes, scores, masks = [], [], []
        for index in range(self.nl):
            # Localization is always the frozen STB/native YOLO26 path.
            boxes.append(branch["box_head"][index](features[index]))
            native_score = branch["cls_head"][index](enhanced[index])
            score = native_score
            if self.config.stage in {"geometry", "frequency"}:
                geometry, mask_logits = self.geometry_levels[index](features[index])
                score = score + geometry
                masks.append(mask_logits)
            if self.config.stage == "frequency":
                if self._frequency_residual is None:
                    raise RuntimeError("SGFR frequency stage tidak menerima input image")
                score = score + self.frequency_levels[index](
                    self._frequency_residual, native_score.shape[-2:]
                )
            scores.append(score)
        batch = features[0].shape[0]
        output: dict[str, Any] = {
            "boxes": torch.cat(
                [value.view(batch, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [value.view(batch, self.nc, -1) for value in scores], dim=-1
            ),
            "feats": features,
        }
        if expose_masks and masks:
            output["sgfr_mask_logits"] = masks
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.config.stage == "frequency" and self._frequency_residual is None:
            raise RuntimeError("SGFR frequency head dipanggil tanpa input image")
        if self.training:
            return {
                "one2many": self._forward_branch(
                    features,
                    self.one2many,
                    expose_masks=self.config.stage == "geometry",
                ),
                "one2one": self._forward_branch(
                    [value.detach() for value in features],
                    self.one2one,
                    expose_masks=False,
                ),
            }
        one2many = (
            self._forward_branch(features, self.one2many, expose_masks=False)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [value.detach() for value in features], self.one2one, expose_masks=False
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_sgfr_weights(model: nn.Module, weights: Any) -> dict[str, int | str]:
    """Strictly transfer STB1 or a complete SGFR checkpoint."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, SGFRDetectHead):
        raise TypeError("Target bukan SGFRDetectHead")
    if len(source_model) != len(target):
        raise RuntimeError("Jumlah layer source dan target SGFR berbeda")
    backbone_items = 0
    for index in range(len(target) - 1):
        result = target[index].load_state_dict(source_model[index].state_dict(), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Transfer backbone/neck layer {index} tidak strict")
        backbone_items += len(source_model[index].state_dict())
    if isinstance(source_head, SGFRDetectHead):
        result = target_head.load_state_dict(source_head.state_dict(), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Resume SGFR tidak strict")
        return {
            "source": "SGFR",
            "items": len(source_head.state_dict()),
            "backbone_items": backbone_items,
        }
    if not isinstance(source_head, STBDetectHead):
        raise TypeError(f"SGFR memerlukan checkpoint STB1, diterima {type(source_head).__name__}")
    base_result = target_head.base_head.load_state_dict(
        source_head.base_head.state_dict(), strict=True
    )
    block_result = target_head.blocks.load_state_dict(source_head.blocks.state_dict(), strict=True)
    if (
        base_result.missing_keys
        or base_result.unexpected_keys
        or block_result.missing_keys
        or block_result.unexpected_keys
    ):
        raise RuntimeError("Transfer head STB1 ke SGFR tidak strict")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {
        "source": "STB1",
        "items": len(source_head.state_dict()),
        "backbone_items": backbone_items,
    }


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore[assignment,misc]


class SGFRDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, sgfr=None) -> None:
        self.sgfr_config = SGFRConfig.from_mapping(sgfr)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = SGFRDetectHead(self.model[-1], self.sgfr_config)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        head = self.model[-1]
        if not isinstance(head, SGFRDetectHead):
            return super().predict(
                x, profile=profile, visualize=visualize, augment=augment, embed=embed
            )
        if head.config.stage == "frequency":
            head.prepare_frequency(x)
        try:
            return super().predict(
                x, profile=profile, visualize=visualize, augment=augment, embed=embed
            )
        finally:
            head.clear_frequency()

    def train(self, mode: bool = True):
        super().train(mode)
        head = self.model[-1] if len(self.model) else None
        if not mode or not isinstance(head, SGFRDetectHead):
            return self
        # Freezing parameters alone is insufficient: BatchNorm running buffers
        # would still change and could silently alter STB boxes.
        for layer in list(self.model)[:-1]:
            layer.eval()
        head.blocks.eval()
        if head.config.stage == "control":
            for branch in (head.one2many, head.one2one):
                for module in branch["box_head"]:
                    module.eval()
                for module in branch["cls_head"]:
                    module.train(True)
            head.geometry_levels.eval()
            head.frequency_levels.eval()
        elif head.config.stage == "geometry":
            head.base_head.eval()
            head.geometry_levels.train(True)
            head.frequency_levels.eval()
        else:
            head.base_head.eval()
            head.geometry_levels.eval()
            head.frequency_levels.train(True)
        return self

    def apply_freeze_policy(self) -> dict[str, int]:
        head = self.model[-1]
        if not isinstance(head, SGFRDetectHead):
            raise TypeError(type(head).__name__)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if head.config.stage == "control":
            modules = [
                module
                for branch in (head.one2many, head.one2one)
                for module in branch["cls_head"]
            ]
        elif head.config.stage == "geometry":
            modules = [head.geometry_levels]
        else:
            modules = [head.frequency_levels]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.train(self.training)
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }
