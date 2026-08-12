"""Ambiguity-gated spatial-frequency classification synthesis for YOLO26."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer
from coffee_detector.stb.model import ClassificationSTB, STBConfig


@dataclass(frozen=True)
class AGSFConfig:
    """Frozen configuration for the three predeclared synthesis arms."""

    frequency_mode: str = "none"  # none | additive | gated
    hidden_dim: int = 64
    context_kernel: int = 3
    correction_scale: float = 1.0
    ambiguity_floor: float = 0.0
    window_size: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    af2_patch_size: int = 32
    af2_overlap: float = 0.50
    af2_radius_ratio: float = 0.05
    af2_gamma: float = 0.10

    @classmethod
    def from_mapping(cls, payload: "AGSFConfig | dict[str, Any] | None") -> "AGSFConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.frequency_mode not in {"none", "additive", "gated"}:
            raise ValueError("frequency_mode harus none, additive, atau gated")
        if result.hidden_dim <= 0 or result.context_kernel not in {3, 5}:
            raise ValueError("hidden_dim/context_kernel AGSF tidak valid")
        if result.correction_scale <= 0 or not 0.0 <= result.ambiguity_floor < 1.0:
            raise ValueError("correction_scale/ambiguity_floor AGSF tidak valid")
        STBConfig(
            window_size=result.window_size,
            num_heads=result.num_heads,
            mlp_ratio=result.mlp_ratio,
        )
        AFABConfig(
            mode="af2",
            patch_size=result.af2_patch_size,
            overlap=result.af2_overlap,
            radius_ratio=result.af2_radius_ratio,
            gamma=result.af2_gamma,
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stb_config(self) -> STBConfig:
        return STBConfig(
            window_size=self.window_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
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


class _DepthwiseContext(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.activation(self.norm(self.depthwise(value))))


class AGSFClassificationCorrection(nn.Module):
    """Fuse STB-enhanced P3/P4/P5 and an optional AF2 residual cue.

    SYN1 and SYN2 instantiate an identical frequency encoder and gate schema.
    SYN1 fixes the frequency coefficient to one, whereas SYN2 learns a
    spatial ambiguity-conditioned coefficient.  This makes their parameter
    counts identical while isolating the gating policy.
    """

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
        config: AGSFConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        hidden = int(config.hidden_dim)
        self.projections = nn.ModuleList(
            [nn.Conv2d(channel, hidden, 1, bias=False) for channel in channels]
        )
        self.level_selectors = nn.ModuleList(
            [nn.Conv2d(3 * hidden + 1, 3, 1) for _ in channels]
        )
        self.contexts = nn.ModuleList(
            [_DepthwiseContext(hidden, config.context_kernel) for _ in channels]
        )
        self.class_corrections = nn.ModuleList(
            [nn.Conv2d(hidden, self.num_classes, 1) for _ in channels]
        )
        for layer in self.class_corrections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        if config.frequency_mode == "none":
            self.frequency_encoders = nn.ModuleList()
            self.frequency_gates = nn.ModuleList()
        else:
            self.frequency_encoders = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(3, hidden, 3, padding=1, bias=False),
                        nn.BatchNorm2d(hidden),
                        nn.SiLU(inplace=True),
                    )
                    for _ in channels
                ]
            )
            # Both frequency arms keep the same state-dict/parameter count.
            self.frequency_gates = nn.ModuleList(
                [nn.Conv2d(2 * hidden + 1, 1, 1) for _ in channels]
            )

    def _ambiguity(self, logits: torch.Tensor) -> torch.Tensor:
        probability = logits.detach().softmax(dim=1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(max(self.num_classes, 2))
        if self.config.ambiguity_floor:
            entropy = (entropy - self.config.ambiguity_floor).clamp_min(0.0)
            entropy = entropy / (1.0 - self.config.ambiguity_floor)
        return entropy

    def forward(
        self,
        enhanced_features: list[torch.Tensor],
        level_logits: list[torch.Tensor],
        frequency_residual: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if len(enhanced_features) != 3 or len(level_logits) != 3:
            raise ValueError("AGSF memerlukan tepat P3, P4, dan P5")
        if self.config.frequency_mode != "none" and frequency_residual is None:
            raise RuntimeError("Arm frequency AGSF tidak menerima residual AF2")
        projected = [
            projection(feature)
            for projection, feature in zip(self.projections, enhanced_features)
        ]
        corrections: list[torch.Tensor] = []
        for target, logits in enumerate(level_logits):
            size = projected[target].shape[-2:]
            aligned = [
                value if value.shape[-2:] == size else F.interpolate(value, size=size, mode="nearest")
                for value in projected
            ]
            ambiguity = self._ambiguity(logits)
            weights = self.level_selectors[target](
                torch.cat((*aligned, ambiguity), dim=1)
            ).softmax(dim=1)
            spatial = sum(
                weights[:, index : index + 1] * value
                for index, value in enumerate(aligned)
            )
            fused = spatial
            if frequency_residual is not None:
                resized = F.interpolate(
                    frequency_residual,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )
                frequency = self.frequency_encoders[target](resized)
                gate_logits = self.frequency_gates[target](
                    torch.cat((spatial, frequency, ambiguity), dim=1)
                )
                if self.config.frequency_mode == "gated":
                    alpha = gate_logits.sigmoid()
                else:
                    # Keep an identical computation graph/schema for the
                    # capacity control, but freeze the policy to full addition.
                    alpha = torch.ones_like(gate_logits) + 0.0 * gate_logits
                fused = spatial + alpha * frequency
            correction = self.class_corrections[target](self.contexts[target](fused))
            corrections.append(
                float(self.config.correction_scale) * ambiguity * correction
            )
        return corrections


class AGSFDetectHead(nn.Module):
    """Native YOLO26 boxes plus an AGSF classification-only pathway."""

    def __init__(self, base_head: nn.Module, config: AGSFConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("AGSF memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("AGSF memerlukan P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.stb_blocks = nn.ModuleList(
            [ClassificationSTB(channel, config.stb_config()) for channel in channels]
        )
        self.correction = AGSFClassificationCorrection(
            channels, int(base_head.nc), config
        )
        self.af2 = (
            AFABInputEnhancer(config.af2_config())
            if config.frequency_mode != "none"
            else None
        )
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
        self._frequency_residual = (
            self.af2(image) - image if self.af2 is not None else None
        )

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
        self, features: list[torch.Tensor], branch: dict[str, nn.Module]
    ) -> dict[str, torch.Tensor]:
        enhanced = [
            block(feature) for block, feature in zip(self.stb_blocks, features)
        ]
        boxes, logits = [], []
        for index in range(self.nl):
            # Localization sees the untouched native pyramid feature.
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](enhanced[index]))
        corrections = self.correction(
            enhanced, logits, self._frequency_residual
        )
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat(
                [value.view(batch, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [
                    (logit + correction).view(batch, self.nc, -1)
                    for logit, correction in zip(logits, corrections)
                ],
                dim=-1,
            ),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.config.frequency_mode != "none" and self._frequency_residual is None:
            raise RuntimeError("AGSF frequency arm dipanggil tanpa input image")
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many),
                "one2one": self._forward_branch(
                    [value.detach() for value in features], self.one2one
                ),
            }
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


def load_agsf_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly map native D0 or resume a complete AGSF checkpoint."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, AGSFDetectHead):
        raise TypeError("Target bukan AGSFDetectHead")
    if isinstance(source_head, AGSFDetectHead):
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect ke AGSF tidak lengkap")
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


class AGSFDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg="yolo26.yaml",
        ch=3,
        nc=None,
        verbose=True,
        agsf=None,
    ) -> None:
        self.agsf_config = AGSFConfig.from_mapping(agsf)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = AGSFDetectHead(self.model[-1], self.agsf_config)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        head = self.model[-1]
        if not isinstance(head, AGSFDetectHead):
            # DetectionModel computes native strides during super().__init__
            # before this subclass replaces Detect with AGSFDetectHead.
            return super().predict(
                x,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
            )
        head.prepare_frequency(x)
        try:
            return super().predict(
                x,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
            )
        finally:
            head.clear_frequency()
