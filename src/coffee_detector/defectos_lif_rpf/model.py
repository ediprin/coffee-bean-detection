from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .config import LIFRPFConfig
from .operator import luminance_residual_cue


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class BoundedCueResidual(nn.Module):
    """Identity-initialized bounded cue residual for one pyramid level."""

    def __init__(self, channels: int, cue_channels: int, maximum_gain: float) -> None:
        super().__init__()
        self.projection = nn.Conv2d(cue_channels, 2 * channels, 1, bias=True)
        self.gate = nn.Parameter(torch.zeros(()))
        self.maximum_gain = float(maximum_gain)

    def forward(self, feature: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            cue, size=feature.shape[-2:], mode="bilinear", align_corners=False
        )
        scale, bias = torch.tanh(self.projection(resized)).chunk(2, dim=1)
        gain = self.maximum_gain * torch.tanh(self.gate)
        return feature + gain * (feature * scale + bias)


class LIFRPFDetectHead(nn.Module):
    """Native YOLO boxes plus luminance-cue classification conditioning."""

    def __init__(self, base_head: nn.Module, config: LIFRPFConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("LIF-RPF memerlukan native Detect")
        self.base_head = base_head
        self.config = LIFRPFConfig.from_mapping(config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.adapters = nn.ModuleList(
            [
                BoundedCueResidual(
                    channel,
                    self.config.cue_channels,
                    self.config.maximum_feature_gain,
                )
                for channel in channels
            ]
        )
        self.current_cue: torch.Tensor | None = None
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

    def set_cue(self, cue: torch.Tensor) -> None:
        self.current_cue = cue

    def _sync_runtime_attributes(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    def _forward_head(
        self,
        features: list[torch.Tensor],
        *,
        box_head: nn.Module,
        cls_head: nn.Module,
    ) -> dict[str, torch.Tensor]:
        if self.current_cue is None:
            raise RuntimeError("Cue LIF-RPF belum disetel")
        batch = features[0].shape[0]
        boxes, scores = [], []
        for index in range(self.nl):
            feature = features[index]
            conditioned = self.adapters[index](feature, self.current_cue)
            boxes.append(box_head[index](feature).view(batch, 4 * self.reg_max, -1))
            scores.append(cls_head[index](conditioned).view(batch, self.nc, -1))
        return {
            "boxes": torch.cat(boxes, dim=-1),
            "scores": torch.cat(scores, dim=-1),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        one2many = self.base_head.one2many
        fused = one2many.get("box_head") is None or one2many.get("cls_head") is None
        predictions = None if fused else self._forward_head(features, **one2many)
        if self.end2end:
            detached = [value.detach() for value in features]
            one2one = self._forward_head(detached, **self.base_head.one2one)
            predictions = (
                {"one2one": one2one}
                if fused
                else {"one2many": predictions, "one2one": one2one}
            )
        elif predictions is None:
            raise RuntimeError("Head LIF-RPF fused tidak boleh dipakai untuk training")
        self.current_cue = None
        if self.training:
            return predictions
        inference = self.base_head._inference(
            predictions["one2one"] if self.end2end else predictions
        )
        if self.end2end:
            inference = self.base_head.postprocess(inference.permute(0, 2, 1))
        return inference if self.export else (inference, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


from ultralytics.nn.tasks import DetectionModel


class LIFRPFDetectionModel(DetectionModel):
    """YOLO with raw RGB localization and LIF-preprocessed score cues."""

    def __init__(
        self,
        cfg: str | dict,
        *,
        ch: int = 3,
        nc: int = 12,
        verbose: bool = False,
        lif_rpf: LIFRPFConfig | Mapping[str, Any] | None = None,
        native_source: nn.Module | None = None,
    ) -> None:
        self.lif_rpf_config = LIFRPFConfig.from_mapping(lif_rpf)
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        if native_source is not None:
            self.load(native_source)
        self.model[-1] = LIFRPFDetectHead(self.model[-1], self.lif_rpf_config)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        head = self.model[-1] if hasattr(self, "model") and len(self.model) else None
        if isinstance(x, torch.Tensor) and isinstance(head, LIFRPFDetectHead):
            head.set_cue(luminance_residual_cue(x, self.lif_rpf_config).detach())
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def build_lif_rpf_model(
    model_yaml: str,
    *,
    nc: int,
    source: nn.Module | None,
    seed: int,
    config: LIFRPFConfig | Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> LIFRPFDetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return LIFRPFDetectionModel(
            model_yaml,
            ch=3,
            nc=nc,
            verbose=verbose,
            lif_rpf=config,
            native_source=source,
        )


def load_lif_rpf_weights(model: nn.Module, weights: Any) -> None:
    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    if not isinstance(source, nn.Module):
        raise TypeError("Checkpoint LIF-RPF tidak mengekspos model torch")
    result = model.load_state_dict(source.float().state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Restore LIF-RPF tidak lengkap")
