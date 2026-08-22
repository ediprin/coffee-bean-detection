"""Classification-only feature-frequency adapter on top of AF2.

The AF2 image frontend remains unchanged.  This module changes only the
classification inputs of the native YOLO26 Detect head.  Regression always
receives the original P3/P4/P5 tensors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
import copy

import torch
from torch import nn

from coffee_detector.afab.model import AFABDetectionModel


@dataclass(frozen=True)
class AF2FFAConfig:
    """Frozen capacity-matched control/candidate settings."""

    conditioning: str = "spectral"  # zero | spectral
    radial_cutoff: float = 0.35
    eps: float = 1.0e-6
    max_added_fraction: float = 0.01
    residual_gain_cap: float | None = None
    gradient_matched_cap: bool = False

    @classmethod
    def from_mapping(
        cls, payload: "AF2FFAConfig | Mapping[str, Any] | None"
    ) -> "AF2FFAConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.conditioning not in {"zero", "spectral"}:
            raise ValueError("conditioning harus zero atau spectral")
        if not 0.0 < result.radial_cutoff < 1.0:
            raise ValueError("radial_cutoff harus berada di (0, 1)")
        if result.eps <= 0 or not 0.0 < result.max_added_fraction <= 0.05:
            raise ValueError("eps/max_added_fraction tidak valid")
        if result.residual_gain_cap is not None and not (
            0.0 < result.residual_gain_cap <= 0.5
        ):
            raise ValueError("residual_gain_cap harus berada di (0, 0.5]")
        if result.gradient_matched_cap and result.residual_gain_cap is None:
            raise ValueError("gradient_matched_cap memerlukan residual_gain_cap")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class FeatureFrequencyAdapter(nn.Module):
    """Channel-wise residual gate driven by fixed radial spectral energy.

    ``alpha`` starts at zero, therefore the adapter is an exact identity at
    initialization.  The zero control has the same parameters and computation
    schema but receives an all-zero spectral descriptor.
    """

    def __init__(self, channels: int, config: AF2FFAConfig) -> None:
        super().__init__()
        self.channels = int(channels)
        self.config = config
        self.scale = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))
        self.alpha = nn.Parameter(torch.zeros(self.channels))

    def spectral_descriptor(self, value: torch.Tensor) -> torch.Tensor:
        original_dtype = value.dtype
        work = value.float()
        spectrum = torch.fft.rfft2(work, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        height, width_rfft = power.shape[-2:]
        fy = torch.fft.fftfreq(height, device=value.device, dtype=work.dtype)
        full_width = max(2, 2 * (width_rfft - 1))
        fx = torch.fft.rfftfreq(full_width, device=value.device, dtype=work.dtype)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        radius = radius / radius.max().clamp_min(self.config.eps)
        high = (radius >= float(self.config.radial_cutoff)).to(power.dtype)
        high_energy = (power * high).sum(dim=(-2, -1))
        total_energy = power.sum(dim=(-2, -1)).clamp_min(self.config.eps)
        descriptor = high_energy / total_energy
        if self.config.conditioning == "zero":
            descriptor = torch.zeros_like(descriptor)
        return descriptor.to(original_dtype)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        descriptor = self.spectral_descriptor(value)
        gate = torch.tanh(
            descriptor * self.scale[None, :] + self.bias[None, :]
        )
        amplitude = self.residual_amplitude()
        multiplier = 1.0 + amplitude[None, :] * gate
        return value * multiplier[:, :, None, None]

    def residual_amplitude(self) -> torch.Tensor:
        """Return the learned residual amplitude under the frozen parameterization.

        ``gradient_matched_cap`` preserves the unbounded arm's unit derivative
        at ``alpha=0`` while still bounding the deployed amplitude to ``±cap``.
        The legacy bounded arm deliberately retains its historical
        ``cap*tanh(alpha)`` behavior for reproducibility.
        """

        cap = self.config.residual_gain_cap
        if cap is None:
            return self.alpha
        if self.config.gradient_matched_cap:
            return float(cap) * torch.tanh(self.alpha / float(cap))
        return float(cap) * torch.tanh(self.alpha)


class AF2FFADetectHead(nn.Module):
    """Native YOLO26 boxes with adapted classification features only."""

    def __init__(self, base_head: nn.Module, config: AF2FFAConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("AF2-FFA memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("AF2-FFA memerlukan tepat P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.adapters = nn.ModuleList(
            [FeatureFrequencyAdapter(channel, config) for channel in channels]
        )
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

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def _forward_branch(
        self, features: list[torch.Tensor], branch: dict[str, nn.Module]
    ) -> dict[str, torch.Tensor]:
        boxes, scores = [], []
        batch = features[0].shape[0]
        for index, feature in enumerate(features):
            boxes.append(branch["box_head"][index](feature))
            adapted = self.adapters[index](feature)
            scores.append(branch["cls_head"][index](adapted))
        return {
            "boxes": torch.cat(
                [item.view(batch, 4 * self.reg_max, -1) for item in boxes], dim=-1
            ),
            "scores": torch.cat(
                [item.view(batch, self.nc, -1) for item in scores], dim=-1
            ),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many),
                "one2one": self._forward_branch(
                    [item.detach() for item in features], self.one2one
                ),
            }
        one2many = (
            self._forward_branch(features, self.one2many)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [item.detach() for item in features], self.one2one
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


class AF2FFADetectionModel(AFABDetectionModel):
    def __init__(
        self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, afab=None, af2_ffa=None
    ) -> None:
        self.af2_ffa_config = AF2FFAConfig.from_mapping(af2_ffa)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = AF2FFADetectHead(self.model[-1], self.af2_ffa_config)


def load_af2_ffa_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load an AF2 source or resume a complete AF2-FFA checkpoint."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, AF2FFADetectHead):
        raise TypeError("Target bukan AF2FFADetectHead")
    if isinstance(source_head, AF2FFADetectHead):
        target_head.load_state_dict(source_head.state_dict(), strict=True)
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(copy.deepcopy(source_head.state_dict()), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect ke AF2-FFA tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}
