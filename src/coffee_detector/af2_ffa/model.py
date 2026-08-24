"""Classification-only feature-frequency adapter on top of AF2.

The AF2 image frontend remains unchanged. This module changes only the
classification inputs of the native YOLO26 Detect head. Regression always
receives the original P3/P4/P5 tensors.

Selective-refinement support is intentionally backward compatible with the
original AF2FFAB2 checkpoints. Runtime strength/level/fusion overrides are
non-persistent diagnostic controls; frozen config fields define any future
trained selective candidate.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from coffee_detector.afab.model import AFABDetectionModel
from .dct import selected_dct_descriptor


@dataclass(frozen=True)
class AF2FFAConfig:
    """Frozen capacity-matched control/candidate settings."""

    conditioning: str = "spectral"  # zero | spectral
    descriptor_type: str = "rfft_ratio"  # rfft_ratio | dct_selected
    radial_cutoff: float = 0.35
    eps: float = 1.0e-6
    max_added_fraction: float = 0.01
    residual_gain_cap: float | None = None
    gradient_matched_cap: bool = False

    # Selective-refinement fields. Defaults reproduce historical AF2FFAB2.
    adapter_strength_scale: float = 1.0
    active_levels: tuple[bool, bool, bool] = (True, True, True)  # P3/P4/P5
    fusion_mode: str = "replace"  # replace | parent_residual
    residual_mix: float = 1.0
    ambiguity_gate: str = "none"  # none | margin
    ambiguity_margin: float = 0.15
    ambiguity_temperature: float = 0.05

    @classmethod
    def from_mapping(
        cls, payload: "AF2FFAConfig | Mapping[str, Any] | None"
    ) -> "AF2FFAConfig":
        if isinstance(payload, cls):
            result = payload
        else:
            values = dict(payload or {})
            if "active_levels" in values:
                values["active_levels"] = tuple(bool(item) for item in values["active_levels"])
            result = cls(**values)
        if result.conditioning not in {"zero", "spectral"}:
            raise ValueError("conditioning harus zero atau spectral")
        if result.descriptor_type not in {"rfft_ratio", "dct_selected"}:
            raise ValueError("descriptor_type harus rfft_ratio atau dct_selected")
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
        if not 0.0 <= float(getattr(result, "adapter_strength_scale", 1.0)) <= 1.0:
            raise ValueError("adapter_strength_scale harus berada di [0, 1]")
        levels = tuple(getattr(result, "active_levels", (True, True, True)))
        if len(levels) != 3 or not any(levels):
            raise ValueError("active_levels harus tiga boolean dan minimal satu aktif")
        if getattr(result, "fusion_mode", "replace") not in {"replace", "parent_residual"}:
            raise ValueError("fusion_mode harus replace atau parent_residual")
        if not 0.0 <= float(getattr(result, "residual_mix", 1.0)) <= 1.0:
            raise ValueError("residual_mix harus berada di [0, 1]")
        if getattr(result, "ambiguity_gate", "none") not in {"none", "margin"}:
            raise ValueError("ambiguity_gate harus none atau margin")
        if not 0.0 <= float(getattr(result, "ambiguity_margin", 0.15)) <= 1.0:
            raise ValueError("ambiguity_margin harus berada di [0, 1]")
        if float(getattr(result, "ambiguity_temperature", 0.05)) <= 0.0:
            raise ValueError("ambiguity_temperature harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class FeatureFrequencyAdapter(nn.Module):
    """Channel-wise residual gate driven by fixed frequency evidence.

    ``alpha`` starts at zero, therefore the adapter is an exact identity at
    initialization. ``runtime_strength`` is a non-persistent diagnostic knob;
    it never changes checkpoint parameters.
    """

    def __init__(self, channels: int, config: AF2FFAConfig) -> None:
        super().__init__()
        self.channels = int(channels)
        self.config = config
        self.scale = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))
        self.alpha = nn.Parameter(torch.zeros(self.channels))
        self.runtime_strength = 1.0

    def _rfft_descriptor(self, value: torch.Tensor) -> torch.Tensor:
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
        return (high_energy / total_energy).to(original_dtype)

    def spectral_descriptor(self, value: torch.Tensor) -> torch.Tensor:
        if self.config.descriptor_type == "rfft_ratio":
            descriptor = self._rfft_descriptor(value)
        elif self.config.descriptor_type == "dct_selected":
            descriptor = selected_dct_descriptor(value, eps=self.config.eps)
        else:  # guarded by config validation; keep fail-closed for checkpoints.
            raise RuntimeError(f"Unknown descriptor_type={self.config.descriptor_type!r}")
        if self.config.conditioning == "zero":
            descriptor = torch.zeros_like(descriptor)
        return descriptor

    def set_runtime_strength(self, strength: float) -> None:
        value = float(strength)
        if not 0.0 <= value <= 1.0:
            raise ValueError("runtime strength harus berada di [0, 1]")
        self.runtime_strength = value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        descriptor = self.spectral_descriptor(value)
        gate = torch.tanh(
            descriptor * self.scale[None, :] + self.bias[None, :]
        )
        configured = float(getattr(self.config, "adapter_strength_scale", 1.0))
        amplitude = configured * float(self.runtime_strength) * self.residual_amplitude()
        multiplier = 1.0 + amplitude[None, :] * gate
        return value * multiplier[:, :, None, None]

    def residual_amplitude(self) -> torch.Tensor:
        """Return learned residual amplitude under the frozen parameterization."""

        cap = self.config.residual_gain_cap
        if cap is None:
            return self.alpha
        if self.config.gradient_matched_cap:
            return float(cap) * torch.tanh(self.alpha / float(cap))
        return float(cap) * torch.tanh(self.alpha)


class AF2FFADetectHead(nn.Module):
    """Native YOLO26 boxes with selectable FFAB classification refinement."""

    LEVEL_NAMES = ("P3", "P4", "P5")

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
        self.runtime_active_levels = tuple(getattr(config, "active_levels", (True, True, True)))
        self.runtime_fusion_mode: str | None = None
        self.runtime_residual_mix: float | None = None
        self.runtime_ambiguity_gate: str | None = None
        self.runtime_ambiguity_margin: float | None = None
        self.runtime_ambiguity_temperature: float | None = None
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

    @staticmethod
    def _normalize_levels(levels: Sequence[bool] | Sequence[str]) -> tuple[bool, bool, bool]:
        values = tuple(levels)
        if len(values) == 3 and all(isinstance(item, bool) for item in values):
            result = tuple(bool(item) for item in values)
        else:
            names = {str(item).upper() for item in values}
            unknown = names - set(AF2FFADetectHead.LEVEL_NAMES)
            if unknown:
                raise ValueError(f"Level tidak dikenal: {sorted(unknown)}")
            result = tuple(name in names for name in AF2FFADetectHead.LEVEL_NAMES)
        if not any(result):
            raise ValueError("Minimal satu level P3/P4/P5 harus aktif")
        return result

    def set_runtime_ablation(
        self,
        *,
        strength: float = 1.0,
        active_levels: Sequence[bool] | Sequence[str] = (True, True, True),
        fusion_mode: str | None = None,
        residual_mix: float | None = None,
        ambiguity_gate: str | None = None,
        ambiguity_margin: float | None = None,
        ambiguity_temperature: float | None = None,
    ) -> None:
        for adapter in self.adapters:
            adapter.set_runtime_strength(strength)
        self.runtime_active_levels = self._normalize_levels(active_levels)
        if fusion_mode is not None and fusion_mode not in {"replace", "parent_residual"}:
            raise ValueError("fusion_mode runtime harus replace atau parent_residual")
        if residual_mix is not None and not 0.0 <= float(residual_mix) <= 1.0:
            raise ValueError("residual_mix runtime harus di [0, 1]")
        if ambiguity_gate is not None and ambiguity_gate not in {"none", "margin"}:
            raise ValueError("ambiguity_gate runtime harus none atau margin")
        if ambiguity_margin is not None and not 0.0 <= float(ambiguity_margin) <= 1.0:
            raise ValueError("ambiguity_margin runtime harus di [0, 1]")
        if ambiguity_temperature is not None and float(ambiguity_temperature) <= 0.0:
            raise ValueError("ambiguity_temperature runtime harus positif")
        self.runtime_fusion_mode = fusion_mode
        self.runtime_residual_mix = None if residual_mix is None else float(residual_mix)
        self.runtime_ambiguity_gate = ambiguity_gate
        self.runtime_ambiguity_margin = (
            None if ambiguity_margin is None else float(ambiguity_margin)
        )
        self.runtime_ambiguity_temperature = (
            None if ambiguity_temperature is None else float(ambiguity_temperature)
        )

    def reset_runtime_ablation(self) -> None:
        for adapter in self.adapters:
            adapter.set_runtime_strength(1.0)
        self.runtime_active_levels = tuple(getattr(self.config, "active_levels", (True, True, True)))
        self.runtime_fusion_mode = None
        self.runtime_residual_mix = None
        self.runtime_ambiguity_gate = None
        self.runtime_ambiguity_margin = None
        self.runtime_ambiguity_temperature = None

    def runtime_state(self) -> dict[str, Any]:
        return {
            "strength": float(self.adapters[0].runtime_strength),
            "active_levels": {
                name: bool(enabled)
                for name, enabled in zip(self.LEVEL_NAMES, self.runtime_active_levels)
            },
            "fusion_mode": self._effective("fusion_mode"),
            "residual_mix": float(self._effective("residual_mix")),
            "ambiguity_gate": self._effective("ambiguity_gate"),
            "ambiguity_margin": float(self._effective("ambiguity_margin")),
            "ambiguity_temperature": float(self._effective("ambiguity_temperature")),
        }

    def _effective(self, name: str):
        runtime = getattr(self, f"runtime_{name}")
        if runtime is not None:
            return runtime
        defaults = {
            "fusion_mode": "replace",
            "residual_mix": 1.0,
            "ambiguity_gate": "none",
            "ambiguity_margin": 0.15,
            "ambiguity_temperature": 0.05,
        }
        return getattr(self.config, name, defaults[name])

    def _adapt_feature(self, index: int, feature: torch.Tensor) -> torch.Tensor:
        if not self.runtime_active_levels[index]:
            return feature
        return self.adapters[index](feature)

    def _ambiguity_weight(self, native_scores: torch.Tensor) -> torch.Tensor | float:
        if self._effective("ambiguity_gate") == "none":
            return 1.0
        probabilities = native_scores.sigmoid()
        if probabilities.shape[1] < 2:
            margin = probabilities[:, :1]
        else:
            top2 = torch.topk(probabilities, k=2, dim=1).values
            margin = top2[:, :1] - top2[:, 1:2]
        threshold = float(self._effective("ambiguity_margin"))
        temperature = float(self._effective("ambiguity_temperature"))
        # Detach prevents the classifier from gaming the routing variable.
        return torch.sigmoid((threshold - margin) / temperature).detach()

    def _classification_scores(
        self,
        index: int,
        feature: torch.Tensor,
        cls_head: nn.Module,
    ) -> torch.Tensor:
        adapted = self._adapt_feature(index, feature)
        fusion_mode = self._effective("fusion_mode")
        if fusion_mode == "replace":
            return cls_head(adapted)
        native = cls_head(feature)
        if adapted is feature:
            return native
        refined = cls_head(adapted)
        mix = float(self._effective("residual_mix"))
        gate = self._ambiguity_weight(native)
        return native + mix * gate * (refined - native)

    def _forward_branch(
        self, features: list[torch.Tensor], branch: dict[str, nn.Module]
    ) -> dict[str, torch.Tensor]:
        boxes, scores = [], []
        batch = features[0].shape[0]
        for index, feature in enumerate(features):
            boxes.append(branch["box_head"][index](feature))
            scores.append(
                self._classification_scores(index, feature, branch["cls_head"][index])
            )
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
    """Load a native/AF2 source or resume a complete AF2-FFA checkpoint."""

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
            runtime_value = getattr(source_head, name)
            setattr(target_head, name, runtime_value)
            setattr(target_head.base_head, name, runtime_value)
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}
