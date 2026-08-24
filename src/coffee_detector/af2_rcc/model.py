"""Classification-only calibration from AF2's already-computed spatial cue.

The detector still performs exactly one AF2 recovery at the input.  Its cue is
reused at P3/P4/P5 to add a small class-specific logit residual.  Geometry and
the native AF2 detector remain frozen and untouched.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import afab_gate, minmax_spatial


@dataclass(frozen=True)
class AF2RCCConfig:
    conditioning: str = "recovered"  # zero | recovered
    gain_cap: float = 0.10
    cue_channels: int = 3
    expected_levels: int = 3
    max_added_parameters: int = 256

    @classmethod
    def from_mapping(
        cls, payload: "AF2RCCConfig | Mapping[str, Any] | None"
    ) -> "AF2RCCConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.conditioning not in {"zero", "recovered"}:
            raise ValueError("conditioning harus zero atau recovered")
        if not 0.0 < result.gain_cap <= 0.25:
            raise ValueError("gain_cap harus berada di (0, 0.25]")
        if result.cue_channels != 3 or result.expected_levels != 3:
            raise ValueError("AF2-RCC dikunci untuk cue RGB dan P3/P4/P5")
        if result.max_added_parameters <= 0:
            raise ValueError("max_added_parameters harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecoveredCueClassCalibration(nn.Module):
    """Bounded per-class projection of a spatial RGB AF2 cue.

    The ``gain_cap * tanh(weight / gain_cap)`` parameterization is identity at
    zero, has unit derivative at initialization, and bounds every complete
    RGB correction to ``[-gain_cap, +gain_cap]`` by averaging three channels.
    """

    def __init__(self, num_classes: int, config: AF2RCCConfig) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.config = config
        self.weight = nn.Parameter(
            torch.zeros(self.num_classes, int(config.cue_channels))
        )

    def bounded_weight(self) -> torch.Tensor:
        cap = float(self.config.gain_cap)
        return cap * torch.tanh(self.weight / cap)

    def forward(self, scores: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 4 or cue.ndim != 4:
            raise ValueError("scores dan cue harus BCHW")
        if scores.shape[1] != self.num_classes:
            raise ValueError("Jumlah kelas scores tidak sesuai kalibrator")
        if cue.shape[1] != self.config.cue_channels:
            raise ValueError("Cue AF2 harus mempunyai tiga channel RGB")
        if cue.shape[-2:] != scores.shape[-2:]:
            cue = F.interpolate(cue, size=scores.shape[-2:], mode="area")
        if self.config.conditioning == "zero":
            cue = torch.zeros_like(cue)
        weight = self.bounded_weight().to(dtype=scores.dtype)
        correction = torch.einsum("bchw,kc->bkhw", cue, weight)
        correction = correction / float(self.config.cue_channels)
        return scores + correction


class AF2RCCDetectHead(nn.Module):
    """Native YOLO26 boxes and logits plus a cue-conditioned class residual."""

    def __init__(self, base_head: nn.Module, config: AF2RCCConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(
            base_head, "end2end", False
        ):
            raise TypeError("AF2-RCC memerlukan native YOLO26 end-to-end Detect")
        if int(base_head.nl) != config.expected_levels:
            raise ValueError("AF2-RCC memerlukan tepat P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.calibrators = nn.ModuleList(
            [RecoveredCueClassCalibration(int(base_head.nc), config) for _ in range(base_head.nl)]
        )
        self._spatial_cue: torch.Tensor | None = None
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

    def set_spatial_cue(self, cue: torch.Tensor) -> None:
        if cue.ndim != 4 or cue.shape[1] != self.config.cue_channels:
            raise ValueError("Cue AF2 harus BCHW RGB")
        self._spatial_cue = cue

    def _take_spatial_cue(self) -> torch.Tensor:
        if self._spatial_cue is None:
            raise RuntimeError("AF2-RCC head dipanggil tanpa recovered cue")
        cue = self._spatial_cue
        self._spatial_cue = None
        return cue

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
        cue: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        boxes, scores = [], []
        batch = features[0].shape[0]
        for index, feature in enumerate(features):
            boxes.append(branch["box_head"][index](feature))
            native_scores = branch["cls_head"][index](feature)
            scores.append(self.calibrators[index](native_scores, cue))
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
        cue = self._take_spatial_cue()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many, cue),
                "one2one": self._forward_branch(
                    [item.detach() for item in features], self.one2one, cue.detach()
                ),
            }
        one2many = (
            self._forward_branch(features, self.one2many, cue)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [item.detach() for item in features], self.one2one, cue.detach()
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


class AF2RCCDetectionModel(AFABDetectionModel):
    def __init__(
        self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, afab=None, af2_rcc=None
    ) -> None:
        self.af2_rcc_config = AF2RCCConfig.from_mapping(af2_rcc)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = AF2RCCDetectHead(self.model[-1], self.af2_rcc_config)
        freeze_for_rcc(self)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            raise ValueError("AF2-RCC tidak mendukung augmented inference")
        enhancer = getattr(self, "afab", None)
        head = self.model[-1]
        if enhancer is not None and isinstance(x, torch.Tensor):
            recovered = enhancer.recover(x)
            cue = minmax_spatial(recovered, eps=enhancer.config.eps)
            x = afab_gate(x, recovered, eps=enhancer.config.eps)
            head.set_spatial_cue(cue)
        # Skip AFABDetectionModel.predict so AF2 recovery is executed once.
        return super(AFABDetectionModel, self).predict(
            x, profile=profile, visualize=visualize, augment=False, embed=embed
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and len(self.model):
            head = self.model[-1]
            for layer in list(self.model)[:-1]:
                layer.eval()
            head.base_head.eval()
            head.train(True)
            head.base_head.eval()
            head.calibrators.train(True)
        return self


def freeze_for_rcc(model: nn.Module) -> dict[str, int]:
    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)) or not isinstance(
        layers[-1], AF2RCCDetectHead
    ):
        raise TypeError("freeze_for_rcc memerlukan AF2RCCDetectHead")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in layers[-1].calibrators.parameters():
        parameter.requires_grad_(True)
    model.train(model.training)
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def load_af2_rcc_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load an original AF2 checkpoint or resume a complete AF2-RCC model."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, AF2RCCDetectHead):
        raise TypeError("Target bukan AF2RCCDetectHead")
    if isinstance(source_head, AF2RCCDetectHead):
        target_head.load_state_dict(source_head.state_dict(), strict=True)
        resume = 1
    else:
        if type(source_head).__name__ != "Detect":
            raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
        result = target_head.base_head.load_state_dict(
            copy.deepcopy(source_head.state_dict()), strict=True
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Transfer native Detect ke AF2-RCC tidak lengkap")
        resume = 0
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    freeze_for_rcc(model)
    return {"native_head_items": len(target_head.base_head.state_dict()), "resume": resume}
