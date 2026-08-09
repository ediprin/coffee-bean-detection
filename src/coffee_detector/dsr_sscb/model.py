from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SSCBConfig:
    mode: str = "calibrated"  # msda | semantic_aux | calibrated
    hidden_dim: int = 64
    sampling_points: int = 4
    max_offset_pixels: float = 2.0
    semantic_aux_weight: float = 0.2
    correction_scale: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "SSCBConfig | dict[str, Any] | None") -> "SSCBConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.mode not in {"msda", "semantic_aux", "calibrated"}:
            raise ValueError("mode harus msda, semantic_aux, atau calibrated")
        if result.hidden_dim <= 0 or result.sampling_points <= 0:
            raise ValueError("hidden_dim/sampling_points harus positif")
        if result.max_offset_pixels <= 0 or result.semantic_aux_weight < 0 or result.correction_scale <= 0:
            raise ValueError("parameter SSCB tidak valid")
        return result

    @property
    def uses_semantics(self) -> bool:
        return self.mode in {"semantic_aux", "calibrated"}

    @property
    def uses_calibration(self) -> bool:
        return self.mode == "calibrated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


def _base_grid(height: int, width: int, *, device, dtype) -> torch.Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("Ukuran grid harus positif")
    if height == 1:
        y = torch.zeros(1, device=device, dtype=dtype)
    else:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    if width == 1:
        x = torch.zeros(1, device=device, dtype=dtype)
    else:
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


class SharedSemanticGenerator(nn.Module):
    """BBox-supervised replacement for DSRDet's CLIP-attention shared-semantic space.

    DSRDet supervises shared semantics using shallow-layer CLIP attention on
    foreground-activation images. Coffee transfer cannot claim that generator;
    this branch learns a dense shared-foreground space from bbox raster targets.
    """

    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.foreground_logit = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        semantic = self.project(feature)
        return semantic, self.foreground_logit(semantic)


class CalibratedMSDALevel(nn.Module):
    """Single-head pure-PyTorch transfer of DSRDet Eq. (16)-(19).

    Query resolution is the target pyramid level. Each query samples K points
    from every source level. Offsets use CtM calibration; attention/value use
    MtC-style additive calibration. Sampling is implemented with grid_sample.
    """

    def __init__(self, hidden_dim: int, levels: int, points: int, config: SSCBConfig) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.levels = int(levels)
        self.points = int(points)
        self.config = config
        n = self.levels * self.points
        self.offset_map = nn.Conv2d(hidden_dim, 2 * n, 1)
        self.attention_map = nn.Conv2d(hidden_dim, n, 1)
        self.semantic_attention_map = nn.Conv2d(hidden_dim, n, 1)
        self.value_maps = nn.ModuleList([nn.Conv2d(hidden_dim, hidden_dim, 1) for _ in range(levels)])
        self.semantic_value_maps = nn.ModuleList(
            [nn.Conv2d(hidden_dim, hidden_dim, 1) for _ in range(levels)]
        )
        self.output = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.lambda_p = nn.Parameter(torch.tensor(0.0))
        self.lambda_a = nn.Parameter(torch.tensor(0.0))
        self.lambda_v = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.offset_map.weight)
        nn.init.zeros_(self.offset_map.bias)

    def forward(
        self,
        query: torch.Tensor,
        values: list[torch.Tensor],
        semantic_query: torch.Tensor | None,
        semantic_values: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, _, ht, wt = query.shape
        calibrated = self.config.uses_calibration and semantic_query is not None
        offset_input = query + self.lambda_p * semantic_query if calibrated else query
        offsets = self.offset_map(offset_input)
        attn = self.attention_map(query)
        if calibrated:
            attn = attn + self.lambda_a * self.semantic_attention_map(semantic_query)
        attn = torch.softmax(attn, dim=1)

        offsets = offsets.view(b, self.levels, self.points, 2, ht, wt)
        attn = attn.view(b, self.levels, self.points, 1, ht, wt)
        base = _base_grid(ht, wt, device=query.device, dtype=query.dtype)
        aggregate = torch.zeros_like(query)

        for level in range(self.levels):
            value = self.value_maps[level](values[level])
            if calibrated and semantic_values is not None:
                value = value + self.lambda_v * self.semantic_value_maps[level](semantic_values[level])
            hs, ws = value.shape[-2:]
            for point in range(self.points):
                delta = torch.tanh(offsets[:, level, point].permute(0, 2, 3, 1))
                dx = delta[..., 0] * (2.0 * self.config.max_offset_pixels / max(float(ws - 1), 1.0))
                dy = delta[..., 1] * (2.0 * self.config.max_offset_pixels / max(float(hs - 1), 1.0))
                grid = base + torch.stack((dx, dy), dim=-1)
                if grid.shape[0] == 1 and b > 1:
                    grid = grid.expand(b, -1, -1, -1)
                sampled = F.grid_sample(
                    value,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                aggregate = aggregate + attn[:, level, point] * sampled
        output = query + self.output(aggregate)
        diagnostics = {
            "lambda_p": self.lambda_p,
            "lambda_a": self.lambda_a,
            "lambda_v": self.lambda_v,
            "attention": attn,
        }
        return output, diagnostics


class SSCBClassificationPath(nn.Module):
    def __init__(self, channels: tuple[int, int, int], num_classes: int, config: SSCBConfig) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("SSCB memerlukan P3/P4/P5")
        self.config = config
        self.input_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(c), config.hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(config.hidden_dim),
                    nn.SiLU(inplace=True),
                )
                for c in channels
            ]
        )
        self.semantic_generators = nn.ModuleList(
            [SharedSemanticGenerator(int(c), config.hidden_dim) for c in channels]
        )
        self.msda = nn.ModuleList(
            [
                CalibratedMSDALevel(config.hidden_dim, 3, config.sampling_points, config)
                for _ in range(3)
            ]
        )
        self.class_corrections = nn.ModuleList(
            [nn.Conv2d(config.hidden_dim, num_classes, 1) for _ in range(3)]
        )
        for layer in self.class_corrections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], dict]:
        projected = [layer(feature) for layer, feature in zip(self.input_projections, features)]
        semantic_values: list[torch.Tensor] = []
        semantic_logits: list[torch.Tensor] = []
        if self.config.uses_semantics:
            for generator, feature in zip(self.semantic_generators, features):
                semantic, logit = generator(feature)
                semantic_values.append(semantic)
                semantic_logits.append(logit)
        else:
            semantic_values = [torch.zeros_like(value) for value in projected]

        corrections, diagnostics = [], {}
        for index, (module, query) in enumerate(zip(self.msda, projected)):
            semantic_query = semantic_values[index] if self.config.uses_semantics else None
            enhanced, diag = module(
                query,
                projected,
                semantic_query,
                semantic_values if self.config.uses_semantics else None,
            )
            corrections.append(float(self.config.correction_scale) * self.class_corrections[index](enhanced))
            diagnostics[f"level{index}"] = diag
        return corrections, semantic_logits, diagnostics


class SSCBDetectHead(nn.Module):
    """Native YOLO26 boxes with DSRDet-inspired classification correction."""

    def __init__(self, base_head: nn.Module, config: SSCBConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("SSCB transfer memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.base_head = base_head
        self.config = config
        self.sscb = SSCBClassificationPath(channels, int(base_head.nc), config)
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

    def _native_branch(self, features: list[torch.Tensor], branch: dict[str, nn.Module]) -> dict[str, Any]:
        """Run a native YOLO26 branch without SSCB correction.

        The one-to-one branch is intentionally kept native. SSCB is a residual
        classification transfer applied only to the one-to-many training path;
        computing SSCB again for one-to-one doubled the expensive semantic/MSDA
        path and could exhaust GPU memory before the first S0 batch completed.
        """
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](features[index]))
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat([v.view(batch, 4 * self.reg_max, -1) for v in boxes], dim=-1),
            "scores": torch.cat([v.view(batch, self.nc, -1) for v in logits], dim=-1),
            "feats": features,
        }

    def _forward_sscb_branch(self, features: list[torch.Tensor], branch: dict[str, nn.Module], *, include_semantic: bool):
        native = self._native_branch(features, branch)
        corrections, semantic_logits, diagnostics = self.sscb(features)
        batch = features[0].shape[0]
        split_logits = []
        start = 0
        for feature in features:
            count = int(feature.shape[-2] * feature.shape[-1])
            split_logits.append(native["scores"][:, :, start : start + count].view(batch, self.nc, feature.shape[-2], feature.shape[-1]))
            start += count
        native["scores"] = torch.cat(
            [(logit + correction).view(batch, self.nc, -1) for logit, correction in zip(split_logits, corrections)],
            dim=-1,
        )
        if include_semantic and semantic_logits:
            native["sscb_semantic_logits"] = semantic_logits
        self.last_sscb_diagnostics = diagnostics
        return native

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            one2many = self._forward_sscb_branch(features, self.one2many, include_semantic=True)
            one2one = self._native_branch([v.detach() for v in features], self.one2one)
            return {"one2many": one2many, "one2one": one2one}
        # Ultralytics validation runs the model in eval mode but still computes
        # the one-to-many training loss from the returned prediction dictionary.
        # Preserve semantic logits here whenever semantics are enabled; inference
        # itself continues to use the native one-to-one path below.
        one2many = (
            self._forward_sscb_branch(
                features,
                self.one2many,
                include_semantic=self.config.uses_semantics,
            )
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._native_branch([v.detach() for v in features], self.one2one)
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_sscb(model: nn.Module, config: SSCBConfig | dict[str, Any] | None) -> int:
    frozen = SSCBConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], SSCBDetectHead):
        return 0
    detector[-1] = SSCBDetectHead(detector[-1], frozen)
    return 1


def load_sscb_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if source_model is None:
        return {"transferred": 0, "total": len(target.state_dict())}
    source_state = source_model.float().state_dict()
    target_state = target.state_dict()
    transferred = 0
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key].copy_(value)
            transferred += 1
    target.load_state_dict(target_state, strict=False)
    return {"transferred": transferred, "total": len(target_state)}
