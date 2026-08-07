from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CGFIConfig:
    hidden_ratio: float = 0.25
    fusion_raw_weight: float = 0.5
    fusion_frequency_weight: float = 0.5

    @classmethod
    def from_mapping(cls, payload: "CGFIConfig | dict[str, Any] | None") -> "CGFIConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.hidden_ratio <= 0:
            raise ValueError("hidden_ratio harus positif")
        if result.fusion_raw_weight < 0 or result.fusion_frequency_weight < 0:
            raise ValueError("fusion weights tidak boleh negatif")
        if abs(result.fusion_raw_weight + result.fusion_frequency_weight - 1.0) > 1e-6:
            raise ValueError("fusion weights harus berjumlah 1 agar identity-init terjaga")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


def _init_scaled_identity(conv: nn.Conv2d, scale: float) -> None:
    if conv.kernel_size != (1, 1) or conv.in_channels != conv.out_channels:
        raise ValueError("identity init memerlukan 1x1 C->C")
    with torch.no_grad():
        conv.weight.zero_()
        diagonal = torch.arange(conv.in_channels)
        conv.weight[diagonal, diagonal, 0, 0] = float(scale)
        if conv.bias is not None:
            conv.bias.zero_()


class ContentAwareFrequencyFilter(nn.Module):
    """CGFI-style content-aware global frequency filtering.

    LFDet Eq. (14) applies a learned transformation T to the frequency
    representation, multiplies the resulting filter with the original Fourier
    coefficients, then applies inverse DFT. The paper states T is a stack of
    linear projection layers but does not specify the hidden width in the text
    available to this implementation. Here, 1x1 convolutions act as per-frequency
    linear projections over real/imaginary channel pairs; hidden_ratio is an
    explicit transfer choice, not a paper hyperparameter claim.
    """

    def __init__(self, channels: int, hidden_ratio: float) -> None:
        super().__init__()
        self.channels = int(channels)
        hidden = max(8, int(round(2 * channels * hidden_ratio)))
        self.filter_net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * channels, 1),
        )
        # Dynamic filter starts as 1+0j, so iFFT(F(x)*filter) == x.
        last = self.filter_net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[:channels].fill_(1.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        frequency = torch.fft.rfft2(value, dim=(-2, -1), norm="ortho")
        packed = torch.cat((frequency.real, frequency.imag), dim=1)
        parameters = self.filter_net(packed)
        real_filter, imag_filter = parameters.split(self.channels, dim=1)
        dynamic_filter = torch.complex(real_filter, imag_filter)
        enhanced_frequency = frequency * dynamic_filter
        return torch.fft.irfft2(
            enhanced_frequency,
            s=value.shape[-2:],
            dim=(-2, -1),
            norm="ortho",
        )


class CGFIFeatureEnhancer(nn.Module):
    """Independent raw/recovered projection followed by sum fusion (LFDet structure-1)."""

    def __init__(self, channels: int, config: CGFIConfig) -> None:
        super().__init__()
        self.filter = ContentAwareFrequencyFilter(channels, config.hidden_ratio)
        self.raw_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.frequency_projection = nn.Conv2d(channels, channels, 1, bias=False)
        _init_scaled_identity(self.raw_projection, config.fusion_raw_weight)
        _init_scaled_identity(self.frequency_projection, config.fusion_frequency_weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        recovered = self.filter(value)
        return self.raw_projection(value) + self.frequency_projection(recovered)


class CGFIDetectHead(nn.Module):
    """YOLO26 Detect wrapper: CGFI affects classification features only; boxes stay native."""

    def __init__(self, base_head: nn.Module, config: CGFIConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("CGFI transfer memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("CGFI transfer memerlukan P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.enhancers = nn.ModuleList([CGFIFeatureEnhancer(c, config) for c in channels])
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

    def _forward_branch(self, features: list[torch.Tensor], branch: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
        enhanced = [module(feature) for module, feature in zip(self.enhancers, features)]
        boxes, logits = [], []
        for index in range(self.nl):
            # Localization uses the untouched native P3/P4/P5 feature.
            boxes.append(branch["box_head"][index](features[index]))
            # Classification uses the CGFI-style frequency-enriched feature.
            logits.append(branch["cls_head"][index](enhanced[index]))
        batch_size = features[0].shape[0]
        return {
            "boxes": torch.cat(
                [value.view(batch_size, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [value.view(batch_size, self.nc, -1) for value in logits], dim=-1
            ),
            "feats": features,
        }

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many),
                "one2one": self._forward_branch([value.detach() for value in features], self.one2one),
            }
        one2many = self._forward_branch(features, self.one2many) if self._has_heads(self.one2many) else None
        one2one = self._forward_branch([value.detach() for value in features], self.one2one)
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_cgfi_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, CGFIDetectHead):
        raise TypeError("Target bukan CGFIDetectHead")
    if isinstance(source_head, CGFIDetectHead):
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
    DetectionModel = nn.Module  # type: ignore


class CGFIDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, cgfi=None):
        self.cgfi_config = CGFIConfig.from_mapping(cgfi)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = CGFIDetectHead(self.model[-1], self.cgfi_config)
