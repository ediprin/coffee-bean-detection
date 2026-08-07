from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class IGEMConfig:
    """Frozen YOLO26 transfer settings for class-aware IGEM.

    Paper/patent-specified values:
      - reference branch depth = 3 x 3x3 convolutions;
      - mask head = 1x1 convolution over N+1 classes;
      - mask loss weight = 0.05;
      - IGEM uses grouped kxk static context, local multi-head attention,
        and soft channel fusion.

    The accessible source does not expose numerical defaults for k, head count,
    or channel reduction, so those three values are explicit transfer choices.
    """

    reference_depth: int = 3
    mask_loss_weight: float = 0.05
    kernel_size: int = 3
    attention_heads: int = 4
    channel_reduction: int = 4
    correction_scale: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "IGEMConfig | Mapping[str, Any] | None") -> "IGEMConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.reference_depth != 3:
            raise ValueError("IGEM transfer dikunci ke 3 reference convolutions")
        if result.mask_loss_weight != 0.05:
            raise ValueError("IGEM mask loss weight dikunci ke 0.05")
        if result.kernel_size <= 0 or result.kernel_size % 2 == 0:
            raise ValueError("kernel_size harus ganjil dan positif")
        if result.attention_heads <= 0:
            raise ValueError("attention_heads harus positif")
        if result.channel_reduction <= 0:
            raise ValueError("channel_reduction harus positif")
        if result.correction_scale <= 0:
            raise ValueError("correction_scale harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d with running-stat fallback for a singleton final batch."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training and value.shape[0] == 1:
            return F.batch_norm(
                value,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(value)


class FeatureGuidedEnhancement(nn.Module):
    """IGEM feature-guided enhancement from patent steps 3-2-1..3-2-5.

    Q = F_ori, V = F_ori W_v, K = R.
    K_s is grouped local context from R. Attention A is generated from
    concatenated [K_s, Q], then used to aggregate kxk neighborhoods of V.
    Static K_s and dynamic K_enh are fused by two-way channel soft attention.

    The exact matrix rendering for the two channel-attention projection rows is
    unavailable in the accessible translated source. This transfer implements
    the stated two-weight Softmax with two learned linear projections of G_c.
    """

    def __init__(self, channels: int, config: IGEMConfig) -> None:
        super().__init__()
        channels = int(channels)
        heads = int(config.attention_heads)
        if channels % heads != 0:
            raise ValueError(f"channels={channels} harus habis dibagi heads={heads}")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.kernel_size = int(config.kernel_size)
        k2 = self.kernel_size * self.kernel_size
        padding = self.kernel_size // 2
        self.static_context = nn.Conv2d(
            channels,
            channels,
            self.kernel_size,
            padding=padding,
            groups=heads,
            bias=False,
        )
        self.value_projection = nn.Conv2d(channels, channels, 1, bias=False)
        attention_hidden = max(channels // int(config.channel_reduction), heads)
        self.attention_w1 = nn.Conv2d(2 * channels, attention_hidden, 1, bias=True)
        self.attention_w2 = nn.Conv2d(attention_hidden, heads * k2, 1, bias=True)

        fusion_hidden = max(channels // int(config.channel_reduction), 1)
        self.fusion_reduce = nn.Linear(channels, fusion_hidden, bias=False)
        self.fusion_bn = SafeBatchNorm1d(fusion_hidden)
        self.fusion_static = nn.Linear(fusion_hidden, channels, bias=True)
        self.fusion_dynamic = nn.Linear(fusion_hidden, channels, bias=True)

    def forward(self, original: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if original.shape != reference.shape:
            raise ValueError(
                f"F_ori dan R harus sama shape, diterima {tuple(original.shape)} vs {tuple(reference.shape)}"
            )
        batch, channels, height, width = original.shape
        if channels != self.channels:
            raise ValueError("Channel input tidak sesuai IGEM")
        k2 = self.kernel_size * self.kernel_size

        query = original
        value = self.value_projection(original)
        static = self.static_context(reference)

        attention = self.attention_w2(F.relu(self.attention_w1(torch.cat((static, query), dim=1))))
        attention = attention.view(batch, self.heads, k2, height * width)
        attention = torch.softmax(attention, dim=2)

        neighborhoods = F.unfold(
            value,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        neighborhoods = neighborhoods.view(
            batch,
            self.heads,
            self.head_dim,
            k2,
            height * width,
        )
        dynamic = (neighborhoods * attention.unsqueeze(2)).sum(dim=3)
        dynamic = dynamic.reshape(batch, channels, height, width)

        fused_sum = static + dynamic
        global_descriptor = fused_sum.mean(dim=(-2, -1))
        compact = F.relu(self.fusion_bn(self.fusion_reduce(global_descriptor)))
        branch_logits = torch.stack(
            (self.fusion_static(compact), self.fusion_dynamic(compact)),
            dim=1,
        )
        weights = torch.softmax(branch_logits, dim=1)
        static_weight = weights[:, 0].unsqueeze(-1).unsqueeze(-1)
        dynamic_weight = weights[:, 1].unsqueeze(-1).unsqueeze(-1)
        return static_weight * static + dynamic_weight * dynamic


class ClassAwareReferenceLevel(nn.Module):
    """Three-convolution class-aware reference branch + IGEM correction."""

    def __init__(self, channels: int, num_classes: int, config: IGEMConfig) -> None:
        super().__init__()
        blocks = []
        for _ in range(config.reference_depth):
            blocks.extend(
                [
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                ]
            )
        self.reference = nn.Sequential(*blocks)
        self.mask_classifier = nn.Conv2d(channels, num_classes + 1, 1)
        self.enhancement = FeatureGuidedEnhancement(channels, config)
        self.class_correction = nn.Conv2d(channels, num_classes, 1)
        # Exact native-D0 initialization contract: new enhancement starts with
        # zero contribution to fine logits even though auxiliary mask supervision
        # is active from the first update.
        nn.init.zeros_(self.class_correction.weight)
        nn.init.zeros_(self.class_correction.bias)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.reference(feature)
        mask_logits = self.mask_classifier(reference)
        enhanced = self.enhancement(feature, reference)
        correction = self.class_correction(enhanced)
        return correction, mask_logits


class IGEMDetectHead(nn.Module):
    """Native YOLO26 boxes + IGEM classification-only residual corrections."""

    def __init__(self, base_head: nn.Module, config: IGEMConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("IGEM memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("IGEM dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("IGEM memerlukan P3/P4/P5")
        if any(channel % config.attention_heads for channel in channels):
            raise ValueError(
                f"Semua channel {channels} harus habis dibagi heads={config.attention_heads}"
            )
        self.base_head = base_head
        self.config = config
        self.levels = nn.ModuleList(
            [ClassAwareReferenceLevel(channel, int(base_head.nc), config) for channel in channels]
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

    def _forward_branch(
        self,
        features: list[torch.Tensor],
        branch: dict[str, nn.Module],
        *,
        expose_masks: bool,
    ) -> dict[str, Any]:
        boxes, scores, masks = [], [], []
        for index in range(self.nl):
            native_box = branch["box_head"][index](features[index])
            native_score = branch["cls_head"][index](features[index])
            correction, mask_logits = self.levels[index](features[index])
            boxes.append(native_box)
            scores.append(native_score + float(self.config.correction_scale) * correction)
            masks.append(mask_logits)
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
        if expose_masks:
            output["igem_mask_logits"] = masks
        return output

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many, expose_masks=True),
                "one2one": self._forward_branch(
                    [value.detach() for value in features], self.one2one, expose_masks=False
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


def load_igem_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strict native D0 head transfer; new reference/IGEM paths remain new."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, IGEMDetectHead):
        raise TypeError("Target bukan IGEMDetectHead")
    if isinstance(source_head, IGEMDetectHead):
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
