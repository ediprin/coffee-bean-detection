from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class FTIFConfig:
    prompt_mode: str = "base_specific"  # specific | base_specific
    bidirectional_alignment: bool = False
    temperature: float = 0.07
    alignment_weight: float = 1.0
    ffn_ratio: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "FTIFConfig | dict[str, Any] | None") -> "FTIFConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.prompt_mode not in {"specific", "base_specific"}:
            raise ValueError("prompt_mode harus specific atau base_specific")
        if result.temperature <= 0:
            raise ValueError("temperature harus positif")
        if result.alignment_weight < 0 or result.ffn_ratio <= 0:
            raise ValueError("alignment_weight/ffn_ratio tidak valid")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class FTIFLevelIntegrator(nn.Module):
    """LFDet Eq. (15)-(18) transferred to one YOLO pyramid level.

    The paper projects frozen text embeddings to the visual dimension, uses
    visual tokens as queries and text tokens as keys/values, then applies a
    residual cross-attention update and a projection-GELU-projection residual.

    For controlled transfer to an already strong YOLO26 detector, the paper
    representation drives a *zero-initialized residual leaf-logit correction*
    rather than replacing the pretrained native classifier at step zero. This
    identity-start stabilization is an explicit transfer choice, not an LFDet
    paper claim.
    """

    def __init__(
        self,
        channels: int,
        text_dim: int,
        num_classes: int,
        config: FTIFConfig,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.config = config
        hidden = max(8, int(round(self.channels * config.ffn_ratio)))
        self.text_projection = nn.Linear(int(text_dim), self.channels, bias=False)
        self.query_projection = nn.Linear(self.channels, self.channels, bias=False)
        self.key_projection = nn.Linear(self.channels, self.channels, bias=False)
        self.value_projection = nn.Linear(self.channels, self.channels, bias=False)
        self.ffn_in = nn.Linear(self.channels, hidden)
        self.ffn_out = nn.Linear(hidden, self.channels)
        self.logit_correction = nn.Conv2d(self.channels, self.num_classes, 1, bias=True)
        nn.init.zeros_(self.logit_correction.weight)
        nn.init.zeros_(self.logit_correction.bias)

    def forward(
        self,
        feature: torch.Tensor,
        frozen_text_embeddings: torch.Tensor,
        *,
        return_similarity: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, channels, height, width = feature.shape
        if channels != self.channels:
            raise ValueError(f"FTIF channel mismatch: {channels} != {self.channels}")
        visual = feature.flatten(2).transpose(1, 2)  # [B, HW, C]
        text = self.text_projection(
            frozen_text_embeddings.to(device=feature.device, dtype=self.text_projection.weight.dtype)
        ).to(dtype=visual.dtype)  # [K, C]

        q = self.query_projection(visual)
        k = self.key_projection(text)
        v = self.value_projection(text)
        attention = torch.softmax(
            torch.matmul(q, k.transpose(0, 1)) / math.sqrt(float(self.channels)),
            dim=-1,
        )
        cross = torch.matmul(attention, v)
        image_mca = visual + cross
        transformed = image_mca + self.ffn_out(F.gelu(self.ffn_in(image_mca)))
        transformed_map = transformed.transpose(1, 2).reshape(batch, channels, height, width)
        correction = self.logit_correction(transformed_map)

        similarity = None
        if return_similarity:
            # LFDet Eq. (19) explicitly aligns pre-interaction visual I_e with
            # projected text T*_e using cosine similarity / tau.
            visual_norm = F.normalize(visual.float(), dim=-1, eps=1e-8)
            text_norm = F.normalize(text.float(), dim=-1, eps=1e-8)
            similarity = (
                torch.matmul(visual_norm, text_norm.transpose(0, 1))
                / float(self.config.temperature)
            ).to(dtype=visual.dtype)
        return correction, similarity


class FTIFDetectHead(nn.Module):
    """YOLO26 Detect wrapper with FTIF on classification while boxes stay native."""

    def __init__(
        self,
        base_head: nn.Module,
        config: FTIFConfig,
        text_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("FTIF transfer memerlukan native YOLO26 end-to-end Detect")
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != int(base_head.nc):
            raise ValueError("FTIF text embeddings harus [nc,text_dim]")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("FTIF transfer dikunci untuk P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.register_buffer("text_embeddings", text_embeddings.float().clone(), persistent=True)
        self.integrators = nn.ModuleList(
            [
                FTIFLevelIntegrator(
                    channel,
                    int(text_embeddings.shape[1]),
                    int(base_head.nc),
                    config,
                )
                for channel in channels
            ]
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
        self,
        features: list[torch.Tensor],
        branch: dict[str, nn.Module],
        *,
        include_alignment: bool,
    ) -> dict[str, torch.Tensor]:
        boxes, logits, similarities = [], [], []
        for index in range(self.nl):
            feature = features[index]
            boxes.append(branch["box_head"][index](feature))
            native_logits = branch["cls_head"][index](feature)
            correction, similarity = self.integrators[index](
                feature,
                self.text_embeddings,
                return_similarity=include_alignment,
            )
            logits.append(native_logits + correction)
            if similarity is not None:
                similarities.append(similarity)
        batch_size = features[0].shape[0]
        output = {
            "boxes": torch.cat(
                [value.view(batch_size, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [value.view(batch_size, self.nc, -1) for value in logits], dim=-1
            ),
            "feats": features,
        }
        if similarities:
            output["ftif_similarity"] = torch.cat(similarities, dim=1)  # [B,N,K]
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            one2many = self._forward_branch(
                features,
                self.one2many,
                include_alignment=bool(self.config.bidirectional_alignment),
            )
            one2one = self._forward_branch(
                [value.detach() for value in features],
                self.one2one,
                include_alignment=False,
            )
            return {"one2many": one2many, "one2one": one2one}

        one2many = (
            # Ultralytics computes validation loss from the auxiliary
            # one-to-many predictions while the module is in eval mode.  FT3
            # therefore still needs Eq. (19) similarity here; decoded
            # inference continues to use one-to-one and is unchanged.
            self._forward_branch(
                features,
                self.one2many,
                include_alignment=bool(self.config.bidirectional_alignment),
            )
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [value.detach() for value in features], self.one2one, include_alignment=False
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_ftif(
    model: nn.Module,
    config: FTIFConfig | dict[str, Any] | None,
    text_embeddings: torch.Tensor,
) -> int:
    frozen = FTIFConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], FTIFDetectHead):
        return 0
    detector[-1] = FTIFDetectHead(detector[-1], frozen, text_embeddings)
    return 1


def load_ftif_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly map native D0 Detect state into FTIF wrapper namespace."""
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, FTIFDetectHead):
        raise TypeError("Target bukan FTIFDetectHead")
    if isinstance(source_head, FTIFDetectHead):
        # Resume checkpoints already contain the full FTIF state via model.load.
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect ke FTIF tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}
