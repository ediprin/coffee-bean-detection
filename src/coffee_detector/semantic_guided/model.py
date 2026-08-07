from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from coffee_detector.semantic_aux.model import DEFAULT_TASKS, semantic_task_spec
from coffee_detector.sni21_ontology import SNI21_CLASSES


@dataclass(frozen=True)
class SemanticGuidedConfig:
    tasks: tuple[str, ...] = DEFAULT_TASKS
    auxiliary_weight: float = 0.20
    correction_scale: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "SemanticGuidedConfig | dict[str, Any] | None") -> "SemanticGuidedConfig":
        if isinstance(payload, cls):
            return payload
        raw = dict(payload or {})
        if "tasks" in raw:
            raw["tasks"] = tuple(str(v) for v in raw["tasks"])
        result = cls(**raw)
        if not result.tasks or len(set(result.tasks)) != len(result.tasks):
            raise ValueError("semantic tasks harus unik dan tidak kosong")
        unknown = sorted(set(result.tasks) - set(DEFAULT_TASKS))
        if unknown:
            raise ValueError(f"semantic task tidak diizinkan: {unknown}")
        if result.auxiliary_weight < 0 or result.correction_scale <= 0:
            raise ValueError("bobot semantic guidance tidak valid")
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        return payload


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class SemanticGuidedLeafCorrection(nn.Module):
    """Coffee-specific SSGB transfer: semantics are predicted then re-fed to leaf logits.

    DSRDet SSGB predicts an upper-level semantic logit from the classification-
    specialized feature, concatenates it back with that feature, and predicts the
    fine-grained category (Eqs. 20-22). Coffee SNI-21 has several observable
    semantic factors rather than one aircraft meta-category, so this transfer uses
    independent ontology heads and concatenates all of their logits with the dense
    P3/P4/P5 input feature before a 21-leaf residual correction.

    The final correction convolutions are zero initialized: before learning, all
    native YOLO26 leaf scores and boxes are exactly unchanged.
    """

    def __init__(self, channels: tuple[int, int, int], num_classes: int, config: SemanticGuidedConfig) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Semantic guidance memerlukan tepat P3/P4/P5")
        self.config = config
        self.num_classes = int(num_classes)
        self.spec = semantic_task_spec(config.tasks)
        self.semantic_heads = nn.ModuleDict()
        total_semantic_channels = 0
        for task in config.tasks:
            groups = len(self.spec[task]["values"])
            total_semantic_channels += groups
            self.semantic_heads[task] = nn.ModuleList(
                [nn.Conv2d(int(channel), groups, 1) for channel in channels]
            )
        self.leaf_corrections = nn.ModuleList(
            [
                nn.Conv2d(int(channel) + total_semantic_channels, self.num_classes, 1)
                for channel in channels
            ]
        )
        for layer in self.leaf_corrections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        if len(features) != 3:
            raise ValueError("Semantic guidance memerlukan tiga feature levels")
        semantic_by_level: list[list[torch.Tensor]] = [[] for _ in features]
        flattened: dict[str, torch.Tensor] = {}
        for task, heads in self.semantic_heads.items():
            dense = []
            for level, (head, feature) in enumerate(zip(heads, features)):
                logits = head(feature)
                semantic_by_level[level].append(logits)
                batch, groups = logits.shape[:2]
                dense.append(logits.view(batch, groups, -1).transpose(1, 2))
            flattened[task] = torch.cat(dense, dim=1)
        corrections = []
        for level, feature in enumerate(features):
            guided = torch.cat((feature, *semantic_by_level[level]), dim=1)
            corrections.append(float(self.config.correction_scale) * self.leaf_corrections[level](guided))
        return corrections, flattened


class SemanticGuidedDetectHead(nn.Module):
    def __init__(self, base_head: nn.Module, config: SemanticGuidedConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("Semantic guidance memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("Semantic guidance memerlukan P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.guidance = SemanticGuidedLeafCorrection(channels, int(base_head.nc), config)
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
        include_semantic_aux: bool,
    ) -> dict[str, torch.Tensor]:
        boxes, native_logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            native_logits.append(branch["cls_head"][index](features[index]))
        corrections, semantics = self.guidance(features)
        batch_size = features[0].shape[0]
        result = {
            "boxes": torch.cat(
                [value.view(batch_size, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [
                    (native + correction).view(batch_size, self.nc, -1)
                    for native, correction in zip(native_logits, corrections)
                ],
                dim=-1,
            ),
            "feats": features,
        }
        if include_semantic_aux:
            result["semantic_aux_logits"] = semantics
        return result

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            one2many = self._forward_branch(features, self.one2many, include_semantic_aux=True)
            one2one = self._forward_branch(
                [value.detach() for value in features], self.one2one, include_semantic_aux=False
            )
            return {"one2many": one2many, "one2one": one2one}

        one2many = (
            self._forward_branch(features, self.one2many, include_semantic_aux=False)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [value.detach() for value in features], self.one2one, include_semantic_aux=False
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_semantic_guided_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, SemanticGuidedDetectHead):
        raise TypeError("Target bukan SemanticGuidedDetectHead")
    if isinstance(source_head, SemanticGuidedDetectHead):
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


class SemanticGuidedDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, semantic_guided=None):
        self.semantic_guided_config = SemanticGuidedConfig.from_mapping(semantic_guided)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if int(self.model[-1].nc) != len(SNI21_CLASSES):
            raise ValueError("Semantic guidance dikunci untuk SNI-21")
        self.model[-1] = SemanticGuidedDetectHead(self.model[-1], self.semantic_guided_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss
        from .loss import SemanticGuidedDetectionLoss
        return E2ELoss(self, loss_fn=SemanticGuidedDetectionLoss)
