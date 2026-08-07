from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from coffee_detector.sni21_ontology import SNI21_CLASSES, load_sni21_ontology


DEFAULT_TASKS = (
    "entity_family",
    "primary_condition",
    "hole_count",
    "integrity_fraction",
    "surface_extent",
)


@dataclass(frozen=True)
class SemanticAuxConfig:
    tasks: tuple[str, ...] = DEFAULT_TASKS
    auxiliary_weight: float = 0.20

    @classmethod
    def from_mapping(cls, payload: "SemanticAuxConfig | dict[str, Any] | None") -> "SemanticAuxConfig":
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
        if result.auxiliary_weight < 0:
            raise ValueError("auxiliary_weight tidak boleh negatif")
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


def semantic_task_spec(tasks: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    ontology = load_sni21_ontology()
    result: dict[str, dict[str, Any]] = {}
    for task in tasks:
        values = tuple(sorted({str(row[task]) for row in ontology["classes"].values() if row.get(task) is not None}))
        value_to_id = {value: i for i, value in enumerate(values)}
        mapping = [
            value_to_id[str(ontology["classes"][name][task])]
            if ontology["classes"][name].get(task) is not None else -1
            for name in SNI21_CLASSES
        ]
        result[task] = {"values": values, "mapping": mapping}
    return result


class SemanticAuxiliaryHeads(nn.Module):
    """Training-only semantic heads; native 21-leaf logits are never marginalized."""

    def __init__(self, channels: tuple[int, int, int], config: SemanticAuxConfig) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Semantic auxiliary memerlukan P3/P4/P5")
        self.config = config
        self.spec = semantic_task_spec(config.tasks)
        self.heads = nn.ModuleDict()
        for task in config.tasks:
            groups = len(self.spec[task]["values"])
            self.heads[task] = nn.ModuleList([nn.Conv2d(c, groups, 1) for c in channels])

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        if len(features) != 3:
            raise ValueError("Semantic auxiliary memerlukan tiga feature levels")
        output = {}
        for task, heads in self.heads.items():
            values = []
            for head, feature in zip(heads, features):
                logits = head(feature)
                b, g = logits.shape[:2]
                values.append(logits.view(b, g, -1).transpose(1, 2))
            output[task] = torch.cat(values, dim=1)
        return output


class SemanticAuxDetectHead(nn.Module):
    """Native YOLO26 Detect + training-only independent semantic supervision."""

    def __init__(self, base_head: nn.Module, config: SemanticAuxConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("Semantic auxiliary memerlukan native YOLO26 Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.base_head = base_head
        self.config = config
        self.semantic_aux = SemanticAuxiliaryHeads(channels, config)
        for name in ("i", "f", "type", "np", "nc", "nl", "reg_max", "stride", "end2end", "max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))

    @property
    def one2many(self):
        return self.base_head.one2many

    @property
    def one2one(self):
        return self.base_head.one2one

    def forward(self, features: list[torch.Tensor]):
        if not self.training:
            # Zero inference overhead and exactly native leaf prediction path.
            return self.base_head(features)
        native = self.base_head(features)
        if not isinstance(native, dict) or "one2many" not in native:
            raise TypeError("Native YOLO26 training output tidak sesuai")
        native["one2many"]["semantic_aux_logits"] = self.semantic_aux(features)
        return native

    def fuse(self) -> None:
        self.base_head.fuse()


def load_semantic_aux_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, SemanticAuxDetectHead):
        raise TypeError("Target bukan SemanticAuxDetectHead")
    if isinstance(source_head, SemanticAuxDetectHead):
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class SemanticAuxDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, semantic_aux=None):
        self.semantic_aux_config = SemanticAuxConfig.from_mapping(semantic_aux)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if int(self.model[-1].nc) != len(SNI21_CLASSES):
            raise ValueError("Semantic auxiliary dikunci untuk SNI-21")
        self.model[-1] = SemanticAuxDetectHead(self.model[-1], self.semantic_aux_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss
        from .loss import SemanticAuxDetectionLoss
        return E2ELoss(self, loss_fn=SemanticAuxDetectionLoss)
