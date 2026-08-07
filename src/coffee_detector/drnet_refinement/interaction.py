from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch import nn

from .model import DRNetFineGrainedBranch, DRNetRefinementConfig, _first_conv_channels


@dataclass(frozen=True)
class DRNetInteractionConfig:
    """YOLO26 transfer of DRNet coarse-to-fine Interaction Verification.

    The official DRNet implementation predicts a coarse class first and then
    removes fine-class scores outside that coarse class' subclass set before
    NMS. Here the same verification rule is applied to dense YOLO26 one-to-one
    classification logits. Coarse classes come only from the frozen SNI
    ontology; no validation confusion information is used.
    """

    correction_scale: float = 1.0
    coarse_loss_weight: float = 1.0
    verification_floor: float = -80.0

    @classmethod
    def from_mapping(
        cls, payload: "DRNetInteractionConfig | Mapping[str, Any] | None"
    ) -> "DRNetInteractionConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.correction_scale <= 0:
            raise ValueError("correction_scale harus positif")
        if result.coarse_loss_weight < 0:
            raise ValueError("coarse_loss_weight tidak boleh negatif")
        if result.verification_floor >= 0:
            raise ValueError("verification_floor harus negatif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ordered_names(names: Mapping[int, str] | Sequence[str]) -> list[str]:
    if isinstance(names, Mapping):
        keys = sorted(int(key) for key in names)
        if keys != list(range(len(keys))):
            raise ValueError("Class ids harus kontigu mulai dari 0")
        return [str(names[key]) for key in keys]
    return [str(value) for value in names]


def build_entity_family_mapping(
    class_names: Mapping[int, str] | Sequence[str],
    ontology_path: str | Path,
) -> tuple[tuple[int, ...], tuple[str, ...], dict[str, list[str]]]:
    """Build coarse taxonomy strictly from frozen ontology ``entity_family``.

    Group order follows first occurrence in the dataset's declared class order,
    making the mapping deterministic without validation statistics.
    """

    names = _ordered_names(class_names)
    ontology_path = Path(ontology_path).expanduser().resolve()
    if not ontology_path.is_file():
        raise FileNotFoundError(f"Ontology tidak ditemukan: {ontology_path}")
    payload = yaml.safe_load(ontology_path.read_text(encoding="utf-8")) or {}
    classes = payload.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("Ontology tidak memiliki mapping classes")

    family_to_id: dict[str, int] = {}
    family_members: dict[str, list[str]] = {}
    mapping: list[int] = []
    for name in names:
        item = classes.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"Class dataset tidak ditemukan di ontology: {name}")
        family = item.get("entity_family")
        if not isinstance(family, str) or not family:
            raise ValueError(f"entity_family kosong untuk {name}")
        if family not in family_to_id:
            family_to_id[family] = len(family_to_id)
            family_members[family] = []
        mapping.append(family_to_id[family])
        family_members[family].append(name)

    if len(mapping) != len(names):
        raise RuntimeError("Ontology mapping tidak mencakup seluruh kelas")
    group_names = tuple(family_to_id.keys())
    if len(group_names) < 2:
        raise ValueError("Interaction Verification membutuhkan >=2 coarse groups")
    return tuple(mapping), group_names, family_members


class DRNetCoarseBranch(nn.Module):
    """Dense coarse classifier on the same P3/P4/P5 fields used by FGB."""

    def __init__(self, channels: tuple[int, int, int], num_groups: int) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Coarse branch memerlukan P3/P4/P5")
        if num_groups < 2:
            raise ValueError("num_groups minimal 2")
        self.classifiers = nn.ModuleList(
            [nn.Conv2d(int(channel), int(num_groups), 1) for channel in channels]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 3:
            raise ValueError("Coarse branch memerlukan tepat tiga feature levels")
        return [classifier(feature) for classifier, feature in zip(self.classifiers, features)]


def verify_fine_logits(
    fine_logits: torch.Tensor,
    coarse_logits: torch.Tensor,
    class_to_group: torch.Tensor,
    *,
    floor: float = -80.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DRNet official-code rule: coarse argmax restricts valid fine subclasses."""

    if fine_logits.ndim != 3 or coarse_logits.ndim != 3:
        raise ValueError("fine/coarse logits harus [B,C,N] dan [B,G,N]")
    if fine_logits.shape[0] != coarse_logits.shape[0] or fine_logits.shape[2] != coarse_logits.shape[2]:
        raise ValueError("fine/coarse logits tidak sejajar")
    class_to_group = class_to_group.to(device=fine_logits.device, dtype=torch.long).reshape(-1)
    if class_to_group.numel() != fine_logits.shape[1]:
        raise ValueError("class_to_group tidak cocok dengan jumlah fine class")
    if int(class_to_group.min()) < 0 or int(class_to_group.max()) >= coarse_logits.shape[1]:
        raise ValueError("class_to_group di luar rentang coarse class")
    coarse_prediction = coarse_logits.argmax(dim=1)
    allowed = class_to_group.view(1, -1, 1) == coarse_prediction.unsqueeze(1)
    verified = fine_logits.masked_fill(~allowed, float(floor))
    return verified, coarse_prediction


class DRNetInteractionDetectHead(nn.Module):
    """Dense YOLO26 adaptation of DRNet FGB + coarse/fine verification."""

    def __init__(
        self,
        base_head: nn.Module,
        config: DRNetInteractionConfig,
        class_to_group: Sequence[int],
        group_names: Sequence[str],
    ) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("Interaction Verification memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("Interaction Verification dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("Interaction Verification memerlukan P3/P4/P5")
        if len(class_to_group) != int(base_head.nc):
            raise ValueError("class_to_group harus mencakup seluruh fine class")
        if not group_names:
            raise ValueError("group_names kosong")

        self.base_head = base_head
        self.config = config
        self.group_names = tuple(str(value) for value in group_names)
        mapping = torch.tensor(tuple(int(value) for value in class_to_group), dtype=torch.long)
        if int(mapping.min()) < 0 or int(mapping.max()) >= len(self.group_names):
            raise ValueError("class_to_group tidak sesuai group_names")
        self.register_buffer("class_to_group", mapping, persistent=True)
        self.fine_grained = DRNetFineGrainedBranch(
            channels,
            int(base_head.nc),
            DRNetRefinementConfig(
                correction_scale=config.correction_scale,
                use_cml=False,
            ),
        )
        self.coarse = DRNetCoarseBranch(channels, len(self.group_names))

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
        expose_coarse_logits: bool,
        apply_verification: bool,
    ) -> dict[str, torch.Tensor]:
        boxes, native_logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            native_logits.append(branch["cls_head"][index](features[index]))
        fine_residual = self.fine_grained(features)
        coarse_fields = self.coarse(features)
        batch_size = features[0].shape[0]
        fine_scores = torch.cat(
            [
                (native + float(self.config.correction_scale) * fine).view(
                    batch_size, self.nc, -1
                )
                for native, fine in zip(native_logits, fine_residual)
            ],
            dim=-1,
        )
        coarse_scores = torch.cat(
            [value.view(batch_size, len(self.group_names), -1) for value in coarse_fields],
            dim=-1,
        )
        if apply_verification:
            fine_scores, coarse_prediction = verify_fine_logits(
                fine_scores,
                coarse_scores,
                self.class_to_group,
                floor=self.config.verification_floor,
            )
        else:
            coarse_prediction = coarse_scores.argmax(dim=1)
        output = {
            "boxes": torch.cat(
                [value.view(batch_size, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": fine_scores,
            "feats": features,
        }
        if expose_coarse_logits:
            output["dr_coarse_logits"] = coarse_scores.transpose(1, 2).contiguous()
        if apply_verification:
            output["dr_coarse_prediction"] = coarse_prediction
        return output

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            one2many = self._forward_branch(
                features,
                self.one2many,
                expose_coarse_logits=True,
                apply_verification=False,
            )
            one2one = self._forward_branch(
                [value.detach() for value in features],
                self.one2one,
                expose_coarse_logits=False,
                apply_verification=False,
            )
            return {"one2many": one2many, "one2one": one2one}

        one2many = (
            self._forward_branch(
                features,
                self.one2many,
                expose_coarse_logits=False,
                apply_verification=True,
            )
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [value.detach() for value in features],
            self.one2one,
            expose_coarse_logits=False,
            apply_verification=True,
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_drnet_interaction_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly transfer native D0 Detect weights; new FGB/coarse heads stay new."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, DRNetInteractionDetectHead):
        raise TypeError("Target bukan DRNetInteractionDetectHead")
    if isinstance(source_head, DRNetInteractionDetectHead):
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
    DetectionModel = nn.Module  # type: ignore[assignment,misc]


class DRNetInteractionDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        drnet_interaction: DRNetInteractionConfig | Mapping[str, Any] | None = None,
        class_to_group: Sequence[int] | None = None,
        group_names: Sequence[str] | None = None,
    ) -> None:
        self.drnet_interaction_config = DRNetInteractionConfig.from_mapping(drnet_interaction)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if class_to_group is None or group_names is None:
            raise ValueError("Interaction model memerlukan frozen class_to_group dan group_names")
        self.model[-1] = DRNetInteractionDetectHead(
            self.model[-1],
            self.drnet_interaction_config,
            class_to_group,
            group_names,
        )

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss
        from .interaction_loss import DRNetInteractionDetectionLoss

        return E2ELoss(self, loss_fn=DRNetInteractionDetectionLoss)
