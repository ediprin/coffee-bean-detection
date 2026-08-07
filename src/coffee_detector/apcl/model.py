from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class APCLConfig:
    embedding_dim: int = 128
    ema_eta: float = 0.4
    loss_weight: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "APCLConfig | dict[str, Any] | None") -> "APCLConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.embedding_dim <= 0:
            raise ValueError("embedding_dim harus positif")
        if not 0.0 < result.ema_eta <= 1.0:
            raise ValueError("ema_eta harus berada pada (0,1]")
        if result.loss_weight < 0:
            raise ValueError("loss_weight tidak boleh negatif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class AdaptivePrototypeContrast(nn.Module):
    """Training-only APCL state following Li et al. TGRS 2025 Eqs. (9)-(13).

    Prototypes are not learnable parameters. For every class represented in the
    current batch, the batch mean embedding is computed and the persistent
    prototype is updated using EMA. The loss penalizes positive cosine
    similarity between each instance and prototypes of *other* classes. It does
    not explicitly attract an instance toward its own prototype, matching the
    distinction made by the paper.
    """

    def __init__(self, num_classes: int, embedding_dim: int, eta: float) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.eta = float(eta)
        self.register_buffer(
            "prototypes",
            torch.zeros(self.num_classes, self.embedding_dim),
            persistent=True,
        )
        self.register_buffer(
            "prototype_seen",
            torch.zeros(self.num_classes, dtype=torch.bool),
            persistent=True,
        )

    @torch.no_grad()
    def _update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        for class_id in labels.unique().tolist():
            class_id = int(class_id)
            members = embeddings[labels == class_id]
            if not len(members):
                continue
            batch_prototype = members.mean(dim=0)
            if not bool(self.prototype_seen[class_id]):
                self.prototypes[class_id].copy_(batch_prototype)
                self.prototype_seen[class_id] = True
            else:
                updated = (
                    (1.0 - self.eta) * self.prototypes[class_id]
                    + self.eta * batch_prototype
                )
                self.prototypes[class_id].copy_(updated)

    def update_and_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embeddings harus [N,{self.embedding_dim}], diterima {tuple(embeddings.shape)}"
            )
        labels = labels.to(device=embeddings.device, dtype=torch.long).reshape(-1)
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError("Jumlah embedding dan label tidak sama")
        if not len(labels):
            return embeddings.sum() * 0.0
        if int(labels.min()) < 0 or int(labels.max()) >= self.num_classes:
            raise ValueError("Label APCL di luar rentang kelas")

        # Prototype updates are state updates, not a gradient path.
        self._update_prototypes(embeddings.detach(), labels)
        z = F.normalize(embeddings, dim=1, eps=1e-8)
        p = F.normalize(self.prototypes.detach(), dim=1, eps=1e-8)
        similarities = z @ p.t()
        own_mask = F.one_hot(labels, num_classes=self.num_classes).bool()
        wrong_similarity = similarities.masked_fill(own_mask, 0.0)
        # Eq. (12): average relu(cosine) over C x N; unseen zero prototypes
        # naturally contribute zero until their first observation.
        return F.relu(wrong_similarity).sum() / float(self.num_classes * len(labels))


class APCLProjectionHead(nn.Module):
    """Project dense P3/P4/P5 classification features into one embedding space."""

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
        config: APCLConfig,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("APCL memerlukan tepat P3/P4/P5")
        self.config = config
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(channel), config.embedding_dim, 1, bias=False),
                    nn.BatchNorm2d(config.embedding_dim),
                    nn.SiLU(inplace=True),
                )
                for channel in channels
            ]
        )
        self.prototype_contrast = AdaptivePrototypeContrast(
            num_classes, config.embedding_dim, config.ema_eta
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError("APCL memerlukan tepat P3/P4/P5")
        embeddings = []
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            batch = value.shape[0]
            value = value.view(batch, self.config.embedding_dim, -1).transpose(1, 2)
            embeddings.append(value)
        return torch.cat(embeddings, dim=1)


class APCLDetectHead(nn.Module):
    """Native YOLO26 Detect with training-only contrastive embeddings."""

    def __init__(self, base_head: nn.Module, config: APCLConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("APCL memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("APCL dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("APCL memerlukan tiga level P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.apcl = APCLProjectionHead(channels, int(base_head.nc), config)
        for name in ("i", "f", "type", "np"):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))
        for name in (
            "nc",
            "nl",
            "reg_max",
            "stride",
            "end2end",
            "max_det",
            "export",
            "format",
            "dynamic",
            "agnostic_nms",
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
        include_apcl: bool,
    ) -> dict[str, torch.Tensor]:
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](features[index]))
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
        if include_apcl:
            output["apcl_embeddings"] = self.apcl(features)
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            # APCL is attached to the one-to-many training assignments only.
            # One-to-one remains the native end-to-end companion loss.
            one2many = self._forward_branch(features, self.one2many, include_apcl=True)
            one2one = self._forward_branch(
                [value.detach() for value in features], self.one2one, include_apcl=False
            )
            return {"one2many": one2many, "one2one": one2one}
        # No projection head is executed at inference: zero APCL inference cost.
        return self.base_head(features)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_apcl(model: nn.Module, config: APCLConfig | dict[str, Any] | None) -> int:
    frozen = APCLConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], APCLDetectHead):
        return 0
    detector[-1] = APCLDetectHead(detector[-1], frozen)
    return 1


def load_apcl_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly map a native D0 Detect state into the APCL wrapper namespace."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, APCLDetectHead):
        raise TypeError("Target bukan APCLDetectHead")
    if isinstance(source_head, APCLDetectHead):
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
