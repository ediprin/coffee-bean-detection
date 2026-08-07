from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PCLConfig:
    embedding_dim: int = 128
    temperature: float = 1.0 / 32.0
    loss_weight: float = 1.0
    prototype_init_std: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "PCLConfig | dict[str, Any] | None") -> "PCLConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.embedding_dim <= 0:
            raise ValueError("embedding_dim harus positif")
        if result.temperature <= 0:
            raise ValueError("temperature harus positif")
        if result.loss_weight < 0:
            raise ValueError("loss_weight tidak boleh negatif")
        if result.prototype_init_std <= 0:
            raise ValueError("prototype_init_std harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class LearnedPrototypeContrast(nn.Module):
    """PCLDet learned prototype bank and ProtoCL loss (Ouyang et al., Eq. 1/3/4).

    This module keeps one learnable prototype per fine-grained class. Prototypes
    are ordinary ``nn.Parameter`` objects and are therefore updated by the same
    optimizer/SGD path as the detector, matching the paper's Eq. (4) principle.

    The original paper states that prototypes are initialized from a normal
    distribution but does not specify its standard deviation in the method
    text. ``prototype_init_std`` is therefore an explicit implementation choice,
    not claimed as a paper-specified hyperparameter. Cosine normalization makes
    the initial direction, rather than magnitude, the primary quantity.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        temperature: float,
        prototype_init_std: float,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(
            torch.empty(self.num_classes, self.embedding_dim)
        )
        nn.init.normal_(self.prototypes, mean=0.0, std=float(prototype_init_std))

    @staticmethod
    def _log1p_sum_exp(values: torch.Tensor) -> torch.Tensor:
        """Stable ``log(1 + sum(exp(values)))``."""
        zero = values.new_zeros((1,))
        return torch.logsumexp(torch.cat((zero, values.reshape(-1))), dim=0)

    def loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
            raise ValueError("Label PCL di luar rentang kelas")

        # Eq. (1): cosine similarity d(i,k).
        z = F.normalize(embeddings, dim=1, eps=1e-8)
        p = F.normalize(self.prototypes, dim=1, eps=1e-8)
        similarities = z @ p.t()  # [N, K]
        scaled = similarities / self.temperature

        represented = labels.unique(sorted=True)
        align_terms = []
        # Eq. (3), first term: pull samples toward own prototype.
        for class_id in represented.tolist():
            class_id = int(class_id)
            members = scaled[labels == class_id, class_id]
            align_terms.append(self._log1p_sum_exp(-members))
        align = torch.stack(align_terms).mean()

        # Eq. (3), second term: push samples away from all non-own prototypes.
        away_terms = []
        for class_id in range(self.num_classes):
            non_members = scaled[labels != class_id, class_id]
            away_terms.append(self._log1p_sum_exp(non_members))
        away = torch.stack(away_terms).mean()
        return align + away


class PCLProjectionHead(nn.Module):
    """Dense P3/P4/P5 projection used to adapt PCLDet to YOLO26 positives."""

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
        config: PCLConfig,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("PCL memerlukan tepat P3/P4/P5")
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
        self.prototype_contrast = LearnedPrototypeContrast(
            num_classes,
            config.embedding_dim,
            config.temperature,
            config.prototype_init_std,
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != 3:
            raise ValueError("PCL memerlukan tepat P3/P4/P5")
        embeddings = []
        for projection, feature in zip(self.projections, features):
            value = projection(feature)
            batch = value.shape[0]
            value = value.view(batch, self.config.embedding_dim, -1).transpose(1, 2)
            embeddings.append(value)
        return torch.cat(embeddings, dim=1)


class PCLDetectHead(nn.Module):
    """Native YOLO26 Detect plus training-only learned-prototype embeddings."""

    def __init__(self, base_head: nn.Module, config: PCLConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("PCL memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("PCL dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("PCL memerlukan tiga level P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.pcl = PCLProjectionHead(channels, int(base_head.nc), config)
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
        include_pcl: bool,
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
        if include_pcl:
            output["pcl_embeddings"] = self.pcl(features)
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            one2many = self._forward_branch(features, self.one2many, include_pcl=True)
            one2one = self._forward_branch(
                [value.detach() for value in features], self.one2one, include_pcl=False
            )
            return {"one2many": one2many, "one2one": one2one}
        # PCL projection/prototype branch is not executed at inference.
        return self.base_head(features)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_pcl(model: nn.Module, config: PCLConfig | dict[str, Any] | None) -> int:
    frozen = PCLConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], PCLDetectHead):
        return 0
    detector[-1] = PCLDetectHead(detector[-1], frozen)
    return 1


def load_pcl_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly map native D0 Detect state into the PCL wrapper namespace."""
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, PCLDetectHead):
        raise TypeError("Target bukan PCLDetectHead")
    if isinstance(source_head, PCLDetectHead):
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
