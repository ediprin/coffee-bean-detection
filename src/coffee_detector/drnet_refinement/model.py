from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DRNetRefinementConfig:
    """One-stage transfer of DRNet's fine-grained branch to YOLO26.

    The paper applies dual refinement to 7x7 RoI features.  This experiment
    transfers only that operator to each native YOLO26 P3/P4/P5 classification
    field.  Localization remains native.  CML is an optional second ablation.
    """

    correction_scale: float = 1.0
    use_cml: bool = False
    cml_lambda1: float = 0.4
    cml_lambda2: float = 0.05
    cml_weight: float = 0.25

    @classmethod
    def from_mapping(
        cls, payload: "DRNetRefinementConfig | dict[str, Any] | None"
    ) -> "DRNetRefinementConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.correction_scale <= 0:
            raise ValueError("correction_scale harus positif")
        if not 0.0 <= result.cml_lambda1 <= 1.0:
            raise ValueError("cml_lambda1 harus berada pada [0,1]")
        if result.cml_lambda2 < 0:
            raise ValueError("cml_lambda2 tidak boleh negatif")
        if result.cml_weight < 0:
            raise ValueError("cml_weight tidak boleh negatif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


def _init_identity_1x1(layer: nn.Conv2d) -> None:
    if layer.kernel_size != (1, 1) or layer.in_channels != layer.out_channels:
        raise ValueError("Identity init hanya valid untuk Conv1x1 square")
    with torch.no_grad():
        layer.weight.zero_()
        diagonal = torch.arange(layer.in_channels)
        layer.weight[diagonal, diagonal, 0, 0] = 1.0
        if layer.bias is not None:
            layer.bias.zero_()


class DualRefinement(nn.Module):
    """DRNet Eqs. (1)-(4): spatial then channel refinement.

    For X[B,C,H,W]:
      U = Conv_1x1(X)
      A = Conv_1x1(X)              -> [B,1,H,W]
      V = A * U
      b = FC(GAP(V))               -> [B,C]
      Y = b * V

    DRNet does not state sigmoid activations in Eqs. (1)-(4), so none are added
    here.  The transfer is initialized as an identity refinement (A=b=1), while
    the downstream class correction is zero-initialized.  Thus the whole model
    begins exactly at the native D0 detector.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.feature = nn.Conv2d(channels, channels, 1, bias=True)
        self.spatial = nn.Conv2d(channels, 1, 1, bias=True)
        self.channel = nn.Linear(channels, channels, bias=True)
        _init_identity_1x1(self.feature)
        nn.init.zeros_(self.spatial.weight)
        nn.init.ones_(self.spatial.bias)
        nn.init.zeros_(self.channel.weight)
        nn.init.ones_(self.channel.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        u = self.feature(feature)
        a = self.spatial(feature)
        v = a * u
        b = self.channel(v.mean(dim=(-2, -1))).unsqueeze(-1).unsqueeze(-1)
        return b * v


class DRNetFineGrainedBranch(nn.Module):
    """Parallel dense fine-grained branch for P3/P4/P5 classification fields."""

    def __init__(
        self,
        channels: tuple[int, int, int],
        num_classes: int,
        config: DRNetRefinementConfig,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("DRNet transfer memerlukan tepat P3/P4/P5")
        self.config = config
        self.num_classes = int(num_classes)
        self.refinements = nn.ModuleList([DualRefinement(channel) for channel in channels])
        self.classifiers = nn.ModuleList(
            [nn.Conv2d(channel, self.num_classes, 1) for channel in channels]
        )
        # The native prediction must be bitwise preserved before learning.
        for classifier in self.classifiers:
            nn.init.zeros_(classifier.weight)
            nn.init.zeros_(classifier.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 3:
            raise ValueError("DRNet transfer memerlukan tepat tiga feature levels")
        return [
            classifier(refinement(feature))
            for refinement, classifier, feature in zip(
                self.refinements, self.classifiers, features
            )
        ]


class DRNetRefinementDetectHead(nn.Module):
    """Native YOLO26 Detect plus a DRNet-inspired subclass branch.

    This is explicitly a one-stage transfer hypothesis, not a literal ORCNN
    reproduction: DRNet's RoI-level FGB is applied densely at P3/P4/P5.  The
    native box path is untouched, and only class logits receive the parallel
    fine-grained residual.
    """

    def __init__(self, base_head: nn.Module, config: DRNetRefinementConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("DRNet transfer memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("DRNet transfer dikunci untuk YOLO26 end-to-end")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("DRNet transfer memerlukan P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.fine_grained = DRNetFineGrainedBranch(channels, int(base_head.nc), config)
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
        expose_fine_logits: bool,
    ) -> dict[str, torch.Tensor]:
        boxes, native_logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            native_logits.append(branch["cls_head"][index](features[index]))
        fine_logits = self.fine_grained(features)
        batch_size = features[0].shape[0]
        output = {
            "boxes": torch.cat(
                [value.view(batch_size, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [
                    (native + float(self.config.correction_scale) * fine).view(
                        batch_size, self.nc, -1
                    )
                    for native, fine in zip(native_logits, fine_logits)
                ],
                dim=-1,
            ),
            "feats": features,
        }
        if expose_fine_logits:
            output["dr_fine_logits"] = torch.cat(
                [value.view(batch_size, self.nc, -1) for value in fine_logits], dim=-1
            ).transpose(1, 2).contiguous()
        return output

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            one2many = self._forward_branch(
                features, self.one2many, expose_fine_logits=self.config.use_cml
            )
            one2one = self._forward_branch(
                [value.detach() for value in features],
                self.one2one,
                expose_fine_logits=False,
            )
            return {"one2many": one2many, "one2one": one2one}

        one2many = (
            self._forward_branch(features, self.one2many, expose_fine_logits=False)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [value.detach() for value in features], self.one2one, expose_fine_logits=False
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_drnet_refinement_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strict native-D0 head transfer into the wrapper namespace."""
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, DRNetRefinementDetectHead):
        raise TypeError("Target bukan DRNetRefinementDetectHead")
    if isinstance(source_head, DRNetRefinementDetectHead):
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


class DRNetRefinementDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        drnet_refinement: DRNetRefinementConfig | dict[str, Any] | None = None,
    ) -> None:
        self.drnet_refinement_config = DRNetRefinementConfig.from_mapping(drnet_refinement)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = DRNetRefinementDetectHead(
            self.model[-1], self.drnet_refinement_config
        )

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss
        from .loss import DRNetDetectionLoss

        return E2ELoss(self, loss_fn=DRNetDetectionLoss)
