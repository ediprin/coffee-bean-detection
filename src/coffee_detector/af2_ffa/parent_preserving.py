"""Parent-preserving AF2 + FFAB2 training support.

This module is deliberately separate from the ordinary from-start FFAB2 path.
A completed AF2 checkpoint is the parent. All parent parameters stay frozen;
only the three FFAB adapters are trainable. The native classification path is
kept explicitly through ``fusion_mode=parent_residual`` and box regression is
unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from coffee_detector.afab import AFABConfig
from .model import AF2FFAConfig, AF2FFADetectionModel, AF2FFADetectHead, load_af2_ffa_weights


class AF2FFAParentPreservingModel(AF2FFADetectionModel):
    """AF2FFADetectionModel whose completed AF2 parent cannot be updated."""

    def __init__(
        self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, afab=None, af2_ffa=None
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab, af2_ffa=af2_ffa)
        self._parent_frozen = False

    @property
    def parent_head(self) -> AF2FFADetectHead:
        head = self.model[-1]
        if not isinstance(head, AF2FFADetectHead):
            raise TypeError(f"Expected AF2FFADetectHead, got {type(head).__name__}")
        return head

    def freeze_parent(self) -> dict[str, int]:
        """Freeze every parameter except the FFAB adapters.

        Ultralytics may call ``model.train()`` repeatedly. ``train()`` below
        therefore also keeps every BatchNorm buffer in eval mode so the frozen
        parent cannot drift through running statistics.
        """

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for adapter in self.parent_head.adapters:
            for parameter in adapter.parameters():
                parameter.requires_grad_(True)
        self._parent_frozen = True
        self._enforce_parent_state()
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {"total_parameters": int(total), "trainable_parameters": int(trainable)}

    def _enforce_parent_state(self) -> None:
        if not getattr(self, "_parent_frozen", False):
            return
        # Adapter modules do not contain BatchNorm. Keeping all BN layers in
        # eval mode prevents frozen-parent running_mean/running_var drift.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for adapter in self.parent_head.adapters:
            adapter.train(self.training)
            for parameter in adapter.parameters():
                parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode:
            self._enforce_parent_state()
        return result


def parent_parameter_names(model: AF2FFAParentPreservingModel) -> tuple[str, ...]:
    """Return all frozen parameter names; useful for audits and contracts."""

    adapter_prefix = f"model.{len(model.model) - 1}.adapters."
    return tuple(
        name for name, _ in model.named_parameters() if not name.startswith(adapter_prefix)
    )


def adapter_parameter_names(model: AF2FFAParentPreservingModel) -> tuple[str, ...]:
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def make_af2_ffa_parent_trainer(
    afab: AFABConfig | dict[str, Any],
    af2_ffa: AF2FFAConfig | dict[str, Any],
    *,
    parent_checkpoint: str | Path,
):
    """Build a DetectionTrainer that updates FFAB adapters only."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_adapter = AF2FFAConfig.from_mapping(af2_ffa)
    if frozen_adapter.fusion_mode != "parent_residual":
        raise ValueError("Parent-preserving trainer requires fusion_mode=parent_residual")
    if frozen_adapter.ambiguity_gate != "none":
        raise ValueError("First parent-preserving study does not authorize ambiguity gating")
    parent = Path(parent_checkpoint).expanduser().resolve()
    if not parent.is_file():
        raise FileNotFoundError(parent)

    class AF2FFAParentTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2FFAParentPreservingModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    af2_ffa=frozen_adapter,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(parent)).model
                transfer = load_af2_ffa_weights(model, source)
                print(f"AF2 PARENT WEIGHT TRANSFER: {transfer}", flush=True)
            elif weights:
                load_af2_ffa_weights(model, weights)
            summary = model.freeze_parent()
            print(f"AF2 PARENT FROZEN: {summary}", flush=True)
            return model

        def preprocess_batch(self, batch):
            # Fail-safe against framework code that toggles requires_grad.
            if isinstance(self.model, AF2FFAParentPreservingModel):
                self.model._enforce_parent_state()
            return super().preprocess_batch(batch)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2FFAParentTrainer.__name__ = (
        f"AF2FFAParentTrainer_{frozen_adapter.conditioning}"
    )
    return AF2FFAParentTrainer
