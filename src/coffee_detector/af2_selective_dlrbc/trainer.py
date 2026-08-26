from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from coffee_detector.afab.operator import AFABConfig

from .model import (
    AF2SelectiveDLRBCConfig,
    AF2SelectiveDLRBCDetectionModel,
    load_af2_selective_weights,
    selective_modules,
)


def make_af2_selective_trainer(
    *,
    af2_checkpoint: str | Path,
    afab: AFABConfig | Mapping[str, Any],
    selective: AF2SelectiveDLRBCConfig | Mapping[str, Any],
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    parent = Path(af2_checkpoint).expanduser().resolve()
    frozen_af2 = AFABConfig.from_mapping(afab)
    frozen_selective = AF2SelectiveDLRBCConfig.from_mapping(selective)

    class AF2SelectiveTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2SelectiveDLRBCDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_af2,
                    selective=frozen_selective,
                )
            )
            source = weights
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(parent)).model
            if source is not None:
                print(
                    "AF2 SELECTIVE WEIGHT TRANSFER:",
                    load_af2_selective_weights(model, source),
                    flush=True,
                )
            return model

        def build_optimizer(
            self,
            model,
            name="auto",
            lr=0.001,
            momentum=0.9,
            decay=1e-5,
            iterations=1e5,
        ):
            modules = selective_modules(model)
            if not modules:
                raise RuntimeError("Selective residual modules tidak ditemukan")
            for parameter in model.parameters():
                parameter.requires_grad = False
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
            trainable = nn.ModuleList(list(modules))
            return super().build_optimizer(
                trainable,
                name=name,
                lr=lr,
                momentum=momentum,
                decay=decay,
                iterations=iterations,
            )

        def _model_train(self):
            self.model.train()
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2SelectiveTrainer.__name__ = "AF2SelectiveDLRBCTrainer"
    return AF2SelectiveTrainer
