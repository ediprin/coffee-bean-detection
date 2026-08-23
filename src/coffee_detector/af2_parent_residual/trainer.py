from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.utils.torch_utils import unwrap_model

from coffee_detector.afab import AFABConfig

from .config import AF2ParentResidualConfig
from .model import (
    AF2ParentResidualDetectionModel,
    freeze_for_parent_residual,
    load_af2_parent_residual_weights,
)


def make_af2_parent_residual_trainer(
    afab: AFABConfig | dict[str, Any],
    parent_residual: AF2ParentResidualConfig | dict[str, Any],
    *,
    initial_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_residual = AF2ParentResidualConfig.from_mapping(parent_residual)
    checkpoint = Path(initial_checkpoint).expanduser().resolve()

    class AF2ParentResidualTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2ParentResidualDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    parent_residual=frozen_residual,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_af2_parent_residual_weights(
                    model, YOLO(str(checkpoint)).model
                )
                print(f"AF2 PARENT STRICT TRANSFER: {transfer}", flush=True)
            elif weights:
                transfer = load_af2_parent_residual_weights(model, weights)
                print(f"AF2 PARENT RESUME TRANSFER: {transfer}", flush=True)
            freeze_for_parent_residual(model)
            return model

        def _setup_train(self):
            super()._setup_train()
            target = unwrap_model(self.model)
            policy = freeze_for_parent_residual(target)
            target.train(True)
            print(f"AF2 PARENT FREEZE POLICY: {policy}", flush=True)

        def _build_train_pipeline(self):
            freeze_for_parent_residual(unwrap_model(self.model))
            return super()._build_train_pipeline()

        def build_optimizer(self, model, *args, **kwargs):
            optimizer = super().build_optimizer(model, *args, **kwargs)
            trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
            for group in optimizer.param_groups:
                group["params"] = [
                    parameter for parameter in group["params"] if id(parameter) in trainable
                ]
            optimized = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if optimized != trainable:
                raise RuntimeError("Optimizer tidak identik dengan parameter residual")
            return optimizer

        def preprocess_batch(self, batch):
            target = unwrap_model(self.model)
            freeze_for_parent_residual(target)
            target.train(True)
            if getattr(self, "_parent_residual_logged_epoch", None) != self.epoch:
                print(
                    f"AF2-{frozen_residual.family.upper()}-{frozen_residual.conditioning} "
                    f"epoch {self.epoch + 1}/{self.epochs}",
                    flush=True,
                )
                self._parent_residual_logged_epoch = self.epoch
            return super().preprocess_batch(batch)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2ParentResidualTrainer.__name__ = (
        f"AF2ParentResidualTrainer_{frozen_residual.family}_{frozen_residual.conditioning}"
    )
    return AF2ParentResidualTrainer
