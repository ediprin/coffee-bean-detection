from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.utils.torch_utils import unwrap_model

from coffee_detector.afab import AFABConfig

from .model import (
    AF2RCCConfig,
    AF2RCCDetectionModel,
    freeze_for_rcc,
    load_af2_rcc_weights,
)


def make_af2_rcc_trainer(
    afab: AFABConfig | dict[str, Any],
    af2_rcc: AF2RCCConfig | dict[str, Any],
    *,
    initial_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_rcc = AF2RCCConfig.from_mapping(af2_rcc)
    checkpoint = Path(initial_checkpoint).expanduser().resolve()

    class AF2RCCTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2RCCDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    af2_rcc=frozen_rcc,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_af2_rcc_weights(model, YOLO(str(checkpoint)).model)
                print(f"AF2-RCC STRICT AF2 TRANSFER: {transfer}", flush=True)
            elif weights:
                transfer = load_af2_rcc_weights(model, weights)
                print(f"AF2-RCC RESUME TRANSFER: {transfer}", flush=True)
            freeze_for_rcc(model)
            return model

        def _setup_train(self):
            super()._setup_train()
            target = unwrap_model(self.model)
            policy = freeze_for_rcc(target)
            print(f"AF2-RCC FREEZE POLICY: {policy}", flush=True)

        def _build_train_pipeline(self):
            # BaseTrainer deliberately re-enables parameters that were frozen
            # before setup. Reassert the audited policy immediately before it
            # builds the optimizer.
            freeze_for_rcc(unwrap_model(self.model))
            return super()._build_train_pipeline()

        def build_optimizer(self, model, *args, **kwargs):
            optimizer = super().build_optimizer(model, *args, **kwargs)
            trainable = {id(p) for p in model.parameters() if p.requires_grad}
            for group in optimizer.param_groups:
                group["params"] = [p for p in group["params"] if id(p) in trainable]
            optimized = {
                id(p) for group in optimizer.param_groups for p in group["params"]
            }
            if optimized != trainable:
                raise RuntimeError("Optimizer AF2-RCC tidak identik dengan parameter trainable")
            return optimizer

        def preprocess_batch(self, batch):
            freeze_for_rcc(unwrap_model(self.model))
            if getattr(self, "_rcc_logged_epoch", None) != self.epoch:
                print(
                    f"AF2-RCC epoch {self.epoch + 1}/{self.epochs} | trainable=189",
                    flush=True,
                )
                self._rcc_logged_epoch = self.epoch
            return super().preprocess_batch(batch)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2RCCTrainer.__name__ = f"AF2RCCTrainer_{frozen_rcc.conditioning}"
    return AF2RCCTrainer
