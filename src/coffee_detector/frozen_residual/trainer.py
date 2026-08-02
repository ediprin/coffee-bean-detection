from __future__ import annotations

from typing import Any

from ultralytics.utils.torch_utils import unwrap_model

from .model import (
    FrozenResidualConfig,
    FrozenResidualDetectionModel,
    freeze_native_detector,
    load_frozen_d0_weights,
)


def make_frozen_residual_trainer(
    config: FrozenResidualConfig | dict[str, Any],
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = FrozenResidualConfig.from_mapping(config)

    class FrozenResidualTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                FrozenResidualDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    frozen_residual=frozen,
                )
            )
            if weights:
                transfer = load_frozen_d0_weights(model, weights)
                print(
                    f"FRM1 native D0 transfer: {transfer['native_head_items']} head items | "
                    f"resume={bool(transfer['resume'])}",
                    flush=True,
                )
            freeze_native_detector(model)
            return model

        def _build_train_pipeline(self):
            freeze_native_detector(unwrap_model(self.model))
            return super()._build_train_pipeline()

        def build_optimizer(self, model, *args, **kwargs):
            optimizer = super().build_optimizer(model, *args, **kwargs)
            trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
            for group in optimizer.param_groups:
                group["params"] = [
                    parameter
                    for parameter in group["params"]
                    if id(parameter) in trainable
                ]
            optimized = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if optimized != trainable:
                raise RuntimeError("Optimizer FRM1 tidak identik dengan parameter trainable")
            return optimizer

        def get_validator(self):
            validator = super().get_validator()
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "res_cls_loss"
            return validator

        def preprocess_batch(self, batch):
            model = unwrap_model(self.model)
            freeze_native_detector(model)
            if getattr(self, "_frm_logged_epoch", None) != self.epoch:
                counts = freeze_native_detector(model)
                print(
                    f"FRM1 epoch {self.epoch + 1}/{self.epochs} | "
                    f"trainable={counts['trainable']:,}/{counts['total']:,}",
                    flush=True,
                )
                self._frm_logged_epoch = self.epoch
            return super().preprocess_batch(batch)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )
            print(
                "FRM1 checkpoint siap; runner melakukan evaluasi validation terpisah.",
                flush=True,
            )

    FrozenResidualTrainer.__name__ = "FrozenD0MultilevelResidualTrainer"
    return FrozenResidualTrainer
