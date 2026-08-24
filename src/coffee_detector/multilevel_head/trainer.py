from __future__ import annotations

from typing import Any

from .model import (
    MultilevelHeadConfig,
    MultilevelHeadDetectionModel,
    load_multilevel_detector_weights,
)


def make_multilevel_head_trainer(
    config: MultilevelHeadConfig | dict[str, Any],
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = MultilevelHeadConfig.from_mapping(config)

    class MultilevelHeadTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                MultilevelHeadDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    multilevel_head=frozen,
                )
            )
            if weights:
                transfer = load_multilevel_detector_weights(model, weights)
                print(f"MULTILEVEL NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def get_validator(self):
            validator = super().get_validator()
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "ml_cls_loss"
            return validator

        def preprocess_batch(self, batch):
            model = self.model.module if hasattr(self.model, "module") else self.model
            head = model.model[-1]
            if self.epoch < frozen.predicted_start_epoch:
                head.proposal_mix = 0.0
            else:
                head.proposal_mix = min(
                    (self.epoch - frozen.predicted_start_epoch + 1)
                    / (frozen.predicted_full_epoch - frozen.predicted_start_epoch + 1),
                    1.0,
                )
            if getattr(self, "_multilevel_logged_epoch", None) != self.epoch:
                print(
                    f"Multilevel {frozen.mode} epoch {self.epoch + 1}/{self.epochs} | "
                    f"predicted-candidate mix={head.proposal_mix:.3f}",
                    flush=True,
                )
                self._multilevel_logged_epoch = self.epoch
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
                "Multilevel checkpoint siap; evaluasi dijalankan oleh runner terpisah.",
                flush=True,
            )

    MultilevelHeadTrainer.__name__ = (
        "MultilevelHeadTrainer" + frozen.mode.title().replace("_", "")
    )
    return MultilevelHeadTrainer
