from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from coffee_detector.afab.operator import AFABConfig

from .config import AF2ScaffoldConfig
from .model import (
    AF2ScaffoldDetectionModel,
    TrainingOnlyMultilevelDetectHead,
    load_af2_scaffold_weights,
)


def make_af2_scaffold_trainer(
    afab: AFABConfig | Mapping[str, Any],
    scaffold: AF2ScaffoldConfig | Mapping[str, Any],
    *,
    initial_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_scaffold = AF2ScaffoldConfig.from_mapping(scaffold)
    checkpoint = Path(initial_checkpoint).expanduser().resolve() if initial_checkpoint else None

    class AF2ScaffoldTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2ScaffoldDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    scaffold=frozen_scaffold,
                )
            )
            if checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_af2_scaffold_weights(model, YOLO(str(checkpoint)).model)
            elif weights:
                transfer = load_af2_scaffold_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"AF2 SCAFFOLD WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def preprocess_batch(self, batch):
            model = self.model.module if hasattr(self.model, "module") else self.model
            head = model.model[-1]
            if not isinstance(head, TrainingOnlyMultilevelDetectHead):
                raise TypeError("Training kehilangan multilevel scaffold")
            strength = frozen_scaffold.strength(self.epoch)
            head.set_scaffold_strength(strength)
            if getattr(self, "_af2_scaffold_logged_epoch", None) != self.epoch:
                print(
                    f"AF2MTS1 epoch {self.epoch + 1}/{self.epochs} | "
                    f"train-only strength={strength:.4f}",
                    flush=True,
                )
                self._af2_scaffold_logged_epoch = self.epoch
            return super().preprocess_batch(batch)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )
            print("AF2MTS1 checkpoint siap; evaluasi dilakukan pada jalur bypass.", flush=True)

    AF2ScaffoldTrainer.__name__ = "AF2ScaffoldTrainer"
    return AF2ScaffoldTrainer
