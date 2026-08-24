from __future__ import annotations

from pathlib import Path
from typing import Any

from coffee_detector.afab.model import load_afab_weights
from coffee_detector.afab.operator import AFABConfig

from .config import DIDAAF2Config
from .model import DIDAAF2DetectionModel
from .style import diversify_appearance


def make_dida_af2_trainer(
    afab: AFABConfig | dict[str, Any],
    dida: DIDAAF2Config | dict[str, Any],
    *,
    initial_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_dida = DIDAAF2Config.from_mapping(dida)
    checkpoint = (
        Path(initial_checkpoint).expanduser().resolve() if initial_checkpoint else None
    )

    class DIDAAF2Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                DIDAAF2DetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    dida=frozen_dida,
                )
            )
            if checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_afab_weights(model, YOLO(str(checkpoint)).model)
            elif weights:
                transfer = load_afab_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"DIDA-AF2 WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def preprocess_batch(self, batch: dict) -> dict:
            batch = super().preprocess_batch(batch)
            weak = batch["img"]
            batch["img_style"] = (
                diversify_appearance(weak, frozen_dida)
                if frozen_dida.dg_enabled
                else weak.clone()
            )
            return batch

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    DIDAAF2Trainer.__name__ = f"DIDAAF2Trainer_{frozen_dida.mode}"
    return DIDAAF2Trainer
