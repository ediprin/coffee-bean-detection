from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from coffee_detector.afab.operator import AFABConfig

from .config import AF2ComplementConfig
from .model import AF2ComplementDetectionModel, load_af2_complement_weights


def make_af2_complement_trainer(
    afab: AFABConfig | Mapping[str, Any],
    complement: AF2ComplementConfig | Mapping[str, Any],
    *,
    initial_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_complement = AF2ComplementConfig.from_mapping(complement)
    checkpoint = (
        Path(initial_checkpoint).expanduser().resolve()
        if initial_checkpoint is not None
        else None
    )

    class AF2ComplementTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2ComplementDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    complement=frozen_complement,
                )
            )
            if checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(checkpoint)).model
                transfer = load_af2_complement_weights(model, source)
            elif weights:
                transfer = load_af2_complement_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"AF2 COMPLEMENT WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )

    AF2ComplementTrainer.__name__ = "AF2ComplementTrainer"
    return AF2ComplementTrainer
