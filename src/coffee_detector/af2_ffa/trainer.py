from __future__ import annotations

from pathlib import Path
from typing import Any

from coffee_detector.afab import AFABConfig

from .model import AF2FFAConfig, AF2FFADetectionModel, load_af2_ffa_weights


def make_af2_ffa_trainer(
    afab: AFABConfig | dict[str, Any],
    af2_ffa: AF2FFAConfig | dict[str, Any],
    *,
    initial_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_adapter = AF2FFAConfig.from_mapping(af2_ffa)
    checkpoint = Path(initial_checkpoint).expanduser().resolve()

    class AF2FFATrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2FFADetectionModel(
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

                source = YOLO(str(checkpoint)).model
                transfer = load_af2_ffa_weights(model, source)
                print(f"AF2-FFA WEIGHT TRANSFER: {transfer}", flush=True)
            elif weights:
                load_af2_ffa_weights(model, weights)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2FFATrainer.__name__ = f"AF2FFATrainer_{frozen_adapter.conditioning}"
    return AF2FFATrainer
