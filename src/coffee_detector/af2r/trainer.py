from __future__ import annotations

from pathlib import Path

from coffee_detector.afab import AFABConfig

from .config import AF2RConfig
from .model import AF2RDetectionModel, load_af2r_weights


def make_af2r_trainer(
    afab: AFABConfig | dict,
    af2r: AF2RConfig | dict,
    *,
    initial_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_af2r = AF2RConfig.from_mapping(af2r)
    checkpoint = Path(initial_checkpoint).expanduser().resolve()

    class AF2RTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2RDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    af2r=frozen_af2r,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(checkpoint)).model
                transfer = load_af2r_weights(model, source)
                print(f"AF2R WEIGHT TRANSFER: {transfer}", flush=True)
            elif weights:
                load_af2r_weights(model, weights)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    AF2RTrainer.__name__ = f"AF2RTrainer_{frozen_af2r.conditioning}"
    return AF2RTrainer
