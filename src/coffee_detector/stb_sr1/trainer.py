from __future__ import annotations

from pathlib import Path
from typing import Any

from coffee_detector.stb import STBConfig

from .model import STBSR1DetectionModel, load_stb_sr1_weights


def make_stb_sr1_trainer(
    config: STBConfig | dict[str, Any], *, d0_checkpoint: str | Path
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = STBConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve()

    class STBSR1Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                STBSR1DetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    stb=frozen,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_stb_sr1_weights(model, YOLO(str(bound)).model)
                print(f"STB-SR1 NATIVE D0 TRANSFER: {transfer}", flush=True)
            elif weights is not None:
                transfer = load_stb_sr1_weights(model, weights)
                print(f"STB-SR1 RESUME WEIGHT TRANSFER: {transfer}", flush=True)
            else:
                raise RuntimeError("Resume STB-SR1 memerlukan checkpoint weights")
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    STBSR1Trainer.__name__ = "STBSR1Trainer"
    return STBSR1Trainer
