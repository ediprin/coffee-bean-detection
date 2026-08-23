from __future__ import annotations
from pathlib import Path
from ultralytics.models.yolo.detect import DetectionTrainer

from .model import AF2OrientCMC0DetectionModel, load_af2_orient_cmc0_weights
from coffee_detector.af2_iso.config import AF2IsolatedConfig
from coffee_detector.stb.model import STBConfig

def make_af2_orient_cmc0_trainer(
    af2_config: AF2IsolatedConfig | dict,
    stb_config: STBConfig | dict,
    *,
    d0_checkpoint: str | Path | None = None,
):
    frozen_af2 = AF2IsolatedConfig.from_mapping(af2_config)
    frozen_stb = STBConfig.from_mapping(stb_config)
    checkpoint = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class AF2OrientCMC0Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2OrientCMC0DetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    af2_iso=frozen_af2,
                    stb=frozen_stb
                )
            )
            if checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO
                transfer = load_af2_orient_cmc0_weights(model, YOLO(str(checkpoint)).model)
            elif weights:
                transfer = load_af2_orient_cmc0_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"AF2-ORIENT-CMC0 DETECTOR WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    AF2OrientCMC0Trainer.__name__ = "AF2OrientCMC0Trainer"
    return AF2OrientCMC0Trainer
