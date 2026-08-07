from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import MRLConfig, MRLDetectionModel, load_mrl_detector_weights


def make_mrl_trainer(
    config: MRLConfig | dict[str, Any], *, d0_checkpoint: str | Path | None = None
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = MRLConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class MRLTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                MRLDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    mrl=frozen,
                )
            )
            if bound is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO
                transfer = load_mrl_detector_weights(model, YOLO(str(bound)).model)
            elif weights:
                transfer = load_mrl_detector_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"MRL NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    MRLTrainer.__name__ = "MRLTrainer"
    return MRLTrainer
