from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from coffee_detector.wav1_factorization.config import WAV1FactorizationConfig

from .config import STBGuidedConfig
from .model import STBGuidedDetectionModel, load_stb_guided_weights


def make_stb_guided_trainer(
    factorization: WAV1FactorizationConfig | Mapping[str, Any],
    guided: STBGuidedConfig | Mapping[str, Any],
    *,
    d0_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_factorization = WAV1FactorizationConfig.from_mapping(factorization)
    if frozen_factorization.arm != "WAV_L1":
        raise ValueError("Trainer STB-guided hanya menerima WAV_L1")
    frozen_guided = STBGuidedConfig.from_mapping(guided)
    d0 = Path(d0_checkpoint).expanduser().resolve()

    class STBGuidedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                STBGuidedDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    factorization=frozen_factorization,
                    stb_guided=frozen_guided,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_stb_guided_weights(model, YOLO(str(d0)).model)
                print(f"STB-GUIDED D0 TRANSFER: {transfer}", flush=True)
            elif weights:
                transfer = load_stb_guided_weights(model, weights)
                print(f"STB-GUIDED RESUME TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )

    STBGuidedTrainer.__name__ = f"STBGuided{frozen_guided.mode.title().replace('_', '')}Trainer"
    return STBGuidedTrainer
