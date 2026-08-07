from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import STBConfig, STBDetectionModel, load_stb_weights


def make_stb_trainer(
    config: STBConfig | dict[str, Any], *, d0_checkpoint: str | Path | None = None
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = STBConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class STBTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                STBDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    stb=frozen,
                )
            )
            if bound is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO
                transfer = load_stb_weights(model, YOLO(str(bound)).model)
            elif weights:
                transfer = load_stb_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"STB NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    STBTrainer.__name__ = "STBTrainer"
    return STBTrainer
