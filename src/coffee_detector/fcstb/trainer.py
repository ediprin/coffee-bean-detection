from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .model import FCSTBConfig, load_fcstb_weights
from .task import FCSTBTaskModel


def make_fcstb_trainer(
    config: FCSTBConfig | Mapping[str, Any],
    *,
    stb: Mapping[str, Any],
    source_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = FCSTBConfig.from_mapping(config)
    source = Path(source_checkpoint).expanduser().resolve()

    class FCSTBTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                FCSTBTaskModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    stb=stb,
                    fcstb=frozen,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_fcstb_weights(model, YOLO(str(source)).model)
                print(f"FC-STB STRICT TRANSFER: {transfer}", flush=True)
            elif weights:
                transfer = load_fcstb_weights(model, weights)
                print(f"FC-STB RESUME TRANSFER: {transfer}", flush=True)
            return model

        def _setup_train(self):
            super()._setup_train()
            target = self.model.module if hasattr(self.model, "module") else self.model
            if not isinstance(target, FCSTBTaskModel):
                raise TypeError(type(target).__name__)
            policy = target.apply_freeze_policy()
            print(f"FC-STB FREEZE POLICY [{frozen.mode}]: {policy}", flush=True)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    FCSTBTrainer.__name__ = f"FCSTB{frozen.mode.title()}Trainer"
    return FCSTBTrainer
