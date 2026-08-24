from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .model import SGFRConfig, load_sgfr_weights
from .task import SGFRTaskModel


def make_sgfr_trainer(
    config: SGFRConfig | Mapping[str, Any],
    *,
    source_checkpoint: str | Path | None = None,
):
    """Create a trainer that enforces the stage-specific frozen parameter set."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = SGFRConfig.from_mapping(config)
    bound = (
        Path(source_checkpoint).expanduser().resolve()
        if source_checkpoint is not None
        else None
    )

    class SGFRTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                SGFRTaskModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    sgfr=frozen,
                )
            )
            if bound is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_sgfr_weights(model, YOLO(str(bound)).model)
            elif weights:
                transfer = load_sgfr_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"SGFR STRICT TRANSFER: {transfer}", flush=True)
            return model

        def _setup_train(self):
            super()._setup_train()
            target = self.model.module if hasattr(self.model, "module") else self.model
            if not isinstance(target, SGFRTaskModel):
                raise TypeError(f"Trainer menerima {type(target).__name__}, bukan SGFRTaskModel")
            policy = target.apply_freeze_policy()
            print(f"SGFR FREEZE POLICY [{frozen.stage}]: {policy}", flush=True)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )

    SGFRTrainer.__name__ = f"SGFR{frozen.stage.title()}Trainer"
    return SGFRTrainer

