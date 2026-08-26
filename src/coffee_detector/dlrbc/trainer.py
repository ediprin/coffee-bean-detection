from __future__ import annotations

from pathlib import Path
from typing import Callable

from .model import DLRBCConfig, DLRBCDetectionModel, load_dlrbc_weights


def make_fresh_trainer(
    *,
    arm: str,
    model_yaml: str | Path,
    pretrained_checkpoint: str | Path,
    seed: int,
    expected_initial_fingerprint: str,
    build_fresh: Callable,
    config: DLRBCConfig | None = None,
):
    """Create a trainer that permits only fresh official init or same-run resume."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    model_yaml = Path(model_yaml).expanduser().resolve()
    pretrained_checkpoint = Path(pretrained_checkpoint).expanduser().resolve()

    class FreshDLRBCTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            resumed = bool(getattr(self.args, "resume", False))
            if resumed:
                if config is None:
                    from ultralytics.nn.tasks import DetectionModel

                    model = DetectionModel(
                        str(model_yaml),
                        nc=self.data["nc"],
                        ch=self.data["channels"],
                        verbose=verbose,
                    )
                    if weights:
                        model.load(weights)
                else:
                    model = DLRBCDetectionModel(
                        str(model_yaml),
                        nc=self.data["nc"],
                        ch=self.data["channels"],
                        verbose=verbose,
                        dlrbc=config,
                    )
                    if weights:
                        load_dlrbc_weights(model, weights)
                return self.set_model_names_for_load(model)

            model, metadata = build_fresh(
                arm=arm,
                pretrained_checkpoint=pretrained_checkpoint,
                seed=seed,
                verbose=verbose,
            )
            fingerprint = metadata["full_state_sha256"]
            if fingerprint != expected_initial_fingerprint:
                raise RuntimeError(
                    "Fresh initial state berbeda dari static audit: "
                    f"{fingerprint} != {expected_initial_fingerprint}"
                )
            print(
                "FRESH OFFICIAL INITIALIZATION MATCH:",
                fingerprint,
                "| arm=",
                arm,
                flush=True,
            )
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    FreshDLRBCTrainer.__name__ = f"{arm.title().replace('_', '')}Trainer"
    return FreshDLRBCTrainer
