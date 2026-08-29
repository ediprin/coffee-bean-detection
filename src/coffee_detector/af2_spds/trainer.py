from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from coffee_detector.afab.operator import AFABConfig

from .config import AF2SPDSConfig
from .model import AF2SPDSDetectionModel, load_af2_spds_weights


def make_af2_spds_trainer(
    afab: AFABConfig | Mapping[str, Any],
    spds: AF2SPDSConfig | Mapping[str, Any],
    *,
    initial_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_spds = AF2SPDSConfig.from_mapping(spds)
    checkpoint = (
        Path(initial_checkpoint).expanduser().resolve()
        if initial_checkpoint is not None
        else None
    )

    class AF2SPDSTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2SPDSDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    spds=frozen_spds,
                )
            )
            if checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(checkpoint)).model
                transfer = load_af2_spds_weights(model, source)
            elif weights:
                transfer = load_af2_spds_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"AF2-SPDS WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            # Ultralytics 8.4.96 exposes no ``de_parallel`` helper.  Directly
            # unwrap DDP/DataParallel, matching the repository's other
            # version-pinned trainers.
            model = self.model.module if hasattr(self.model, "module") else self.model
            model.af2_spds_epoch = int(self.epoch)
            return batch

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2SPDSTrainer.__name__ = "AF2SPDSTrainer"
    return AF2SPDSTrainer
