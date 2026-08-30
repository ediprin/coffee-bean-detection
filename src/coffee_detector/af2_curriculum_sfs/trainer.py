from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from coffee_detector.afab.operator import AFABConfig

from .config import AF2CurriculumSFSConfig, curriculum_state
from .model import AF2CurriculumSFSDetectionModel, load_af2_curriculum_sfs_weights


def make_af2_curriculum_sfs_trainer(
    afab: AFABConfig | Mapping[str, Any],
    curriculum: AF2CurriculumSFSConfig | Mapping[str, Any],
    *,
    initial_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_curriculum = AF2CurriculumSFSConfig.from_mapping(curriculum)
    checkpoint = Path(initial_checkpoint).expanduser().resolve()

    class AF2CurriculumSFSTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2CurriculumSFSDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    curriculum=frozen_curriculum,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(checkpoint)).model
                transfer = load_af2_curriculum_sfs_weights(model, source)
            elif weights:
                transfer = load_af2_curriculum_sfs_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"AF2 CURRICULUM-SFS WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            model = self.model.module if hasattr(self.model, "module") else self.model
            model.af2_curriculum_epoch = int(self.epoch)
            state = curriculum_state(
                frozen_curriculum, epoch=int(self.epoch), epochs=int(self.epochs)
            )
            if getattr(self, "_af2_curriculum_logged_epoch", None) != self.epoch:
                print(
                    f"AF2CURR1 epoch {self.epoch + 1}/{self.epochs} | "
                    f"phase={state.phase} sfs={state.sfs_strength:.3f} "
                    f"aux={state.auxiliary_gain:.4f}",
                    flush=True,
                )
                self._af2_curriculum_logged_epoch = self.epoch
            return batch

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2CurriculumSFSTrainer.__name__ = "AF2CurriculumSFSTrainer"
    return AF2CurriculumSFSTrainer
