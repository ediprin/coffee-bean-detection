from __future__ import annotations

from pathlib import Path
from typing import Any

from coffee_detector.afab.operator import AFABConfig
from coffee_detector.fsce_cpe.model import FSCECPEConfig, load_fsce_cpe_detector_weights

from .model import AF2CPEDetectionModel


def make_af2_cpe_trainer(
    afab: AFABConfig | dict[str, Any],
    cpe: FSCECPEConfig | dict[str, Any],
    *,
    af2_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_cpe = FSCECPEConfig.from_mapping(cpe)
    checkpoint = Path(af2_checkpoint).expanduser().resolve()

    class AF2CPETrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2CPEDetectionModel(
                    cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose,
                    afab=frozen_afab, cpe=frozen_cpe,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(checkpoint)).model
                # The wrapper changes native Detect keys to ``base_head.*``;
                # use the strict native-head remap instead of a partial generic load.
                transfer = load_fsce_cpe_detector_weights(model, source)
                print(f"AF2+CPE AF2 CHECKPOINT TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    AF2CPETrainer.__name__ = "AF2CPETrainer"
    return AF2CPETrainer
