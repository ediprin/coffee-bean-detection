from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import PCLConfig, load_pcl_detector_weights
from .task import PCLDetectionModel


def make_pcl_trainer(
    config: PCLConfig | dict[str, Any],
    *,
    d0_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = PCLConfig.from_mapping(config)
    bound_checkpoint = (
        Path(d0_checkpoint).expanduser().resolve()
        if d0_checkpoint is not None
        else None
    )

    class PCLTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                PCLDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    pcl=frozen,
                )
            )
            if bound_checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(bound_checkpoint)).model
                transfer = load_pcl_detector_weights(model, source)
            elif weights:
                transfer = load_pcl_detector_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"PCL NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )

    PCLTrainer.__name__ = "PCLTrainer"
    return PCLTrainer
