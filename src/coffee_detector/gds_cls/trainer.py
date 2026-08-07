from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import GDSClsConfig, GDSClsDetectionModel, load_gds_cls_weights


def make_gds_cls_trainer(
    config: GDSClsConfig | dict[str, Any], *, d0_checkpoint: str | Path | None = None
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = GDSClsConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class GDSClsTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                GDSClsDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    gds_cls=frozen,
                )
            )
            if bound is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO
                transfer = load_gds_cls_weights(model, YOLO(str(bound)).model)
            elif weights:
                transfer = load_gds_cls_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"GDSC1 NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    GDSClsTrainer.__name__ = "GDSClsTrainer"
    return GDSClsTrainer
