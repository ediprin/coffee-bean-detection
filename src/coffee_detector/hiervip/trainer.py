from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .model import HierVIPConfig, build_sni_hierarchy, load_hiervip_detector_weights
from .task import HierVIPDetectionModel


def make_hiervip_trainer(
    config: HierVIPConfig | Mapping[str, Any],
    *,
    ontology_path: str | Path,
    d0_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = HierVIPConfig.from_mapping(config)
    ontology = Path(ontology_path).expanduser().resolve()
    bound_checkpoint = (
        Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint is not None else None
    )

    class HierVIPTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            hierarchy = build_sni_hierarchy(self.data["names"], ontology)
            print(f"HIERVIP HIERARCHY: {hierarchy.to_dict()}", flush=True)
            model = self.set_model_names_for_load(
                HierVIPDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    hiervip=frozen,
                    hierarchy=hierarchy,
                )
            )
            if bound_checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(bound_checkpoint)).model
                transfer = load_hiervip_detector_weights(model, source)
            elif weights:
                transfer = load_hiervip_detector_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"HIERVIP NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )

    HierVIPTrainer.__name__ = "HierVIPTrainer"
    return HierVIPTrainer
