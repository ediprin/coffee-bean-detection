from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .interaction import (
    DRNetInteractionConfig,
    DRNetInteractionDetectionModel,
    build_entity_family_mapping,
    load_drnet_interaction_weights,
)


def make_drnet_interaction_trainer(
    config: DRNetInteractionConfig | Mapping[str, Any],
    *,
    ontology_path: str | Path,
    d0_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = DRNetInteractionConfig.from_mapping(config)
    ontology = Path(ontology_path).expanduser().resolve()
    bound_checkpoint = (
        Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint is not None else None
    )

    class DRNetInteractionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            class_to_group, group_names, members = build_entity_family_mapping(
                self.data["names"], ontology
            )
            print(
                "DRNET INTERACTION COARSE TAXONOMY: "
                + "; ".join(f"{name}={members[name]}" for name in group_names),
                flush=True,
            )
            model = self.set_model_names_for_load(
                DRNetInteractionDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    drnet_interaction=frozen,
                    class_to_group=class_to_group,
                    group_names=group_names,
                )
            )
            if bound_checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(bound_checkpoint)).model
                transfer = load_drnet_interaction_weights(model, source)
            elif weights:
                transfer = load_drnet_interaction_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"DRNET INTERACTION NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )

    DRNetInteractionTrainer.__name__ = "DRNetInteractionTrainer"
    return DRNetInteractionTrainer
