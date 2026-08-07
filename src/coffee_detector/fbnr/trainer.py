from __future__ import annotations

from typing import Any

from .augment import FBNRConfig, apply_fbnr_transfer


def make_fbnr_trainer(config: FBNRConfig | dict[str, Any]):
    """DetectionTrainer with training-only FBNR image regularization.

    Validation/inference remain native Ultralytics behavior because the custom
    operation lives exclusively in the training trainer's preprocess_batch.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = FBNRConfig.from_mapping(config)

    class FBNRTrainer(DetectionTrainer):
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            batch["img"] = apply_fbnr_transfer(
                batch["img"], batch["bboxes"], batch["batch_idx"], frozen
            )
            return batch

    suffix = {
        "foreground_only": "FGRC",
        "background_linear": "BRBBLinear",
        "background_gradient": "BRBBGradient",
        "stochastic_decoupled": "Decoupled",
    }[frozen.mode]
    FBNRTrainer.__name__ = f"FBNRTrainer{suffix}"
    return FBNRTrainer
