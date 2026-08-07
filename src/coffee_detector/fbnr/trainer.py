from __future__ import annotations

from typing import Any

from .augment import FBNRConfig, apply_fbnr_transfer


def make_fbnr_trainer(config: FBNRConfig | dict[str, Any]):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = FBNRConfig.from_mapping(config)

    class FBNRTrainer(DetectionTrainer):
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            images = batch["img"]
            bboxes = batch["bboxes"]
            batch_idx = batch["batch_idx"]
            batch["img"] = apply_fbnr_transfer(
                images, bboxes, batch_idx, frozen
            )
            return batch

    suffix = "FGC" if frozen.mode == "foreground_only" else "FBR"
    FBNRTrainer.__name__ = f"FBNRTransferTrainer{suffix}"
    return FBNRTrainer
