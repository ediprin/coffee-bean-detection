from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import FTIFConfig, load_ftif_detector_weights
from .task import FTIFDetectionModel
from .text_embeddings import load_text_embedding_payload


def make_ftif_trainer(
    config: FTIFConfig | dict[str, Any],
    *,
    text_embedding_path: str | Path,
    d0_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = FTIFConfig.from_mapping(config)
    cache_path = Path(text_embedding_path).expanduser().resolve()
    bound_checkpoint = (
        Path(d0_checkpoint).expanduser().resolve()
        if d0_checkpoint is not None
        else None
    )

    class FTIFTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            embeddings, metadata = load_text_embedding_payload(
                cache_path,
                class_names=self.data["names"],
                prompt_mode=frozen.prompt_mode,
            )
            model = self.set_model_names_for_load(
                FTIFDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    ftif=frozen,
                    text_embeddings=embeddings,
                )
            )
            model.ftif_text_metadata = metadata
            if bound_checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO
                source = YOLO(str(bound_checkpoint)).model
                transfer = load_ftif_detector_weights(model, source)
            elif weights:
                transfer = load_ftif_detector_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"FTIF NATIVE HEAD TRANSFER: {transfer}", flush=True)
            print(f"FTIF TEXT CACHE: {metadata}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last.get("train_results")},
                )

    FTIFTrainer.__name__ = "FTIFTrainer"
    return FTIFTrainer
