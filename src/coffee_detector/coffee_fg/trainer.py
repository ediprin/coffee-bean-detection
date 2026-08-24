from __future__ import annotations

from typing import Any

from .model import CoffeeFGConfig, CoffeeFGDetectionModel


def make_coffee_fg_trainer(config: CoffeeFGConfig | dict[str, Any]):
    """Create an Ultralytics trainer whose model/loss understand CoffeeFG."""

    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang; jalankan pip install -e .") from error

    frozen = config if isinstance(config, CoffeeFGConfig) else CoffeeFGConfig.from_mapping(config)

    class CoffeeFGTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                CoffeeFGDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    coffee_fg=frozen,
                )
            )
            if weights:
                model.load(weights)
            return model

        def get_validator(self):
            validator = super().get_validator()
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "fg_cls_loss"
            return validator

        def final_eval(self):
            """Strip checkpoints; the experiment runner validates in a fresh process.

            Ultralytics' second, in-process reload of a custom pickled model can
            terminate the Windows worker after the normal final-epoch validation.
            The checkpoint itself is valid. Avoiding that duplicate reload also
            keeps long remote runs from failing after all epochs are complete.
            """

            from ultralytics.utils.torch_utils import strip_optimizer

            last_checkpoint = (
                strip_optimizer(self.last) if self.last.exists() else {}
            )
            if self.best.exists():
                strip_optimizer(
                    self.best,
                    updates={"train_results": last_checkpoint.get("train_results")},
                )
            print(
                "CoffeeFG checkpoint final siap; evaluasi terkunci dijalankan "
                "oleh experiment runner pada proses terpisah.",
                flush=True,
            )

        def preprocess_batch(self, batch):
            model = self.model.module if hasattr(self.model, "module") else self.model
            head = model.model[-1]
            start = frozen.predicted_start_epoch
            end = frozen.predicted_full_epoch
            if self.epoch < start:
                head.proposal_mix = 0.0
            else:
                head.proposal_mix = min(
                    (self.epoch - start + 1) / (end - start + 1),
                    1.0,
                )
            if getattr(self, "_coffee_fg_logged_epoch", None) != self.epoch:
                print(
                    f"CoffeeFG epoch {self.epoch + 1}/{self.epochs} | "
                    f"predicted-candidate mix={head.proposal_mix:.3f}",
                    flush=True,
                )
                self._coffee_fg_logged_epoch = self.epoch
            return super().preprocess_batch(batch)

    CoffeeFGTrainer.__name__ = f"CoffeeFGTrainer{frozen.mode.title().replace('_', '')}"
    return CoffeeFGTrainer
