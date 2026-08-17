from __future__ import annotations

from pathlib import Path

from coffee_detector.afab import AFABConfig

from .model import AF2CalibratedDetectionModel, load_af2cal_weights


def make_af2cal_trainer(
    afab: AFABConfig | dict,
    *,
    initial_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = AFABConfig.from_mapping(afab)
    checkpoint = Path(initial_checkpoint).expanduser().resolve()

    class AF2CALTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2CalibratedDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(checkpoint)).model
                transfer = load_af2cal_weights(model, source)
                print(f"AF2-CAL3 WEIGHT TRANSFER: {transfer}", flush=True)
            elif weights:
                load_af2cal_weights(model, weights)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2CALTrainer.__name__ = "AF2CALTrainer"
    return AF2CALTrainer

