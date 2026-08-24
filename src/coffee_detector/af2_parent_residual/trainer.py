from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics.utils.torch_utils import unwrap_model

from coffee_detector.afab import AFABConfig

from .config import AF2ParentResidualConfig
from .model import (
    AF2ParentResidualDetectionModel,
    AF2ParentResidualDetectHead,
    freeze_for_parent_residual,
    load_af2_parent_residual_weights,
)


def _is_residual_state_key(name: str) -> bool:
    return ".residual." in name


def sync_frozen_parent_to_ema(model, ema_model) -> dict[str, int]:
    """Restore every frozen-parent tensor in EMA from the live frozen model.

    Ultralytics ModelEMA averages every floating tensor in the state dict, even
    parameters/buffers that are intentionally frozen. That is harmless for
    ordinary training, but a parent-preserving experiment needs the serialized
    parent to remain exactly the source parent. Residual tensors are deliberately
    left under normal EMA updates; only non-residual state is copied exactly.
    """

    live = unwrap_model(model)
    ema = unwrap_model(ema_model)
    live_state = live.state_dict()
    ema_state = ema.state_dict()
    if live_state.keys() != ema_state.keys():
        raise RuntimeError("EMA dan live model tidak memiliki state schema identik")
    copied = 0
    skipped = 0
    with torch.no_grad():
        for name, ema_value in ema_state.items():
            if _is_residual_state_key(name):
                skipped += 1
                continue
            source = live_state[name]
            if ema_value.shape != source.shape or ema_value.dtype != source.dtype:
                raise RuntimeError(f"EMA parent state tidak kompatibel: {name}")
            ema_value.copy_(source)
            copied += 1
    return {"parent_state_items_synced": copied, "residual_state_items_left_to_ema": skipped}


def assert_parent_residual_runtime_state(model) -> dict[str, int]:
    """Fail closed unless only the residual is trainable and parent modules are eval."""

    target = unwrap_model(model)
    layers = getattr(target, "model", None)
    if not layers or not isinstance(layers[-1], AF2ParentResidualDetectHead):
        raise RuntimeError("Runtime model bukan AF2 parent-residual")
    head = layers[-1]
    parent_trainable = [
        name
        for name, parameter in target.named_parameters()
        if parameter.requires_grad and ".residual." not in name
    ]
    residual_trainable = [
        name
        for name, parameter in target.named_parameters()
        if parameter.requires_grad and ".residual." in name
    ]
    if parent_trainable:
        raise RuntimeError(f"Parent parameter kembali trainable: {parent_trainable[:5]}")
    if not residual_trainable:
        raise RuntimeError("Tidak ada residual parameter trainable")
    if any(layer.training for layer in list(layers)[:-1]):
        raise RuntimeError("Backbone/neck parent tidak berada pada eval mode")
    if head.base_head.training:
        raise RuntimeError("Native Detect parent tidak berada pada eval mode")
    if not head.residual.training:
        raise RuntimeError("Residual tidak berada pada train mode")
    return {
        "parent_trainable": 0,
        "residual_trainable_tensors": len(residual_trainable),
    }


def make_af2_parent_residual_trainer(
    afab: AFABConfig | dict[str, Any],
    parent_residual: AF2ParentResidualConfig | dict[str, Any],
    *,
    initial_checkpoint: str | Path,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_residual = AF2ParentResidualConfig.from_mapping(parent_residual)
    checkpoint = Path(initial_checkpoint).expanduser().resolve()

    class AF2ParentResidualTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2ParentResidualDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    parent_residual=frozen_residual,
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_af2_parent_residual_weights(
                    model, YOLO(str(checkpoint)).model
                )
                print(f"AF2 PARENT STRICT TRANSFER: {transfer}", flush=True)
            elif weights:
                transfer = load_af2_parent_residual_weights(model, weights)
                print(f"AF2 PARENT RESUME TRANSFER: {transfer}", flush=True)
            freeze_for_parent_residual(model)
            return model

        def _setup_train(self):
            # Ultralytics temporarily re-enables floating frozen parameters in
            # BaseTrainer._setup_train. Our _build_train_pipeline freezes them
            # again before optimizer construction; this postcondition reasserts
            # the contract after the base setup is complete.
            super()._setup_train()
            target = unwrap_model(self.model)
            policy = freeze_for_parent_residual(target)
            target.train(True)
            runtime = assert_parent_residual_runtime_state(target)
            if self.ema:
                ema_sync = sync_frozen_parent_to_ema(target, self.ema.ema)
            else:
                ema_sync = {}
            print(
                f"AF2 PARENT FREEZE POLICY: {policy} runtime={runtime} ema={ema_sync}",
                flush=True,
            )

        def _build_train_pipeline(self):
            # This override is intentionally before BaseTrainer builds the
            # optimizer. It prevents its earlier generic unfreeze pass from
            # leaking parent parameters into optimizer groups.
            target = unwrap_model(self.model)
            freeze_for_parent_residual(target)
            target.train(True)
            return super()._build_train_pipeline()

        def build_optimizer(self, model, *args, **kwargs):
            optimizer = super().build_optimizer(model, *args, **kwargs)
            trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
            for group in optimizer.param_groups:
                group["params"] = [
                    parameter for parameter in group["params"] if id(parameter) in trainable
                ]
            optimized = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if optimized != trainable:
                raise RuntimeError("Optimizer tidak identik dengan parameter residual")
            if any(
                parameter.requires_grad and ".residual." not in name
                for name, parameter in model.named_parameters()
            ):
                raise RuntimeError("Optimizer dibangun saat parent masih trainable")
            return optimizer

        def optimizer_step(self):
            super().optimizer_step()
            target = unwrap_model(self.model)
            freeze_for_parent_residual(target)
            target.train(True)
            assert_parent_residual_runtime_state(target)
            if self.ema:
                sync_frozen_parent_to_ema(target, self.ema.ema)

        def preprocess_batch(self, batch):
            target = unwrap_model(self.model)
            freeze_for_parent_residual(target)
            target.train(True)
            if getattr(self, "_parent_residual_logged_epoch", None) != self.epoch:
                runtime = assert_parent_residual_runtime_state(target)
                print(
                    f"AF2-{frozen_residual.family.upper()}-{frozen_residual.conditioning} "
                    f"epoch {self.epoch + 1}/{self.epochs} runtime={runtime}",
                    flush=True,
                )
                self._parent_residual_logged_epoch = self.epoch
            return super().preprocess_batch(batch)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    AF2ParentResidualTrainer.__name__ = (
        f"AF2ParentResidualTrainer_{frozen_residual.family}_{frozen_residual.conditioning}"
    )
    return AF2ParentResidualTrainer
