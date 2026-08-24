"""IGEM-confirmation trainer wrapper that keeps the frozen parent exact inside EMA checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics.utils.torch_utils import unwrap_model

from coffee_detector.afab import AFABConfig

from .config import AF2ParentResidualConfig
from .trainer import make_af2_parent_residual_trainer


def _sync_frozen_parent_to_ema(model: torch.nn.Module, ema_model: torch.nn.Module) -> None:
    """Copy every non-residual state item exactly into EMA, leaving residual EMA untouched."""

    live = unwrap_model(model).state_dict()
    ema = unwrap_model(ema_model).state_dict()
    if live.keys() != ema.keys():
        raise RuntimeError("Live/EMA state schema berbeda")
    with torch.no_grad():
        for key, value in live.items():
            if ".residual." not in key:
                ema[key].copy_(value.detach())


def make_af2_igem_confirmation_trainer(
    afab: AFABConfig | dict[str, Any],
    parent_residual: AF2ParentResidualConfig | dict[str, Any],
    *,
    initial_checkpoint: str | Path,
):
    """Wrap the frozen-parent trainer and force exact parent state in its EMA snapshots."""

    residual = AF2ParentResidualConfig.from_mapping(parent_residual)
    if residual.family != "igem":
        raise ValueError("Audited confirmation trainer hanya untuk IGEM")
    BaseTrainer = make_af2_parent_residual_trainer(
        afab, residual, initial_checkpoint=initial_checkpoint
    )

    class AF2IGEMAuditedTrainer(BaseTrainer):
        def _sync_parent_ema(self) -> None:
            if self.ema is not None:
                _sync_frozen_parent_to_ema(self.model, self.ema.ema)

        def _setup_train(self):
            super()._setup_train()
            self._sync_parent_ema()

        def optimizer_step(self):
            # Base implementation performs the residual optimizer step and then EMA.update(model).
            super().optimizer_step()
            # Restore the frozen AF2 part exactly; residual EMA remains the ordinary Ultralytics EMA.
            self._sync_parent_ema()

    AF2IGEMAuditedTrainer.__name__ = f"AF2IGEMAuditedTrainer_{residual.conditioning}"
    return AF2IGEMAuditedTrainer


__all__ = ["make_af2_igem_confirmation_trainer", "_sync_frozen_parent_to_ema"]
