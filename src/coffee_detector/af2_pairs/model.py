from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from coffee_detector.afab import AFABConfig, AFABInputEnhancer
from coffee_detector.igem import IGEMConfig, IGEMDetectionModel, load_igem_detector_weights
from coffee_detector.safpn_alignment import (
    SAFPNAlignmentConfig,
    SAFPNAlignmentDetectionModel,
    load_safpn_alignment_detector_weights,
)
from coffee_detector.stb import STBConfig, STBDetectionModel, load_stb_weights


AF2_CONFIG = AFABConfig(
    mode="af2",
    patch_size=32,
    overlap=0.50,
    radius_ratio=0.05,
    gamma=0.10,
    angular_bins=360,
    chunk_size=128,
    eps=1e-8,
)


class _AF2FrontendMixin:
    """Apply the historical parameter-free AF2 operator before every tensor forward."""

    def _attach_af2(self, config: AFABConfig | Mapping[str, Any] | None) -> None:
        self.af2_config = AFABConfig.from_mapping(config or AF2_CONFIG)
        if self.af2_config.mode != "af2":
            raise ValueError("Pasangan ini dikunci ke AF2, bukan AF1/AF12")
        self.af2 = AFABInputEnhancer(self.af2_config)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "af2", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


class AF2STBDetectionModel(_AF2FrontendMixin, STBDetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, *, stb=None, af2=None):
        STBDetectionModel.__init__(self, cfg=cfg, ch=ch, nc=nc, verbose=verbose, stb=stb)
        self._attach_af2(af2)


class AF2IGEMDetectionModel(_AF2FrontendMixin, IGEMDetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, *, igem=None, af2=None):
        IGEMDetectionModel.__init__(self, cfg=cfg, ch=ch, nc=nc, verbose=verbose, igem=igem)
        self._attach_af2(af2)


class AF2SAFDetectionModel(_AF2FrontendMixin, SAFPNAlignmentDetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, *, safpn_alignment=None, af2=None):
        SAFPNAlignmentDetectionModel.__init__(
            self, cfg=cfg, ch=ch, nc=nc, verbose=verbose,
            safpn_alignment=safpn_alignment,
        )
        self._attach_af2(af2)


def _build_pair_model(arm: str, cfg, *, nc: int, ch: int, verbose: bool, strong: Mapping[str, Any], af2):
    if arm == "AF2STB1":
        return AF2STBDetectionModel(cfg, nc=nc, ch=ch, verbose=verbose, stb=STBConfig.from_mapping(strong), af2=af2)
    if arm == "AF2IGEM1":
        return AF2IGEMDetectionModel(cfg, nc=nc, ch=ch, verbose=verbose, igem=IGEMConfig.from_mapping(strong), af2=af2)
    if arm == "AF2SAF1":
        return AF2SAFDetectionModel(
            cfg, nc=nc, ch=ch, verbose=verbose,
            safpn_alignment=SAFPNAlignmentConfig.from_mapping(strong), af2=af2,
        )
    raise ValueError(f"Arm pasangan tidak dikenal: {arm}")


def _load_pair_weights(arm: str, model, weights) -> dict[str, int]:
    if arm == "AF2STB1":
        return load_stb_weights(model, weights)
    if arm == "AF2IGEM1":
        return load_igem_detector_weights(model, weights)
    if arm == "AF2SAF1":
        return load_safpn_alignment_detector_weights(model, weights)
    raise ValueError(f"Arm pasangan tidak dikenal: {arm}")


def make_af2_pair_trainer(
    arm: str,
    strong: Mapping[str, Any],
    *,
    af2: AFABConfig | Mapping[str, Any] | None = None,
    d0_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_af2 = AFABConfig.from_mapping(af2 or AF2_CONFIG)
    frozen_strong = dict(strong)
    bound = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class AF2PairTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(_build_pair_model(
                arm, cfg, nc=self.data["nc"], ch=self.data["channels"],
                verbose=verbose, strong=frozen_strong, af2=frozen_af2,
            ))
            if bound is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO
                transfer = _load_pair_weights(arm, model, YOLO(str(bound)).model)
            elif weights:
                transfer = _load_pair_weights(arm, model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"{arm} NATIVE D0 TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    AF2PairTrainer.__name__ = f"{arm}Trainer"
    return AF2PairTrainer
