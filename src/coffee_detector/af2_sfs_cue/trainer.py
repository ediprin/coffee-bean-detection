from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from coffee_detector.afab.operator import AFABConfig

from .config import AF2SFSCUEConfig
from .model import AF2SFSCUEDetectionModel, canonical_native_state, load_af2_sfs_cue_weights


def _fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_af2_sfs_cue_direct_trainer(
    afab: AFABConfig | Mapping,
    sfs_cue: AF2SFSCUEConfig | Mapping,
    *,
    model_yaml: str | Path,
    pretrained_checkpoint: str | Path,
    seed: int,
    expected_native_fingerprint: str,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_afab = AFABConfig.from_mapping(afab)
    frozen_combo = AF2SFSCUEConfig.from_mapping(sfs_cue)
    model_yaml = str(Path(model_yaml).expanduser().resolve())
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()

    class AF2SFSCUEDirectTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            resumed = bool(getattr(self.args, "resume", False))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                model = AF2SFSCUEDetectionModel(
                    model_yaml,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    afab=frozen_afab,
                    sfs_cue=frozen_combo,
                )
            if resumed:
                if weights:
                    load_af2_sfs_cue_weights(model, weights)
                return self.set_model_names_for_load(model)

            from ultralytics import YOLO

            source = YOLO(str(checkpoint)).model
            load_af2_sfs_cue_weights(model, source)
            observed = _fingerprint(canonical_native_state(model))
            if observed != expected_native_fingerprint:
                raise RuntimeError(
                    "Initial native detector state berbeda dari static audit: "
                    f"{observed} != {expected_native_fingerprint}"
                )
            print("AF2-SFS-CUE DIRECT NATIVE INIT MATCH:", observed, flush=True)
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    return AF2SFSCUEDirectTrainer
