"""Adapt Zenodo run_baseline_v4.py to a paired, validation-only screen."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.segment import SegmentationTrainer

from coffee_detector.analysis.auliya_robusta_preflight import _sha256
from coffee_detector.auliya_frequency_model import ARMS, FrequencySegmentationModel
from coffee_detector.auliya_mask2 import (
    MASK2_ARM, MASK2_PROTOCOL, CommonMaskGridValidator, HaarMask2Model, prototypes_at_grid,
)
from coffee_detector.auliya_rival_margin import RIVAL_ARM, RIVAL_CONTRACT, RIVAL_PROTOCOL, RivalMarginSegmentationModel
from coffee_detector.experiments.prepare_auliya_official_fold import PROTOCOL, validate_official_fold
from coffee_detector.experiments.run_auliya_robusta_frequency import (
    _publish_file, _restore_results, _summarize_masks, _write_json,
)

MODEL = "yolo26s-seg.pt"
WEIGHTS_URL = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{MODEL}"
WEIGHTS_SHA256 = "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4"
ULTRALYTICS_VERSION = "8.4.210"
SCREEN_ARMS = (*ARMS, MASK2_ARM, RIVAL_ARM)
ARM_PROTOCOLS = {MASK2_ARM: MASK2_PROTOCOL, RIVAL_ARM: RIVAL_PROTOCOL}
SCHEDULE = {
    "epochs": 80, "patience": 25, "imgsz": 640, "batch": 16, "workers": 4, "seed": 0,
    "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005, "cos_lr": True,
    "hsv_h": 0.015, "hsv_s": 0.4, "hsv_v": 0.25, "translate": 0.1, "scale": 0.2,
    "perspective": 0.0005, "mosaic": 0.3, "fliplr": 0.5, "mixup": 0.0,
}


def _schedule_for(arm: str) -> dict:
    return {**SCHEDULE, "mask_ratio": 2} if arm == MASK2_ARM else SCHEDULE


def trainer_for(arm: str) -> type[SegmentationTrainer]:
    if arm not in SCREEN_ARMS:
        raise ValueError(f"Unknown arm: {arm}")

    class OfficialTrainer(SegmentationTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if arm == "RGB":
                return super().get_model(cfg, weights, verbose)
            params = {"nc": self.data["nc"], "ch": self.data["channels"], "verbose": verbose}
            if arm == MASK2_ARM:
                model = HaarMask2Model(cfg, **params)
            elif arm == RIVAL_ARM:
                model = RivalMarginSegmentationModel(cfg, **params)
            else:
                model = FrequencySegmentationModel(cfg, arm=arm, **params)
            model = self.set_model_names_for_load(model)
            if weights is not None:
                model.load(weights)
            return model

        def build_dataset(self, img_path, mode="train", batch=None):
            if arm != MASK2_ARM or mode == "train":
                return super().build_dataset(img_path, mode, batch)
            # Select best.pt on the same polygon raster as the historical HAAR arm.
            args = copy(self.args)
            args.mask_ratio = 4
            stride = max(int(self.model.stride.max()), 32)
            return build_yolo_dataset(args, img_path, batch, self.data, mode=mode, rect=True, stride=stride)

        def get_validator(self):
            if arm != MASK2_ARM:
                return super().get_validator()
            args = copy(self.args)
            args.mask_ratio = 4
            return CommonMaskGridValidator(self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks)

        def final_eval(self):
            # Preserve optimizer and stopping state for interruption recovery. The
            # runner evaluates the selected best checkpoint on validation below.
            pass

    return OfficialTrainer


def _ensure_runtime(weights: Path) -> None:
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS_VERSION}, found {ultralytics.__version__}")
    if _sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("Pretrained asset does not match the official YOLO26s-seg SHA-256")


def preflight(development: Path, weights: Path) -> dict:
    """CPU-safe; initialize from the official asset and verify paired weights without training."""
    _ensure_runtime(weights)
    data = validate_official_fold(development.resolve())
    contract = json.loads((data.parent / "official_fold_contract.json").read_text(encoding="utf-8"))
    native = YOLO(str(weights), task="segment").model
    models = {}
    for arm in ARMS:
        trainer = object.__new__(trainer_for(arm))
        trainer.data = {"nc": 21, "channels": 3, "names": dict(enumerate(contract["names"]))}
        trainer.args = SimpleNamespace(cls_remap=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SCHEDULE["seed"])
            models[arm] = trainer.get_model(native.yaml, native, verbose=False).eval()
    baseline = models["RGB"].state_dict()
    same = {}
    for arm, candidate in models.items():
        state = candidate.state_dict()
        same[arm] = state.keys() == baseline.keys() and all(
            torch.equal(state[key], tensor) for key, tensor in baseline.items()
        )
    sample = torch.rand((1, 3, 96, 96), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        outputs = {arm: model(sample)[0][0] for arm, model in models.items()}
    active = {arm: not torch.equal(outputs[arm], outputs["RGB"]) for arm in ("AF2", "HAAR")}
    if not all(same.values()) or not all(active.values()) or not all(x.isfinite().all() for x in outputs.values()):
        raise RuntimeError(f"Paired frontend audit failed: state={same}, active={active}")
    return {"protocol": PROTOCOL, "fold": "A", "held_out": "A", "images": contract["images"],
            "same_initial_state": same, "frontend_active": active,
            "test_images_accessed": False, "training_executed": False}


def preflight_mask2(development: Path, weights: Path) -> dict:
    """Test exact initialization, real YOLO mask loss, and a serialized inference round-trip."""
    _ensure_runtime(weights)
    data = validate_official_fold(development.resolve())
    dataset = json.loads((data.parent / "official_fold_contract.json").read_text(encoding="utf-8"))
    source = YOLO(str(weights), task="segment").model
    models = []
    for arm in ("HAAR", MASK2_ARM):
        trainer = object.__new__(trainer_for(arm))
        trainer.data = {"nc": 21, "channels": 3, "names": dict(enumerate(dataset["names"]))}
        trainer.args = SimpleNamespace(cls_remap=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SCHEDULE["seed"])
            models.append(trainer.get_model(source.yaml, source, verbose=False).eval())
    base, fine = models
    baseline, candidate = base.state_dict(), fine.state_dict()
    added = set(candidate) - set(baseline)
    changed = [k for k, v in baseline.items() if k not in candidate or not torch.equal(v, candidate[k])]
    if changed or len(added) != 1 or not next(iter(added)).endswith(".proto.detail.weight"):
        raise RuntimeError(f"Mask2 altered native tensors: new={added}; changed={changed}")
    input_hw = (96, 128)
    x = torch.rand((2, 3, *input_hw), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        det_base, proto_base = base(x)[0]
        det_fine, proto_fine = fine(x)[0]
    coarse = prototypes_at_grid(proto_fine, input_hw, 4)
    if (not torch.equal(det_base, det_fine)
            or proto_fine.shape[-2:] != tuple(n // 2 for n in input_hw)
            or not torch.equal(proto_base, coarse)):
        raise RuntimeError(f"Mask2 initialization differs: box/score/coeff max abs="
            f"{(det_base-det_fine).abs().max().item()}, proto max abs={(proto_base-coarse).abs().max().item()}, "
            f"proto shape={tuple(proto_fine.shape)}")
    with TemporaryDirectory(prefix="mask2-preflight-") as temp:
        path = Path(temp) / "roundtrip.pt"
        torch.save({"model": fine, "train_args": {"task": "segment"}}, path)
        checkpoint = YOLO(str(path), task="segment")
        if checkpoint.task != "segment":
            raise RuntimeError("Reloaded checkpoint is not a segmentation model")
        model = checkpoint.model.eval()
        with torch.inference_mode():
            det_reload, proto_reload = model(x)[0]
        if not torch.equal(det_fine, det_reload) or not torch.equal(proto_fine, proto_reload):
            raise RuntimeError("Checkpoint round-trip changed Mask2 outputs")
    # No dataset image and no optimizer step. Exercise the complete official
    # E2E loss with synthetic positive instances on the actual stride-2 grid.
    fine.args = get_cfg(overrides={"task": "segment", "mask_ratio": 2})
    fine.train()
    masks = torch.zeros((2, 48, 64), dtype=torch.uint8)
    masks[:, 12:36, 16:48] = 1
    batch = {"img": x, "batch_idx": torch.tensor([0, 1]), "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[.5, .5, .5, .5], [.5, .5, .5, .5]]), "masks": masks,
        "sem_masks": masks.clone()}
    loss, _ = fine(batch)
    if not torch.isfinite(loss).all():
        raise RuntimeError("Official segmentation loss is not finite")
    loss.sum().backward()
    grad = fine.model[-1].proto.detail.weight.grad
    if grad is None or not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
        raise RuntimeError("The higher-resolution mask loss did not reach the detail branch")
    return {"protocol": MASK2_PROTOCOL, "native_state_exact": True, "raw_detections_exact": True,
        "common_grid4_initial_prototypes_exact": True, "prototype_stride": 2,
        "added_parameters": candidate[next(iter(added))].numel(), "synthetic_loss_finite": True,
        "subpixel_gradient_nonzero": True, "checkpoint_reload_exact": True,
        "training_executed": False, "test_images_accessed": False}


def _validate_resume(checkpoint, arm: str) -> int:
    args = checkpoint.ckpt["train_args"]
    epoch = checkpoint.ckpt["epoch"] + 1
    if (getattr(checkpoint.model, "frequency_arm", "RGB") != arm
            or not 1 <= epoch <= SCHEDULE["epochs"]
            or checkpoint.ckpt.get("optimizer") is None
            or any(args.get(k) != v for k, v in _schedule_for(arm).items())):
        raise RuntimeError("Resume checkpoint does not match the official schedule or arm")
    return epoch


def _require_reference(folder: Path, treatment_contract: dict, arm: str) -> str:
    reference = json.loads((folder / "validation_result.json").read_text(encoding="utf-8"))
    expected = {**treatment_contract, "arm": arm, "protocol": PROTOCOL, "schedule": SCHEDULE}
    if (reference.get("run_contract") != expected or reference.get("arm") != arm
            or reference.get("evaluation_split") != "val" or reference.get("test_images_accessed") is not False
            or reference.get("checkpoint_sha256") != _sha256(folder / "weights" / "best.pt")):
        raise RuntimeError(f"{arm} baseline and treatment must share dataset, training recipe, GPU and software versions")
    return reference["checkpoint_sha256"]


def run_arm(arm: str, development: Path, weights: Path, output: Path, device: str,
            *, haar_reference: Path | None = None, rgb_reference: Path | None = None) -> dict:
    if arm not in SCREEN_ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if not device.isdecimal() or int(device) >= torch.cuda.device_count():
        raise RuntimeError("Training requires an available CUDA device")
    _ensure_runtime(weights)
    data = validate_official_fold(development.resolve())
    persistent = output.resolve() / arm
    schedule = _schedule_for(arm)
    contract = {
        "protocol": ARM_PROTOCOLS.get(arm, PROTOCOL),
        "arm": arm, "fold": "A", "model": MODEL, "schedule": schedule,
        "pretrained_sha256": WEIGHTS_SHA256, "ultralytics": ULTRALYTICS_VERSION,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(int(device)),
        "development_contract_sha256": _sha256(data.parent / "official_fold_contract.json"),
    }
    if arm == MASK2_ARM:
        if haar_reference is None:
            raise ValueError("HAAR_MASK2 requires the existing HAAR run directory")
        contract["haar_reference_checkpoint_sha256"] = _require_reference(haar_reference, contract, "HAAR")
    if arm == RIVAL_ARM:
        if rgb_reference is None:
            raise ValueError(f"{arm} requires the existing RGB run directory")
        contract["rgb_reference_checkpoint_sha256"] = _require_reference(rgb_reference, contract, "RGB")
        from coffee_detector.experiments.audit_auliya_rival_margin import preflight_rival_margin

        contract["classification_objective"] = RIVAL_CONTRACT
        preflight_rival_margin(development, weights, device=f"cuda:{device}")
    last = persistent / "weights" / "last.pt"
    best = persistent / "weights" / "best.pt"
    marker = persistent / "run_contract.json"
    progress = persistent / "progress.json"
    result = persistent / "validation_result.json"
    last.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("The run contract changed; use a new output directory")
    elif last.exists() or best.exists() or result.exists() or progress.exists():
        raise RuntimeError("Unidentified output artifacts; use a new output directory")
    else:
        _write_json(marker, contract)
    if result.exists():
        report = json.loads(result.read_text(encoding="utf-8"))
        if (report.get("run_contract") != contract or report.get("checkpoint_sha256") != _sha256(best)
                or report.get("evaluation_split") != "val" or report.get("test_images_accessed") is not False):
            raise RuntimeError("Existing result failed its run/checkpoint contract")
        return report

    with TemporaryDirectory(prefix="auliya-official-") as temp:
        local = Path(temp) / arm
        local_last = local / "weights" / "last.pt"
        local_best = local / "weights" / "best.pt"
        local_last.parent.mkdir(parents=True)
        state = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else {}
        if last.exists() != bool(state) or best.exists() != bool(state):
            raise RuntimeError("Incomplete checkpoint publication; recover a consistent checkpoint pair first")
        if state:
            for source, dest in ((last, local_last), (best, local_best)):
                if _sha256(source) != state[f"{source.stem}_sha256"]:
                    raise RuntimeError("Resume checkpoint digest differs from published progress")
                shutil.copyfile(source, dest)
            resume_model = YOLO(str(local_last), task="segment")
            completed = _validate_resume(resume_model, arm)
            if completed != state["epoch"]:
                raise RuntimeError("Checkpoint epoch differs from progress marker")
            _restore_results(local / "results.csv", resume_model.ckpt.get("train_results", {}))
        else:
            completed = 0
            resume_model = None
        print(f"{arm} fold A seed 0: {completed}/80 epochs saved", flush=True)
        if not state.get("training_complete", False):
            model = resume_model if resume_model is not None else YOLO(str(weights), task="segment")
            best_digest = state.get("best_sha256")

            def publish(trainer):
                nonlocal best_digest
                if any(getattr(trainer.args, k) != v for k, v in schedule.items()):
                    raise RuntimeError("Trainer changed the paired schedule")
                _publish_file(trainer.last, last)
                if trainer.best_fitness == trainer.fitness or best_digest is None:
                    _publish_file(trainer.best, best)
                    best_digest = _sha256(trainer.best)
                _publish_file(trainer.csv, persistent / "results.csv")
                _write_json(progress, {"epoch": trainer.epoch + 1, "maximum_epochs": SCHEDULE["epochs"],
                    "training_complete": bool(trainer.stop), "last_sha256": _sha256(trainer.last), "best_sha256": best_digest})

            model.add_callback("on_model_save", publish)
            model.train(trainer=trainer_for(arm), data=str(data), save_dir=str(local),
                resume=str(local_last) if completed else False, device=device, plots=False, **schedule)
        final_state = json.loads(progress.read_text(encoding="utf-8"))
        if not final_state["training_complete"] or _sha256(local_best) != final_state["best_sha256"]:
            raise RuntimeError("Training has not produced a finalized best checkpoint")
        model = YOLO(str(local_best), task="segment")
        if getattr(model.model, "frequency_arm", "RGB") != arm:
            raise RuntimeError("Best checkpoint arm differs from the requested arm")
        evaluation = {"validator": CommonMaskGridValidator, "mask_ratio": 4} if arm == MASK2_ARM else {}
        evaluated = model.val(data=str(data), split="val", imgsz=SCHEDULE["imgsz"],
            batch=SCHEDULE["batch"], workers=SCHEDULE["workers"], device=device,
            project=str(local), name="best_validation", plots=False, **evaluation)
        metrics = _summarize_masks(evaluated.seg, evaluated.names)
    report = {"arm": arm, "completed_epochs": final_state["epoch"], "evaluation_split": "val",
        "metrics": metrics, "run_contract": contract, "checkpoint_sha256": final_state["best_sha256"],
        "test_images_accessed": False}
    _write_json(result, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=SCREEN_ARMS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--haar-reference", type=Path)
    parser.add_argument("--rgb-reference", type=Path)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    if args.authorize_training:
        if args.arm is None or args.output is None:
            parser.error("training requires --arm and --output")
        value = run_arm(args.arm, args.development, args.weights, args.output, args.device,
            haar_reference=args.haar_reference, rgb_reference=args.rgb_reference)
    elif args.arm == RIVAL_ARM:
        from coffee_detector.experiments.audit_auliya_rival_margin import preflight_rival_margin

        value = preflight_rival_margin(args.development, args.weights, device=args.device)
    else:
        value = (preflight_mask2 if args.arm == MASK2_ARM else preflight)(args.development, args.weights)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
