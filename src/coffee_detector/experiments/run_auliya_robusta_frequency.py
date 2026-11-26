"""Train one matched YOLO26n-seg arm on the prepared Auliya development split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer

from coffee_detector.analysis.auliya_robusta_preflight import (
    CLASS_COUNT, METADATA_SHA256, REUSED_OBJECT_CLASSES, _sha256,
)
from coffee_detector.auliya_frequency_model import ARMS, FrequencySegmentationModel
from coffee_detector.experiments.prepare_auliya_robusta_frequency import (
    ARCHIVE_SHA256,
    GROUPS,
    IMAGE_COUNTS,
    PROTOCOL,
)


SEED = 42
EPOCHS = 50
INITIAL_WEIGHTS = "yolo26n-seg.pt"
WEIGHTS_SHA256 = "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"
IMAGE_SIZE = 640
BATCH = 8
ULTRALYTICS_VERSION = "8.4.96"


def _trainer_for(arm: str) -> type[SegmentationTrainer]:
    class FrequencyTrainer(SegmentationTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(FrequencySegmentationModel(
                cfg, arm=arm, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose,
            ))
            if weights is not None:
                model.load(weights)
            return model

        def final_eval(self):
            # Preserve epoch, optimizer and EMA in last.pt, including at epoch 50.
            # The runner below evaluates the fixed last checkpoint once.
            pass

    return FrequencyTrainer


def _development_yaml(root: Path) -> Path:
    contract = json.loads((root / "development_contract.json").read_text(encoding="utf-8"))
    if (
        contract.get("protocol") != PROTOCOL
        or contract.get("archive_sha256") != ARCHIVE_SHA256
        or contract.get("metadata_sha256") != METADATA_SHA256
        or contract.get("groups") != {**GROUPS, "held_out": "A"}
        or contract.get("images") != IMAGE_COUNTS
        or len(contract.get("names", [])) != CLASS_COUNT
        or (root / "test").exists()
    ):
        raise ValueError("Expected the frozen B/C development bundle")
    data_yaml = root / "data.yaml"
    dataset = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if dataset != {"train": "train/images", "val": "val/images", "names": contract["names"]}:
        raise ValueError("Unexpected paths or classes in prepared data.yaml")
    for split, expected in IMAGE_COUNTS.items():
        images = {p.stem for p in (root / split / "images").glob("*.jpg")}
        labels = {p.stem for p in (root / split / "labels").glob("*.txt")}
        if len(images) != expected or images != labels:
            raise ValueError(f"Dataset image/label contract failed for {split}")
    return data_yaml


def _summarize_masks(mask, names: dict[int, str]) -> dict:
    by_class = {int(index): float(ap) for index, ap in zip(mask.ap_class_index, mask.ap, strict=True)}
    if set(by_class) != set(range(CLASS_COUNT)) or not all(np.isfinite(list(by_class.values()))):
        raise RuntimeError("Mask AP is missing for one or more validation classes")
    clean = sorted(ap for index, ap in by_class.items() if index not in REUSED_OBJECT_CLASSES)
    return {
        "mask_map50_95_18": float(np.mean(clean)),
        "bottom3_mask_ap50_95_18": float(np.mean(clean[:3])),
        "worst_mask_ap50_95_18": clean[0],
        "mask_map50_95_21": float(np.mean(list(by_class.values()))),
        "mask_ap_per_class": {names[index]: by_class[index] for index in sorted(by_class)},
    }


def _publish_file(source: Path, destination: Path) -> None:
    with NamedTemporaryFile(dir=destination.parent, prefix=".publishing-", delete=False) as stream:
        staged = Path(stream.name)
    try:
        shutil.copyfile(source, staged)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def _write_json(destination: Path, payload: dict) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
        staged = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    try:
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def _load_stage(last: Path, arm: str) -> tuple[int, dict]:
    checkpoint = YOLO(str(last), task="segment")
    state = checkpoint.ckpt
    args = state.get("train_args", {})
    epoch = state.get("epoch", -1) + 1
    expected = {"epochs": EPOCHS, "batch": BATCH, "imgsz": IMAGE_SIZE, "seed": SEED}
    if (
        getattr(checkpoint.model, "frequency_arm", None) != arm
        or not 1 <= epoch <= EPOCHS
        or state.get("optimizer") is None
        or any(args.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError(f"Checkpoint does not match the frozen {arm} schedule: {last}")
    return epoch, state


def _restore_results(path: Path, results: dict) -> None:
    if not results:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(results)
        writer.writerows(zip(*results.values(), strict=True))


def run_arm(
    arm: str, development: Path, output: Path, device: str,
    *, weights: Path = Path(INITIAL_WEIGHTS), input_bundle_sha256: str | None = None,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if not torch.cuda.is_available() or not device.isdecimal() or int(device) >= torch.cuda.device_count():
        raise RuntimeError("A valid CUDA device is required for this training run")
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS_VERSION}, found {ultralytics.__version__}")
    data = _development_yaml(development.expanduser().resolve())
    output = output.expanduser().resolve()
    weights = weights.expanduser().resolve()
    if _sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("Initial checkpoint differs from the official YOLO26n-seg asset")
    persistent = output / arm
    last = persistent / "weights" / "last.pt"
    marker = persistent / "run_contract.json"
    report_path = persistent / "validation_result.json"
    contract = {
        "protocol": PROTOCOL, "arm": arm, "seed": SEED, "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE, "batch": BATCH,
        "pretrained_weights_sha256": WEIGHTS_SHA256,
        "development_contract_sha256": _sha256(data.parent / "development_contract.json"),
        "input_bundle_sha256": input_bundle_sha256,
        "ultralytics_version": ULTRALYTICS_VERSION,
    }
    last.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != contract:
            raise RuntimeError(f"Existing run contract disagrees with this run: {marker}")
    elif last.exists() or report_path.exists():
        raise RuntimeError(f"Existing artifacts without a run contract: {persistent}")
    else:
        _write_json(marker, contract)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("run_contract") != contract
            or report.get("checkpoint_sha256") != _sha256(last)
            or report.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Existing validation report does not match the run: {report_path}")
        return report

    with TemporaryDirectory(prefix=f"auliya-{arm.lower()}-") as temp_dir:
        local = Path(temp_dir) / arm
        local_last = local / "weights" / "last.pt"
        local_last.parent.mkdir(parents=True)
        completed = 0
        if last.exists():
            shutil.copyfile(last, local_last)
            completed, state = _load_stage(local_last, arm)
            _restore_results(local / "results.csv", state.get("train_results", {}))
            del state
        print(f"{arm}: {completed}/{EPOCHS} epochs saved", flush=True)
        if completed < EPOCHS:
            _write_json(persistent / "progress.json", {"epoch": completed, "epochs": EPOCHS})
            model = YOLO(str(local_last if completed else weights), task="segment")

            def publish_epoch(trainer):
                if trainer.batch_size != BATCH or trainer.epochs != EPOCHS:
                    raise RuntimeError("Ultralytics changed a frozen training parameter")
                _publish_file(trainer.last, last)
                _publish_file(trainer.csv, persistent / "results.csv")
                _write_json(persistent / "progress.json", {"epoch": trainer.epoch + 1, "epochs": EPOCHS})

            model.add_callback("on_model_save", publish_epoch)
            model.train(
                trainer=_trainer_for(arm), data=str(data), project=str(local.parent), name=arm,
                save_dir=str(local), resume=str(local_last) if completed else False,
                epochs=EPOCHS, patience=0, seed=SEED, imgsz=IMAGE_SIZE, batch=BATCH,
                workers=2, deterministic=True, device=device, plots=False, val=True,
            )
            del model
        if _load_stage(local_last, arm)[0] != EPOCHS:
            raise RuntimeError("Training stopped before the fixed endpoint")
        _publish_file(local / "results.csv", persistent / "results.csv")
        _write_json(persistent / "progress.json", {"epoch": EPOCHS, "epochs": EPOCHS})
        final = YOLO(str(local_last), task="segment")
        validation = final.val(
            data=str(data), split="val", imgsz=IMAGE_SIZE, batch=BATCH, device=device,
            project=str(local), name="last_validation", plots=False,
        )
        metrics = _summarize_masks(validation.seg, validation.names)

    report = {
        "protocol": PROTOCOL, "arm": arm, "seed": SEED, "epochs": EPOCHS,
        "evaluation_split": "val", "checkpoint": str(last),
        "pretrained_weights_sha256": WEIGHTS_SHA256,
        "checkpoint_sha256": _sha256(last),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": str(torch.__version__),
        "device": torch.cuda.get_device_name(int(device)),
        "metrics": metrics,
        "run_contract": contract,
        "test_images_accessed": False,
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", type=Path, default=Path(INITIAL_WEIGHTS))
    parser.add_argument("--input-bundle-sha256")
    args = parser.parse_args()
    print(json.dumps(run_arm(
        args.arm, args.development, args.output, args.device,
        weights=args.weights, input_bundle_sha256=args.input_bundle_sha256,
    ), indent=2))


if __name__ == "__main__":
    main()
