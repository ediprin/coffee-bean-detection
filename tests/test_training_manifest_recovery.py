import csv
import json
from pathlib import Path

import yaml

from coffee_detector.train import recover_completed_training_manifest


def _write_config(path: Path, epochs: int = 3) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "code": "D0",
                "variant": "baseline",
                "model": "yolo26n.pt",
                "train": {"epochs": epochs},
            }
        ),
        encoding="utf-8",
    )


def _write_dataset(root: Path) -> None:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "names": {0: "bean"},
            }
        ),
        encoding="utf-8",
    )


def _write_results(path: Path, epochs: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP50-95(B)"])
        writer.writeheader()
        for epoch in range(epochs):
            writer.writerow({"epoch": epoch, "metrics/mAP50-95(B)": 0.5})


def test_recover_manifest_only_after_all_epochs_survive(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    data = tmp_path / "data"
    run = tmp_path / "run"
    _write_config(config, epochs=3)
    _write_dataset(data)
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"best")
    (run / "weights" / "last.pt").write_bytes(b"last")

    _write_results(run / "results.csv", epochs=2)
    assert not recover_completed_training_manifest(config, data, run, seed=42)
    assert not (run / "experiment_manifest.json").exists()

    _write_results(run / "results.csv", epochs=3)
    assert recover_completed_training_manifest(config, data, run, seed=42)
    manifest = json.loads((run / "experiment_manifest.json").read_text())
    assert manifest["completed_epochs"] == 3
    assert manifest["recovered_after_runtime_disconnect"] is True
