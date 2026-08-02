import json
import hashlib
import sys
import types
from pathlib import Path

import yaml

from coffee_detector.train import train_experiment


def test_explicit_weights_are_forwarded_even_when_yaml_disables_pretrained(
    tmp_path: Path, monkeypatch
) -> None:
    data = tmp_path / "data"
    for split in ("train", "val"):
        (data / split / "images").mkdir(parents=True)
        (data / split / "labels").mkdir(parents=True)
    (data / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(data),
                "train": "train/images",
                "val": "val/images",
                "names": {0: "bean"},
            }
        ),
        encoding="utf-8",
    )
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text("nc: 1\n", encoding="utf-8")
    weights = tmp_path / "d0.pt"
    weights.write_bytes(b"checkpoint-placeholder")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "code": "T0",
                "variant": "baseline",
                "model": str(model_yaml),
                "weights": str(weights),
                "train": {"epochs": 1, "pretrained": False},
            }
        ),
        encoding="utf-8",
    )

    events: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, reference):
            events["reference"] = reference

        def load(self, reference):
            events["loaded"] = reference
            return self

        def train(self, **kwargs):
            events["train"] = kwargs

        def add_callback(self, *_args, **_kwargs):
            raise AssertionError("callback tidak diharapkan")

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    run = train_experiment(
        config, data, tmp_path / "outputs", seed=42, weights_override=weights
    )
    assert Path(events["loaded"]).resolve() == weights.resolve()
    assert events["train"]["pretrained"] is True
    manifest = json.loads((run / "experiment_manifest.json").read_text())
    assert manifest["weights"] == str(weights)
    assert manifest["weights_override_sha256"] == hashlib.sha256(
        weights.read_bytes()
    ).hexdigest()
