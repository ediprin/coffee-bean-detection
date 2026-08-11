import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from coffee_detector.experiments import run_faruq_v3_acmc_locked_test as module
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path):
    test_root = tmp_path / "test_root"
    test_root.mkdir()
    (test_root / "data.yaml").write_text(
        "val: test/images\ntest: test/images\nnames: {0: bean}\nlocked_test_only: true\n",
        encoding="utf-8",
    )
    (test_root / "faruq_locked_test_manifest.json").write_text("[]", encoding="utf-8")
    eligibility = _write_json(
        tmp_path / "eligibility.json",
        {
            "format": "coffee_detector.faruq_locked_test_eligibility.v1",
            "decision": "PASS",
            "next_action": "AUTHORIZE_FROZEN_ACMC_TEST_INFERENCE",
            "training_executed": False,
            "inference_executed": False,
            "gates": {"all": True},
        },
    )
    confirmation = _write_json(
        tmp_path / "confirmation.json",
        {
            "protocol": "faruq-v3-acmc-paired-optimization-confirmation-v1",
            "decision": "PASS",
            "next_action": "AUTHORIZE_SINGLE_LOCKED_TEST_EVALUATION",
            "seeds": [42, 123, 2026],
            "evaluation_split": "val",
            "test_images_accessed": False,
        },
    )
    d0ft, acmc = [], []
    for seed in (42, 123, 2026):
        left = tmp_path / f"D0FT_seed{seed}.pt"
        right = tmp_path / f"ACMC1_seed{seed}.pt"
        left.write_bytes(f"d0ft-{seed}".encode())
        right.write_bytes(f"acmc-{seed}".encode())
        d0ft.append(left)
        acmc.append(right)
    return test_root, eligibility, confirmation, tuple(d0ft), tuple(acmc)


def test_locked_test_requires_explicit_authority(tmp_path: Path) -> None:
    test_root, eligibility, confirmation, d0ft, acmc = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="belum diotorisasi"):
        module.run_faruq_v3_acmc_locked_test(
            test_root, eligibility, confirmation, tmp_path / "output", d0ft, acmc
        )


def test_locked_test_aggregates_paired_frozen_checkpoints(tmp_path: Path, monkeypatch) -> None:
    test_root, eligibility, confirmation, d0ft, acmc = _fixture(tmp_path)

    def fake_evaluate(checkpoint, data_yaml, test_root, output, **kwargs):
        is_acmc = "ACMC1" in checkpoint.name
        base = 0.70 + (0.02 if is_acmc else 0.0)
        return {
            "metrics": {
                "macro_map50_95": base,
                "bottom3_class_map50_95": base - 0.1,
                "worst_class_map50_95": base - 0.15,
                "map50_95_by_class": {name: base for name in SNI21_CLASSES},
            },
            "prediction_observations": [],
        }

    monkeypatch.setattr(module, "_evaluate_checkpoint", fake_evaluate)
    result = module.run_faruq_v3_acmc_locked_test(
        test_root,
        eligibility,
        confirmation,
        tmp_path / "output",
        d0ft,
        acmc,
        authorize_test=True,
    )
    assert result["conclusion"] == "CONFIRMED"
    assert result["training_executed"] is False
    assert result["further_tuning_authorized"] is False
    assert result["aggregate"]["macro_map50_95"]["head_improved_seeds"] == 3
    assert result["aggregate"]["macro_map50_95"]["head_delta_mean"] == pytest.approx(0.02)


def test_parent_bootstrap_uses_paired_images_and_detects_positive_delta() -> None:
    target = {"class_id": 0, "xyxy": [0.0, 0.0, 10.0, 10.0]}
    correct = {"class_id": 0, "score": 0.9, "xyxy": [0.0, 0.0, 10.0, 10.0]}
    wrong = {"class_id": 0, "score": 0.9, "xyxy": [20.0, 20.0, 30.0, 30.0]}
    observations = {}
    for seed in (42, 123, 2026):
        observations[str(seed)] = {
            "D0FT": [
                {"image_name": "a.jpg", "targets": [target], "predictions": [wrong]},
                {"image_name": "b.jpg", "targets": [target], "predictions": [wrong]},
            ],
            "ACMC1": [
                {"image_name": "a.jpg", "targets": [target], "predictions": [correct]},
                {"image_name": "b.jpg", "targets": [target], "predictions": [correct]},
            ],
        }
    result = module._paired_parent_bootstrap(observations, iterations=10, seed=7)
    assert result["independent_parents"] == 2
    assert result["custom_macro_point_delta"] > 0.9
    assert result["probability_positive"] == 1.0


def test_runtime_yaml_adds_required_schema_keys_without_changing_locked_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "locked" / "data.yaml"
    source.parent.mkdir()
    source.write_text(
        yaml.safe_dump(
            {
                "path": str(source.parent),
                "val": "test/images",
                "test": "test/images",
                "names": {0: "bean"},
                "locked_test_only": True,
            }
        ),
        encoding="utf-8",
    )
    runtime = module._ultralytics_test_yaml(source, tmp_path / "output")
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    compatible = yaml.safe_load(runtime.read_text(encoding="utf-8"))
    assert "train" not in original
    assert compatible["train"] == "test/images"
    assert compatible["val"] == "test/images"
    assert compatible["test"] == "test/images"
    assert compatible["runtime_schema_alias_only"] is True


def test_saved_validation_predictions_are_reused_for_bootstrap(tmp_path: Path) -> None:
    test_root = tmp_path / "locked"
    images = test_root / "test/images"
    labels = test_root / "test/labels"
    predictions = tmp_path / "validation/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    predictions.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(images / "bean.jpg")
    (labels / "bean.txt").write_text("0 0.5 0.5 0.2 0.25\n", encoding="utf-8")
    (predictions / "bean.txt").write_text(
        "0 0.5 0.5 0.2 0.25 0.9\n", encoding="utf-8"
    )
    rows = module._collect_saved_prediction_observations(test_root, predictions)
    assert len(rows) == 1
    assert rows[0]["targets"][0]["xyxy"] == pytest.approx([40, 30, 60, 50])
    assert rows[0]["predictions"][0]["xyxy"] == pytest.approx([40, 30, 60, 50])
    assert rows[0]["predictions"][0]["score"] == pytest.approx(0.9)
