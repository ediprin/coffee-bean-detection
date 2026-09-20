import ast
import json
from pathlib import Path

import pytest
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import METRICS
from coffee_detector.experiments.run_coffee_standard_j25_yolov8s_matched import (
    ARM,
    PROTOCOL,
    build_comparison,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v8s_matched_config_uses_exact_safeaug_schedule():
    arm = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8S_MATCHED.yaml").read_text(encoding="utf-8")
    )
    safeaug = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/SAFEAUG0.yaml").read_text(encoding="utf-8")
    )
    assert arm["dataset"] == safeaug["dataset"]
    assert arm["train"] == safeaug["train"]
    assert arm["sampler"] == "none"
    assert arm["model"] == "yolov8s.pt"
    assert arm["weights"] == "yolov8s.pt"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_comparison_is_descriptive_and_keeps_test_locked(tmp_path):
    v8s, safeaug, cwcf = (tmp_path / f"{name}.json" for name in ("v8s", "safe", "cwcf"))
    common = {"seed": 42, "test_images_accessed": False}
    _write(
        v8s,
        {
            **common,
            "protocol": PROTOCOL,
            "metrics": _metrics(.63, .22, .02),
            "target_class_map50_95": .02,
        },
    )
    _write(
        safeaug,
        {
            **common,
            "metrics": _metrics(.624, .218, .0186),
            "target_class_map50_95": .0186,
        },
    )
    _write(
        cwcf,
        {
            **common,
            "metrics": _metrics(.636, .247, .0149),
            # CWCF1's actual arm result stores classwise AP in this map,
            # not in a top-level target_class_map50_95 field.
            "map50_95_by_class": {"Biji Hitam Pecah": .0149},
        },
    )
    result = build_comparison(v8s, safeaug, cwcf, tmp_path / "summary.json")
    assert result["decision"] == "DESCRIPTIVE_BASELINE_COMPLETE"
    assert result["v8s_minus_safeaug0"]["macro_map50_95"] == pytest.approx(.006)
    assert result["cwcf1_minus_v8s"]["macro_map50_95"] == pytest.approx(.006)
    assert result["historical_reference"]["head_to_head_comparable"] is False
    assert result["test_opened"] is False


def test_protocol_and_notebook_freeze_seed42_without_test_access():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_YOLOV8S_MATCHED_PROTOCOL_2026-09-20.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "not a promotion/kill gate" in protocol
    assert "DESCRIPTIVE_BASELINE_COMPLETE" in protocol

    notebook = json.loads(
        (
            ROOT
            / "notebooks/Coffee_Standard_J25_YOLOv8s_MATCHED_Seed42_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8s-matched-baseline'" in code
    assert "run_coffee_standard_j25_yolov8s_matched" in code
    assert "--authorize-training" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
    assert ARM == "V8S_MATCHED"
