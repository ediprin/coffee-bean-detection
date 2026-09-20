import ast
import json
from pathlib import Path

import pytest
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import METRICS
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_matched import (
    ARM,
    PROTOCOL,
    build_comparison,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v8n_matched_config_uses_exact_safeaug_schedule():
    arm = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_MATCHED.yaml").read_text(encoding="utf-8")
    )
    safeaug = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/SAFEAUG0.yaml").read_text(encoding="utf-8")
    )
    v8s = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8S_MATCHED.yaml").read_text(encoding="utf-8")
    )
    assert arm["dataset"] == safeaug["dataset"] == v8s["dataset"]
    assert arm["train"] == safeaug["train"] == v8s["train"]
    assert arm["sampler"] == "none"
    assert arm["model"] == "yolov8n.pt"
    assert arm["weights"] == "yolov8n.pt"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_comparison_reports_capacity_and_family_deltas(tmp_path):
    v8n, v8s, safeaug, cwcf = (
        tmp_path / f"{name}.json" for name in ("v8n", "v8s", "safe", "cwcf")
    )
    common = {"seed": 42, "test_images_accessed": False}
    _write(
        v8n,
        {
            **common,
            "protocol": PROTOCOL,
            "metrics": _metrics(.68, .25, .025),
            "target_class_map50_95": .025,
        },
    )
    _write(
        v8s,
        {
            **common,
            "metrics": _metrics(.728, .284, .035),
            "target_class_map50_95": .035,
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
            "map50_95_by_class": {"Biji Hitam Pecah": .0149},
        },
    )
    result = build_comparison(v8n, v8s, safeaug, cwcf, tmp_path / "summary.json")
    assert result["decision"] == "CAPACITY_FAMILY_PROBE_COMPLETE"
    assert result["v8n_minus_safeaug0"]["macro_map50_95"] == pytest.approx(.056)
    assert result["v8n_minus_cwcf1"]["macro_map50_95"] == pytest.approx(.044)
    assert result["v8s_minus_v8n"]["macro_map50_95"] == pytest.approx(.048)
    assert result["target_values"]["CWCF1"] == pytest.approx(.0149)
    assert result["test_opened"] is False


def test_protocol_and_notebook_freeze_seed42_without_test_access():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_YOLOV8N_MATCHED_PROTOCOL_2026-09-20.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "CAPACITY_FAMILY_PROBE_COMPLETE" in protocol
    assert "No fixed numerical threshold" in protocol

    notebook = json.loads(
        (
            ROOT
            / "notebooks/Coffee_Standard_J25_YOLOv8n_MATCHED_Seed42_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8n-matched-baseline'" in code
    assert "run_coffee_standard_j25_yolov8n_matched" in code
    assert "V8S_RESULT" in code
    assert "--authorize-training" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
    assert ARM == "V8N_MATCHED"
