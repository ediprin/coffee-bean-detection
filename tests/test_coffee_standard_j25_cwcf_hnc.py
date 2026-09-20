import ast
import json
from pathlib import Path

import pytest

from coffee_detector.experiments.run_coffee_standard_j25_cwcf_hnc import (
    ARM,
    PROTOCOL,
    TARGET_CLASS,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _result(protocol, values, target):
    classwise = {f"class-{index}": 0.5 for index in range(25)}
    classwise.pop("class-7")
    classwise[TARGET_CLASS] = target
    return {
        "protocol": protocol,
        "metrics": dict(zip(METRICS, values)),
        "map50_95_by_class": classwise,
        "test_images_accessed": False,
    }


def test_decision_passes_only_when_target_repaired_without_headline_damage(tmp_path):
    baseline = _result(
        "coffee-standard-j25-cwcf-seed42-v1", (0.636, 0.247, 0.015), 0.015
    )
    candidate = _result(PROTOCOL, (0.635, 0.245, 0.018), 0.025)
    baseline_path, candidate_path = tmp_path / "base.json", tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    result = build_decision(
        baseline_path, candidate_path, tmp_path / "decision.json"
    )
    assert result["decision"] == "PASS"
    assert result["next"] == "FREEZE_CONFIRMATION"
    assert result["target_delta"] == pytest.approx(0.01)
    assert result["test_opened"] is False


def test_notebook_is_resumable_and_test_locked():
    notebook = json.loads(
        (ROOT / "notebooks/Coffee_Standard_J25_CWCFHNC1_Seed42_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-cwcf-conditional-confusion'" in code
    assert "run_coffee_standard_j25_cwcf_hnc" in code
    assert f"ARM='{ARM}'" in code
    assert "--authorize-training" in code
    assert "last.pt tersimpan di Drive setiap epoch" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
