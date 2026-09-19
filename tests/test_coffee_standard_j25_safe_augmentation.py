import ast
import json
from pathlib import Path

import pytest
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_strength_sweep import (
    PROTOCOL as STRENGTH_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_safe_augmentation import (
    DIRECT_PROTOCOL,
    METRICS,
    PROTOCOL,
    SAFE_D0_PROTOCOL,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def test_safeaug_config_keeps_safe_augmentation_and_removes_sampler():
    arm = yaml.safe_load((ROOT / "configs/coffee_standard_j25/SAFEAUG0.yaml").read_text())
    safe_d0 = yaml.safe_load((ROOT / "configs/coffee_standard_j25/SAFED0.yaml").read_text())
    assert arm["model"] == safe_d0["model"]
    assert arm["train"] == safe_d0["train"]
    assert arm["sampler"] == "none"
    assert arm["frontend"] == "none"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_decision_reports_sampler_effect_and_target_recovery(tmp_path):
    direct, safe_d0, strength, arm = (tmp_path / f"{name}.json" for name in range(4))
    _write(direct, {"protocol": DIRECT_PROTOCOL, "metrics": _metrics(.60, .20, .02), "test_images_accessed": False})
    _write(safe_d0, {"protocol": SAFE_D0_PROTOCOL, "metrics": _metrics(.67, .27, 0), "target_class_map50_95": 0, "test_images_accessed": False})
    _write(strength, {"protocol": STRENGTH_PROTOCOL, "values": {"0p00": {**_metrics(.66, .28, 0), "target_class_map50_95": 0}}, "test_opened": False})
    _write(arm, {"protocol": PROTOCOL, "metrics": _metrics(.66, .25, .03), "target_class_map50_95": .03, "test_images_accessed": False})
    result = build_decision(direct, safe_d0, strength, arm, tmp_path / "decision.json")
    assert result["decision"] == "SAMPLER_IMPLICATED_IN_TARGET_COLLAPSE"
    assert result["target_recovered_without_sampler"] is True
    assert result["sampler_effect_safe_d0_minus_safeaug0"]["macro_map50_95"] == pytest.approx(.01)
    assert result["test_opened"] is False


def test_protocol_and_notebook_freeze_one_resumable_arm():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_SAFE_AUGMENTATION_CONTROL_PROTOCOL_2026-09-19.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "removes identity repeat sampling" in protocol

    notebook = json.loads(
        (ROOT / "notebooks/Coffee_Standard_J25_SAFEAUG0_Seed42_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-safe-augmentation-control'" in code
    assert "run_coffee_standard_j25_safe_augmentation" in code
    assert "--authorize-training" in code
    assert "last.pt tersimpan di Drive setiap epoch" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code

