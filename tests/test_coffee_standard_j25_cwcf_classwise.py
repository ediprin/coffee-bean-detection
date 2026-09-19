import json

import pytest

from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES
from coffee_detector.analysis.coffee_standard_j25_cwcf_classwise import (
    run_cwcf_classwise_audit,
)


def _report(path, offset, target):
    values = {name: 0.40 + index * 0.001 + offset for index, name in enumerate(J25_CLASSES)}
    values["Biji Hitam Pecah"] = target
    payload = {
        "split": "val",
        "metrics": {
            "map50_95_by_class": values,
            "classes_without_ground_truth": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_classwise_audit_reports_target_groups_and_test_lock(tmp_path):
    d0, safe, cwcf = (tmp_path / f"{name}.json" for name in ("d0", "safe", "cwcf"))
    _report(d0, 0.0, 0.02)
    _report(safe, 0.01, 0.018)
    _report(cwcf, 0.03, 0.015)
    result = run_cwcf_classwise_audit(d0, safe, cwcf, tmp_path / "result.json")
    assert len(result["classwise"]) == 25
    assert len(result["attribute_groups"]) == 14
    assert result["target_values"]["CWCF1"] == pytest.approx(0.015)
    assert result["black_broken_conjunction"]["SAFEAUG0"]["target_delta"] == pytest.approx(-0.003)
    assert result["training_executed"] is False
    assert result["test_opened"] is False
