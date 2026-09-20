import ast
import json
from pathlib import Path

import pytest
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_cwcf2 import (
    DIRECT_PROTOCOL,
    SAFEAUG_PROTOCOL,
    CWCF1_PROTOCOL,
    PROTOCOL,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def test_cwcf2_changes_only_explicit_composition_contract():
    cwcf2 = yaml.safe_load((ROOT / "configs/coffee_standard_j25/CWCF2.yaml").read_text())
    cwcf1 = yaml.safe_load((ROOT / "configs/coffee_standard_j25/CWCF1.yaml").read_text())
    safe = yaml.safe_load((ROOT / "configs/coffee_standard_j25/SAFEAUG0.yaml").read_text())
    assert cwcf2["model"] == cwcf1["model"] == safe["model"]
    assert cwcf2["train"] == cwcf1["train"] == safe["train"]
    for key in ("cue_channels", "attribute_gain", "wavelet_levels", "cue_clip"):
        assert cwcf2["cwcf"][key] == cwcf1["cwcf"][key]
    assert cwcf2["cwcf"]["explicit_composition"] is True
    assert cwcf2["cwcf"]["class_balanced_attributes"] is True
    assert cwcf2["sampler"] == "none"


def _write(path, protocol, values):
    path.write_text(json.dumps({
        "protocol": protocol,
        "metrics": dict(zip(METRICS, values)),
        "test_images_accessed": False,
    }))


def test_decision_compares_matched_controls_without_test(tmp_path):
    paths = [tmp_path / f"{index}.json" for index in range(4)]
    for path, protocol, values in zip(paths, (DIRECT_PROTOCOL, SAFEAUG_PROTOCOL, CWCF1_PROTOCOL, PROTOCOL), ((.60,.19,.02),(.62,.21,.018),(.636,.246,.014),(.64,.25,.02))):
        _write(path, protocol, values)
    result = build_decision(*paths, tmp_path / "decision.json")
    assert result["status"] == "DOMINATES_CWCF1"
    assert result["cwcf2_minus"]["CWCF1"]["macro_map50_95"] == pytest.approx(.004)
    assert result["test_opened"] is False


def test_decision_rejects_quarantined_concurrent_writer_result(tmp_path):
    paths = [tmp_path / f"{index}.json" for index in range(4)]
    for path, protocol, values in zip(paths, (DIRECT_PROTOCOL, SAFEAUG_PROTOCOL, CWCF1_PROTOCOL, PROTOCOL), ((.60,.19,.02),(.62,.21,.018),(.636,.246,.014),(.64,.25,.02))):
        _write(path, protocol, values)
    payload = json.loads(paths[-1].read_text())
    payload["valid_for_claims"] = False
    paths[-1].write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="quarantine"):
        build_decision(*paths, tmp_path / "decision.json")


def test_protocol_and_notebook_are_fresh_resumable_and_test_locked():
    protocol = (ROOT / "docs/COFFEE_STANDARD_J25_CWCF2_PROTOCOL_2026-09-19.md").read_text()
    assert "Status: **frozen before training**" in protocol
    assert "single fresh seed-42" in protocol
    notebook = json.loads((ROOT / "notebooks/Coffee_Standard_J25_CWCF2_Seed42_Colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    ast.parse(code)
    assert "BRANCH='codex/j25-chromatic-wavelet-composition'" in code
    assert "run_coffee_standard_j25_cwcf2" in code
    assert "--authorize-training" in code
    assert "quarantine" in code.lower()
    assert "--authorize-test" not in code and "test/images" not in code
