import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/breadth_screening/faruq_v3_batch_v1.json"
CONTROLLER = ROOT / "tools/run_faruq_v3_breadth_batch.py"
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Breadth_Screening_Batch_V1_Colab.ipynb"


def _controller_module():
    spec = importlib.util.spec_from_file_location("breadth_controller", CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_freezes_22_enabled_common_detectors_by_exact_sha():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    enabled = [item for item in payload["candidates"] if item.get("enabled") and item.get("group") == "detector"]
    assert len(enabled) == 22
    ids = [item["id"] for item in enabled]
    assert len(ids) == len(set(ids))
    for item in enabled:
        assert item["branch"].startswith("agent/")
        assert len(item["sha"]) == 40
        int(item["sha"], 16)
        assert item["runner_token"]


def test_manifest_canonical_gate_matches_frozen_breadth_rule():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    gate = payload["canonical_gate"]
    assert gate["macro_vs_d0ft_min"] == -0.01
    assert gate["bottom3_vs_d0ft_min"] == -0.02
    assert gate["worst_vs_d0ft_min"] == -0.02
    assert gate["discovery_signal_any"] == {
        "macro_vs_d0ft_min": 0.002,
        "bottom3_vs_d0ft_min": 0.005,
        "worst_vs_d0ft_min": 0.005,
    }
    assert payload["seed"] == 42
    assert payload["evaluation_split"] == "val"
    assert payload["primary_control"] == "D0FT"


def test_gds_and_dc2_are_not_in_common_detector_batch():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in payload["candidates"]}
    assert by_id["GDSC1"]["enabled"] is False
    assert by_id["GDSC1"]["group"] == "parked"
    for candidate_id in ("DC2A", "DC2B", "DC2C", "DC2D"):
        assert by_id[candidate_id]["enabled"] is False
        assert by_id[candidate_id]["group"] == "diagnostic"


def test_canonical_decision_requires_retention_and_a_discovery_signal():
    module = _controller_module()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    gate = payload["canonical_gate"]
    d0ft = {
        "macro_map50_95": 0.86,
        "bottom3_class_map50_95": 0.75,
        "worst_class_map50_95": 0.72,
    }
    positive = {
        "macro_map50_95": 0.863,
        "bottom3_class_map50_95": 0.751,
        "worst_class_map50_95": 0.721,
    }
    result = module.canonical_decision(positive, d0ft, gate)
    assert result["decision"] == "RETAIN"
    assert result["criteria"]["discovery_signal"] is True

    no_signal = {
        "macro_map50_95": 0.861,
        "bottom3_class_map50_95": 0.752,
        "worst_class_map50_95": 0.724,
    }
    result = module.canonical_decision(no_signal, d0ft, gate)
    assert result["decision"] == "REJECT"
    assert result["criteria"]["discovery_signal"] is False

    tail_drop = {
        "macro_map50_95": 0.870,
        "bottom3_class_map50_95": 0.755,
        "worst_class_map50_95": 0.699,
    }
    result = module.canonical_decision(tail_drop, d0ft, gate)
    assert result["decision"] == "REJECT"
    assert result["criteria"]["worst_retention"] is False


def test_extract_candidate_rows_prefers_candidate_and_excludes_controls():
    module = _controller_module()
    payload = {
        "controls": {
            "D0FT": {
                "macro_map50_95": 0.8,
                "bottom3_class_map50_95": 0.7,
                "worst_class_map50_95": 0.6,
            }
        },
        "candidate": {
            "A": {
                "macro_map50_95": 0.81,
                "bottom3_class_map50_95": 0.72,
                "worst_class_map50_95": 0.63,
            },
            "B": {
                "macro_map50_95": 0.82,
                "bottom3_class_map50_95": 0.73,
                "worst_class_map50_95": 0.64,
            },
        },
    }
    rows = module.extract_candidate_rows(payload, "X")
    assert [row["arm"] for row in rows] == ["A", "B"]


def test_runner_discovery_uses_token_and_rejects_ambiguity(tmp_path):
    module = _controller_module()
    root = tmp_path / "src/coffee_detector/experiments"
    root.mkdir(parents=True)
    (root / "run_faruq_v3_apcl_screening.py").write_text("# a")
    (root / "run_faruq_v3_safpn_screening.py").write_text("# b")
    assert module.discover_runner(tmp_path, "apcl").name == "run_faruq_v3_apcl_screening.py"
    assert module.discover_runner(tmp_path, "safpn").name == "run_faruq_v3_safpn_screening.py"


def test_master_notebook_has_three_complete_chunks_and_never_requests_test_split():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert "agent/breadth-screening-batch-v1" in source
    assert "run_faruq_v3_breadth_batch.py" in source
    assert "--continue-on-error" in source
    assert "CHUNK_ID = 1" in source
    for candidate_id in (
        "SG1", "SSCB", "MRL", "APCL1", "PCL1", "CPE", "BHCL", "HIERVIP",
        "SAF1", "DRNET", "DRIV", "SF1", "CF1", "SC1", "STB1", "IGEM", "PWCA",
        "FBNR", "SEMAUX", "CG1", "AFAB", "FTIF",
    ):
        assert candidate_id in source
    assert "split=test" not in source.lower()
    assert "--split test" not in source.lower()
