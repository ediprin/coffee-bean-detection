import json
from pathlib import Path

import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    AF2_CONFIG,
    NATIVE_CONFIG,
    build_decision,
    validate_j25_development,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_configs_are_matched_and_protocol_is_frozen() -> None:
    native = yaml.safe_load(NATIVE_CONFIG.read_text())
    af2 = yaml.safe_load(AF2_CONFIG.read_text())
    assert native["model"] == af2["model"]
    assert native["train"] == af2["train"]
    assert native["train"]["epochs"] == 50
    protocol = (NATIVE_CONFIG.parents[2] / "docs/COFFEE_STANDARD_J25_AF2_DIRECT_PROTOCOL_2026-09-18.md").read_text()
    assert "Status: **frozen before training**" in protocol
    assert "Only train and validation representatives are extracted" in protocol


def test_development_contract_rejects_test_and_accepts_matched_provenance(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir()
    (root / "data.yaml").write_text(
        yaml.safe_dump({"train": "train/images", "val": "val/images", "names": list(range(25))})
    )
    contract = _write(
        tmp_path / "contract.json",
        {
            "format": "coffee_detector.coffee_standard_j25_source_split.v1",
            "decision": "PASS",
            "source_archive_sha256": "abc",
            "images": {"train": 315, "val": 68},
            "test_images_extracted": False,
        },
    )
    provenance = _write(
        tmp_path / "provenance.json",
        {
            "decision": "PASS_THESIS_LINEAGE_WITH_ONE_ANNOTATION_DISCREPANCY",
            "source_archive_sha256": "abc",
            "test_images_accessed_by_model": False,
        },
    )
    result = validate_j25_development(root, contract, provenance)
    assert all(result["gates"].values())


def test_decision_promotes_tail_pareto_route(tmp_path: Path) -> None:
    reports = tmp_path / "runs" / "val_reports"
    base = {"macro_map50_95": 0.80, "bottom3_class_map50_95": 0.70, "worst_class_map50_95": 0.65}
    candidate = {"macro_map50_95": 0.799, "bottom3_class_map50_95": 0.72, "worst_class_map50_95": 0.67}
    for arm, metrics in (("D0DIRECT", base), ("AF2DIRECT", candidate)):
        _write(
            reports / f"{arm}_seed42_result.json",
            {"arm": arm, "seed": 42, "metrics": metrics, "test_images_accessed": False},
        )
    result = build_decision(tmp_path / "runs", tmp_path / "decision.json")
    assert result["criteria"]["lower_tail_route"] is True
    assert result["decision"] == "PROMOTE_TO_PAIRED_3_SEED"
    assert result["test_opened"] is False
