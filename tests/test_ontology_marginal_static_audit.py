from pathlib import Path

from coffee_detector.analysis.ontology_marginal_static_audit import (
    audit_ontology_marginal_static,
)


def test_static_audit_proves_zero_parameter_identical_inference(tmp_path: Path) -> None:
    result = audit_ontology_marginal_static(tmp_path / "audit.json")
    assert result["decision"] == "PASS"
    assert all(result["gates"].values())
    assert len({row["parameters"] for row in result["models"].values()}) == 1
    assert result["maximum_inference_difference"] == {
        "C0_vs_D0": 0.0,
        "S0_vs_D0": 0.0,
    }
    assert result["training_executed"] is False
    assert result["dataset_accessed"] is False
    assert result["test_images_accessed"] is False
    assert result["training_authorized"] is False
