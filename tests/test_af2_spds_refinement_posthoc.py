from __future__ import annotations

import json
from pathlib import Path

from coffee_detector.analysis.af2_spds_refinement_posthoc import (
    paired_class_bootstrap,
    run_af2_spds_refinement_posthoc,
)


def _classes(offset: float = 0.0) -> dict[str, float]:
    return {f"class_{index:02d}": 0.60 + index * 0.01 + offset for index in range(21)}


def _write(root, arm: str, values: dict[str, float]) -> None:
    reports = root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    ordered = sorted(values.values())
    payload = {
        "arm": arm,
        "seed": 42,
        "metrics": {
            "macro_map50_95": sum(ordered) / len(ordered),
            "bottom3_class_map50_95": sum(ordered[:3]) / 3,
            "worst_class_map50_95": ordered[0],
            "map50_95_by_class": values,
            "classes_without_ground_truth": [],
        },
        "test_images_accessed": False,
    }
    (reports / f"{arm}_seed42_result.json").write_text(json.dumps(payload))


def test_paired_class_bootstrap_detects_uniform_gain() -> None:
    base = _classes()
    candidate = {name: value + 0.02 for name, value in base.items()}
    result = paired_class_bootstrap(base, candidate, iterations=500, seed=7)
    for row in result["metrics"].values():
        assert abs(row["point_delta"] - 0.02) < 1.0e-12
        assert row["ci95_low"] > 0.019
        assert row["probability_positive"] == 1.0
    assert result["unit"] == "validation_class_identity"


def test_posthoc_retains_pareto_status_without_overriding_formal_gate(tmp_path) -> None:
    original = tmp_path / "original"
    refinement = tmp_path / "refinement"
    base = _classes()
    spds = {name: value + 0.03 for name, value in base.items()}
    cue = {name: value + 0.04 for name, value in base.items()}
    decay = {name: value + 0.01 for name, value in base.items()}
    for arm, values in (("AF2BASE", base), ("AF2SPDS", spds)):
        _write(original, arm, values)
    for arm, values in (("AF2CUE1", cue), ("AF2DECAY1", decay)):
        _write(refinement, arm, values)

    result = run_af2_spds_refinement_posthoc(
        original, refinement, tmp_path / "posthoc.json", iterations=200, seed=11
    )
    assert result["formal_frozen_decision"] == "FAIL_KILL_GATE"
    assert result["exploratory_research_status"] == "RETAIN_PARETO_EXPLORATORY"
    assert result["class_summary"]["cue1_improved_vs_base"] == 21
    assert result["training_executed"] is False
    assert result["test_opened"] is False


def test_posthoc_colab_is_validation_only_and_compiles() -> None:
    path = Path("notebooks/Faruq_V3_AF2CUE1_Posthoc_Analysis_Colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:{index}", "exec")
    assert "run_af2_spds_refinement_posthoc" in code
    assert "print_json=False" in code
    assert "authorize-training" not in code
    assert "model.train" not in code
    assert "test/images" not in code
    assert "test/labels" not in code
