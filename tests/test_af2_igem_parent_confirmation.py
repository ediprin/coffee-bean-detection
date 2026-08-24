import json
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_af2_igem_parent_arm import (
    run_faruq_v3_af2_igem_parent_arm,
)
from coffee_detector.experiments.run_faruq_v3_af2_igem_parent_decision import (
    run_igem_parent_decision,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)


def _result(arm: str, conditioning: str, seed: int, parent, values):
    per_class = {f"class_{index}": 0.8 for index in range(21)}
    return {
        "format": "coffee_detector.af2_parent_residual.igem_arm_result.v1",
        "protocol": "faruq-v3-af2fs-igem-parent-confirmation-v1",
        "arm": arm,
        "family": "igem",
        "conditioning": conditioning,
        "seed": seed,
        "baseline_metrics": dict(parent, map50_95_by_class=per_class),
        "metrics": dict(values, map50_95_by_class=per_class),
        "initial_af2_checkpoint_sha256": f"parent-{seed}",
        "parent_frozen": True,
        "trainable_scope": "igem_residual_only",
        "evaluation_split": "val",
        "test_images_accessed": False,
    }


def _write_triplet(tmp_path: Path, controls, candidates):
    control_paths, candidate_paths = [], []
    for seed, control, candidate in zip(SEEDS, controls, candidates):
        cp = tmp_path / f"AF2IGEM0_seed{seed}_result.json"
        fp = tmp_path / f"AF2IGEM1_seed{seed}_result.json"
        cp.write_text(json.dumps(control), encoding="utf-8")
        fp.write_text(json.dumps(candidate), encoding="utf-8")
        control_paths.append(cp)
        candidate_paths.append(fp)
    return control_paths, candidate_paths


def test_three_seed_decision_accepts_tail_pareto_route(tmp_path: Path):
    parent = dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770)
    controls, candidates = [], []
    for seed in SEEDS:
        controls.append(_result(
            "AF2IGEM0", "zero", seed, parent,
            dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770),
        ))
        candidates.append(_result(
            "AF2IGEM1", "feature", seed, parent,
            dict(macro_map50_95=0.8795, bottom3_class_map50_95=0.796, worst_class_map50_95=0.781),
        ))
    control_paths, candidate_paths = _write_triplet(tmp_path, controls, candidates)
    result = run_igem_parent_decision(control_paths, candidate_paths, tmp_path / "decision.json")
    assert result["decision"] == "RETAIN"
    assert result["criteria"]["tail_pareto_route_pass"] is True
    assert result["test_opened"] is False


def test_three_seed_decision_rejects_flat_information_pair(tmp_path: Path):
    parent = dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770)
    controls, candidates = [], []
    for seed in SEEDS:
        values = dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770)
        controls.append(_result("AF2IGEM0", "zero", seed, parent, values))
        candidates.append(_result("AF2IGEM1", "feature", seed, parent, values))
    control_paths, candidate_paths = _write_triplet(tmp_path, controls, candidates)
    result = run_igem_parent_decision(control_paths, candidate_paths, tmp_path / "decision.json")
    assert result["decision"] == "REJECT"
    assert not result["criteria"]["superiority_route_pass"]
    assert not result["criteria"]["tail_pareto_route_pass"]


def test_runner_rejects_unapproved_training_before_artifact_access(tmp_path: Path):
    try:
        run_faruq_v3_af2_igem_parent_arm(
            "AF2IGEM1", tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, seed=42
        )
    except RuntimeError as error:
        assert "belum diotorisasi" in str(error)
    else:
        raise AssertionError("Runner lolos tanpa authorization")


def test_protocol_and_kaggle_notebook_are_frozen_and_test_locked():
    protocol = (ROOT / "docs/FARUQ_V3_AF2FS_IGEM_PARENT_CONFIRMATION_PROTOCOL_2026-08-24.md").read_text(encoding="utf-8")
    assert "FROZEN BEFORE TRAINING" in protocol
    assert "AF2IGEM0/1 x seeds 42,123,2026" in protocol
    assert "Locked test: **closed**" in protocol

    notebook = ROOT / "notebooks/Faruq_V3_AF2FS_IGEM_Parent_Confirmation_All_Seeds_Kaggle.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(notebook), "exec")
    assert "AF2IGEM0" in source and "AF2IGEM1" in source
    assert "SEEDS=(42,123,2026)" in source
    assert "af2-ffab2-all-seeds-all-stages-state.zip" in source
    assert "run_faruq_v3_af2_igem_parent_arm" in source
    assert "run_faruq_v3_af2_igem_parent_decision" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
    assert "--split test" not in source.lower()
