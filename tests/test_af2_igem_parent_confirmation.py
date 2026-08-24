import copy
import hashlib
import json
from pathlib import Path

import torch

from coffee_detector.af2_parent_residual import (
    AF2ParentResidualConfig,
    AF2ParentResidualDetectionModel,
    load_af2_parent_residual_weights,
)
from coffee_detector.af2_parent_residual.igem_confirmation import (
    AUDIT_REVISION,
    run_af2_igem_parent_static_audit,
)
from coffee_detector.af2_parent_residual.igem_trainer import _sync_frozen_parent_to_ema
from coffee_detector.afab import AFABConfig
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.experiments.run_faruq_v3_af2_igem_parent_arm import (
    run_faruq_v3_af2_igem_parent_arm,
)
from coffee_detector.experiments.run_faruq_v3_af2_igem_parent_decision import (
    run_igem_parent_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
SEEDS = (42, 123, 2026)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _parent(seed: int, values):
    per_class = {f"class_{index}": 0.8 for index in range(21)}
    return {
        "format": "coffee_detector.af2_ffa.from_start_arm_result.v1",
        "arm": "AF2FS",
        "seed": seed,
        "checkpoint_sha256": f"parent-{seed}",
        "metrics": dict(values, map50_95_by_class=per_class),
        "test_images_accessed": False,
    }


def _arm(arm: str, conditioning: str, seed: int, parent_payload, parent_result_sha: str, values):
    per_class = {f"class_{index}": 0.8 for index in range(21)}
    return {
        "format": "coffee_detector.af2_parent_residual.igem_arm_result.v2",
        "protocol": "faruq-v3-af2fs-igem-parent-confirmation-v1",
        "arm": arm,
        "family": "igem",
        "conditioning": conditioning,
        "seed": seed,
        "baseline_metrics": parent_payload["metrics"],
        "baseline_source": "canonical_AF2FS_parent_result",
        "metrics": dict(values, map50_95_by_class=per_class),
        "initial_af2_checkpoint_sha256": parent_payload["checkpoint_sha256"],
        "parent_result_sha256": parent_result_sha,
        "parent_frozen": True,
        "final_parent_state_bitwise_exact": True,
        "trainable_scope": "igem_residual_only",
        "evaluation_split": "val",
        "run_contract_sha256": f"contract-{arm}-{seed}",
        "test_images_accessed": False,
    }


def _write_case(tmp_path: Path, parent_values, control_values, candidate_values):
    parent_paths, control_paths, candidate_paths = [], [], []
    for seed in SEEDS:
        parent_payload = _parent(seed, parent_values)
        pp = tmp_path / f"AF2FS_seed{seed}_result.json"
        pp.write_text(json.dumps(parent_payload), encoding="utf-8")
        parent_sha = _sha256(pp)
        cp = tmp_path / f"AF2IGEM0_seed{seed}_result.json"
        fp = tmp_path / f"AF2IGEM1_seed{seed}_result.json"
        cp.write_text(json.dumps(_arm("AF2IGEM0", "zero", seed, parent_payload, parent_sha, control_values)), encoding="utf-8")
        fp.write_text(json.dumps(_arm("AF2IGEM1", "feature", seed, parent_payload, parent_sha, candidate_values)), encoding="utf-8")
        parent_paths.append(pp)
        control_paths.append(cp)
        candidate_paths.append(fp)
    return parent_paths, control_paths, candidate_paths


def _igem_model():
    afab = AFABConfig(mode="af2")
    source = AFABDetectionModel(str(MODEL_YAML), nc=21, verbose=False, afab=afab).eval()
    target = AF2ParentResidualDetectionModel(
        str(MODEL_YAML),
        nc=21,
        verbose=False,
        afab=afab,
        parent_residual=AF2ParentResidualConfig(family="igem", conditioning="feature"),
    ).eval()
    load_af2_parent_residual_weights(target, source)
    return target


def test_ema_sync_restores_parent_exact_without_overwriting_residual_ema():
    live = _igem_model()
    ema = copy.deepcopy(live)
    with torch.no_grad():
        live.model[0].conv.weight.add_(0.25)
        first_ema_residual = next(ema.model[-1].residual.parameters())
        first_ema_residual.add_(0.5)
    residual_before = {
        key: value.detach().clone() for key, value in ema.model[-1].residual.state_dict().items()
    }
    _sync_frozen_parent_to_ema(live, ema)
    live_state, ema_state = live.state_dict(), ema.state_dict()
    for key in live_state:
        if ".residual." not in key:
            assert torch.equal(live_state[key], ema_state[key])
    for key, value in residual_before.items():
        assert torch.equal(value, ema.model[-1].residual.state_dict()[key])


def test_three_seed_decision_accepts_tail_pareto_route(tmp_path: Path):
    parent = dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770)
    control = dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770)
    candidate = dict(macro_map50_95=0.8795, bottom3_class_map50_95=0.796, worst_class_map50_95=0.781)
    parents, controls, candidates = _write_case(tmp_path, parent, control, candidate)
    result = run_igem_parent_decision(parents, controls, candidates, tmp_path / "decision.json")
    assert result["decision"] == "RETAIN"
    assert result["criteria"]["tail_pareto_route_pass"] is True
    assert result["criteria"]["canonical_parent_binding_verified"] is True
    assert result["criteria"]["serialized_parent_state_verified"] is True
    assert result["test_opened"] is False


def test_three_seed_decision_rejects_flat_information_pair(tmp_path: Path):
    values = dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770)
    parents, controls, candidates = _write_case(tmp_path, values, values, values)
    result = run_igem_parent_decision(parents, controls, candidates, tmp_path / "decision.json")
    assert result["decision"] == "REJECT"
    assert not result["criteria"]["superiority_route_pass"]
    assert not result["criteria"]["tail_pareto_route_pass"]


def test_decision_rejects_wrong_parent_binding(tmp_path: Path):
    values = dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770)
    parents, controls, candidates = _write_case(tmp_path, values, values, values)
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload["initial_af2_checkpoint_sha256"] = "wrong-parent"
    candidates[0].write_text(json.dumps(payload), encoding="utf-8")
    try:
        run_igem_parent_decision(parents, controls, candidates, tmp_path / "decision.json")
    except RuntimeError as error:
        assert "canonical" in str(error)
    else:
        raise AssertionError("Decision menerima parent yang salah")


def test_decision_rejects_result_without_exact_serialized_parent(tmp_path: Path):
    values = dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770)
    parents, controls, candidates = _write_case(tmp_path, values, values, values)
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload["final_parent_state_bitwise_exact"] = False
    candidates[0].write_text(json.dumps(payload), encoding="utf-8")
    try:
        run_igem_parent_decision(parents, controls, candidates, tmp_path / "decision.json")
    except RuntimeError as error:
        assert "parent exact" in str(error)
    else:
        raise AssertionError("Decision menerima best.pt tanpa exact parent")


def test_runner_rejects_unapproved_training_before_artifact_access(tmp_path: Path):
    try:
        run_faruq_v3_af2_igem_parent_arm(
            "AF2IGEM1",
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path,
            seed=42,
        )
    except RuntimeError as error:
        assert "belum diotorisasi" in str(error)
    else:
        raise AssertionError("Runner lolos tanpa authorization")


def test_dedicated_igem_static_audit_passes_for_serialized_af2_parent(tmp_path: Path):
    source = AFABDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, afab=AFABConfig(mode="af2")
    ).eval()
    checkpoint = tmp_path / "af2.pt"
    torch.save({"model": source, "train_args": {"seed": 42}}, checkpoint)
    result = run_af2_igem_parent_static_audit(
        checkpoint, tmp_path / "igem_static.json", device="cpu", image_size=64
    )
    assert result["decision"] == "PASS"
    assert result["audit_revision"] == AUDIT_REVISION
    assert result["activity_contract"] == "direct_residual_correction_zero_vs_finite_nonzero"
    assert set(result["records"]) == {"AF2IGEM0", "AF2IGEM1"}
    assert result["gates"]["same_initial_residual_state"]
    assert result["gates"]["control_parent_transfer_exact"]
    assert result["gates"]["candidate_parent_transfer_exact"]
    assert result["gates"]["control_zero_information_residual_exact"]
    assert result["gates"]["candidate_feature_residual_live"]
    assert result["records"]["AF2IGEM0"]["active_residual_correction_max_abs"] == 0.0
    assert result["records"]["AF2IGEM1"]["active_residual_correction_max_abs"] > 0.0
    assert result["gates"]["control_parent_state_unchanged"]
    assert result["gates"]["candidate_parent_state_unchanged"]
    assert result["training_authorized"] is True
    assert result["test_access_authorized"] is False


def test_protocol_and_kaggle_notebook_are_frozen_and_test_locked():
    protocol = (ROOT / "docs/FARUQ_V3_AF2FS_IGEM_PARENT_CONFIRMATION_PROTOCOL_2026-08-24.md").read_text(encoding="utf-8")
    assert "FROZEN BEFORE TRAINING" in protocol
    assert "AF2IGEM0/1 x seeds 42,123,2026" in protocol
    assert "Locked test: **closed**" in protocol
    assert "EMA" in protocol

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
    assert "run_igem_parent_decision" in source
    assert "--parent-result" in source
    assert "AUDIT_REVISION" in source
    assert "final_parent_state_bitwise_exact" in source
    assert "os.chdir(WORK)" in source
    assert source.index("os.chdir(WORK)") < source.index("shutil.rmtree(REPO)")
    assert "split=test" not in source.lower()
    assert "--split test" not in source.lower()
