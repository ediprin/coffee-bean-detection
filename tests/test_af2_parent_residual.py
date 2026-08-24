import json
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_parent_residual import (
    AF2ParentResidualConfig,
    AF2ParentResidualDetectionModel,
    freeze_for_parent_residual,
    load_af2_parent_residual_weights,
    run_af2_parent_residual_static_audit,
)
from coffee_detector.af2_parent_residual.audit import ATOL, RTOL
from coffee_detector.afab import AFABConfig
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.experiments.run_faruq_v3_af2_parent_residual_arm import (
    run_faruq_v3_af2_parent_residual_arm,
)
from coffee_detector.experiments.run_faruq_v3_af2_parent_residual_decision import (
    run_faruq_v3_af2_parent_residual_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _close(left, right):
    return torch.allclose(left, right, atol=ATOL, rtol=RTOL)


def _models(family="saf", conditioning="feature"):
    afab = AFABConfig(mode="af2")
    torch.manual_seed(61)
    source = AFABDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, afab=afab
    ).eval()
    candidate = AF2ParentResidualDetectionModel(
        str(MODEL_YAML),
        nc=21,
        verbose=False,
        afab=afab,
        parent_residual=AF2ParentResidualConfig(
            family=family, conditioning=conditioning
        ),
    ).eval()
    load_af2_parent_residual_weights(candidate, source)
    return source, candidate


def _activate(candidate):
    head = candidate.model[-1]
    with torch.no_grad():
        if head.config.family == "saf":
            for layer in head.residual.class_corrections:
                layer.weight.fill_(1.0)
        else:
            for level in head.residual:
                level.class_correction.weight.fill_(1.0)


def test_both_families_start_at_af2_and_preserve_boxes_when_active():
    image = torch.rand(1, 3, 64, 64)
    for family in ("saf", "igem"):
        source, candidate = _models(family)
        with torch.inference_mode():
            native = source(image)
            identity = candidate(image)
        assert _close(
            native[1]["one2one"]["boxes"], identity[1]["one2one"]["boxes"]
        )
        assert _close(
            native[1]["one2one"]["scores"], identity[1]["one2one"]["scores"]
        )
        _activate(candidate)
        with torch.inference_mode():
            active = candidate(image)
        assert _close(
            identity[1]["one2one"]["boxes"], active[1]["one2one"]["boxes"]
        )
        # Historical activity check: any deterministic score change is enough.
        # Numerical tolerance is reserved for identity/box-preservation gates.
        assert not torch.equal(
            identity[1]["one2one"]["scores"], active[1]["one2one"]["scores"]
        )


def test_zero_controls_match_schema_and_hide_features():
    image = torch.rand(1, 3, 64, 64)
    for family in ("saf", "igem"):
        _, control = _models(family, "zero")
        _, candidate = _models(family, "feature")
        assert {
            key: tuple(value.shape) for key, value in control.state_dict().items()
        } == {key: tuple(value.shape) for key, value in candidate.state_dict().items()}
        _activate(control)
        _activate(candidate)
        with torch.inference_mode():
            control_output = control(image)
            candidate_output = candidate(image)
        # This legacy test only checks that feature conditioning can produce a
        # distinct computation; materiality is enforced by the dedicated IGEM audit.
        assert not torch.equal(
            control_output[1]["one2one"]["scores"],
            candidate_output[1]["one2one"]["scores"],
        )


def test_freeze_policy_exposes_only_residual_parameters():
    _, candidate = _models("saf")
    policy = freeze_for_parent_residual(candidate)
    trainable_names = [
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    ]
    assert policy["trainable"] > 0
    assert trainable_names
    assert all("model.23.residual" in name for name in trainable_names)
    assert all(
        not parameter.requires_grad
        for name, parameter in candidate.named_parameters()
        if "model.23.residual" not in name
    )


def test_four_configs_are_family_matched_and_protocol_frozen():
    payloads = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (ROOT / "configs/af2_parent_residual").glob("*.yaml")
    }
    assert set(payloads) == {"AF2SAF0", "AF2SAF1", "AF2IGEM0", "AF2IGEM1"}
    for family, codes in (
        ("saf", ("AF2SAF0", "AF2SAF1")),
        ("igem", ("AF2IGEM0", "AF2IGEM1")),
    ):
        left, right = (payloads[code] for code in codes)
        assert left["parent_residual"]["family"] == family
        assert right["parent_residual"]["family"] == family
        assert left["train"] == right["train"]
        assert left["train"]["epochs"] == 20
        left_residual = dict(left["parent_residual"])
        right_residual = dict(right["parent_residual"])
        left_residual.pop("conditioning")
        right_residual.pop("conditioning")
        assert left_residual == right_residual
    protocol = (
        ROOT / "docs/FARUQ_V3_AF2_PARENT_RESIDUAL_PROTOCOL_2026-08-23.md"
    ).read_text(encoding="utf-8")
    assert "Status: frozen before training" in protocol
    assert "frozen completed AF2 parent" in protocol
    assert "Test: unavailable and locked" in protocol


def test_static_audit_passes_for_serialized_af2_parent(tmp_path: Path):
    source, _ = _models("saf")
    checkpoint = tmp_path / "af2.pt"
    torch.save({"model": source, "train_args": {"seed": 42}}, checkpoint)
    result = run_af2_parent_residual_static_audit(
        checkpoint, tmp_path / "static.json", device="cpu", image_size=64
    )
    assert result["decision"] == "PASS"
    assert result["training_authorized"] is True
    assert result["test_access_authorized"] is False
    for family in ("saf", "igem"):
        assert result["gates"][f"{family}_same_parameter_count"]
        assert result["gates"][f"{family}_same_state_schema"]
    for arm in ("AF2SAF0", "AF2SAF1", "AF2IGEM0", "AF2IGEM1"):
        assert result["gates"][f"{arm}_initial_identity"]
        assert result["gates"][f"{arm}_boxes_preserved"]
        assert result["gates"][f"{arm}_frozen_parent"]


def _result(arm, family, conditioning, values):
    per_class = {f"class_{index}": 0.8 for index in range(21)}
    metrics = dict(values, map50_95_by_class=per_class)
    baseline = dict(
        macro_map50_95=0.88,
        bottom3_class_map50_95=0.79,
        worst_class_map50_95=0.77,
        map50_95_by_class=per_class,
    )
    return {
        "format": "coffee_detector.af2_parent_residual.arm_result.v1",
        "arm": arm,
        "family": family,
        "conditioning": conditioning,
        "seed": 42,
        "baseline_metrics": baseline,
        "metrics": metrics,
        "initial_af2_checkpoint_sha256": "same",
        "test_images_accessed": False,
    }


def test_decision_accepts_tail_pareto_without_rigid_macro_gain(tmp_path: Path):
    control = _result(
        "AF2SAF0", "saf", "zero",
        dict(macro_map50_95=0.880, bottom3_class_map50_95=0.790, worst_class_map50_95=0.770),
    )
    candidate = _result(
        "AF2SAF1", "saf", "feature",
        dict(macro_map50_95=0.8795, bottom3_class_map50_95=0.796, worst_class_map50_95=0.781),
    )
    control_path, candidate_path = tmp_path / "control.json", tmp_path / "candidate.json"
    control_path.write_text(json.dumps(control), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    decision = run_faruq_v3_af2_parent_residual_decision(
        "saf", control_path, candidate_path, tmp_path / "decision.json"
    )
    assert decision["decision"] == "RETAIN"
    assert decision["criteria"]["tail_pareto_route"]


def test_runner_fails_before_artifact_access_without_authorization(tmp_path: Path):
    try:
        run_faruq_v3_af2_parent_residual_arm(
            "AF2SAF1", tmp_path, tmp_path, tmp_path, tmp_path, tmp_path
        )
    except RuntimeError as error:
        assert "belum diotorisasi" in str(error)
    else:
        raise AssertionError("Runner lolos tanpa otorisasi")


def test_arm_notebooks_are_separate_resumable_sparse_and_test_locked():
    for arm in ("SAF0", "SAF1", "IGEM0", "IGEM1"):
        notebook = (
            ROOT / f"notebooks/Faruq_V3_AF2_Parent_Residual_{arm}_Colab.ipynb"
        )
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload.get("cells", [])
        )
        for cell in payload["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), str(notebook), "exec")
        assert f"ARM='AF2{arm}'" in source
        assert "torch.cuda.is_available()" in source
        assert "T4 GPU" in source
        assert "run_faruq_v3_af2_parent_residual_arm" in source
        assert "--authorize-training" in source
        assert "time.sleep(60)" in source
        assert "last.pt" not in source
        assert "split=test" not in source.lower()
        assert "--split test" not in source.lower()

    decision_notebook = (
        ROOT / "notebooks/Faruq_V3_AF2_Parent_Residual_Decision_Colab.ipynb"
    )
    payload = json.loads(decision_notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(decision_notebook), "exec")
    assert "run_faruq_v3_af2_parent_residual_decision" in source
    assert "TRAINING: False | TEST: False" in source
    assert "--authorize-training" not in source
