import json
from pathlib import Path

import torch
import yaml

from coffee_detector.afab import AFABConfig
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.af2_ffa import (
    AF2FFAConfig,
    AF2FFAParentPreservingModel,
    adapter_parameter_names,
    load_af2_ffa_weights,
    run_af2_ffa_parent_preserving_audit,
)
from coffee_detector.experiments.run_faruq_v3_af2_ffab2_parent_decision import (
    run_parent_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG_DIR = ROOT / "configs/af2_ffa_parent_preserving"


def _models(nc=5):
    afab = AFABConfig(mode="af2")
    torch.manual_seed(20260824)
    source = AFABDetectionModel(str(MODEL_YAML), nc=nc, verbose=False, afab=afab).eval()
    candidate = AF2FFAParentPreservingModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        afab=afab,
        af2_ffa=AF2FFAConfig(
            conditioning="spectral",
            residual_gain_cap=0.10,
            gradient_matched_cap=True,
            fusion_mode="parent_residual",
            residual_mix=1.0,
        ),
    )
    load_af2_ffa_weights(candidate, source)
    candidate.freeze_parent()
    return source, candidate


def test_parent_configs_are_matched_and_frozen():
    paths = sorted(CONFIG_DIR.glob("AF2FFAPR*.yaml"))
    assert len(paths) == 2
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    by_code = {item["code"]: item for item in payloads}
    assert set(by_code) == {"AF2FFAPR0", "AF2FFAPR1"}
    assert by_code["AF2FFAPR0"]["af2_ffa"]["conditioning"] == "zero"
    assert by_code["AF2FFAPR1"]["af2_ffa"]["conditioning"] == "spectral"
    for item in payloads:
        assert item["af2_ffa"]["fusion_mode"] == "parent_residual"
        assert item["af2_ffa"]["ambiguity_gate"] == "none"
        assert item["train"]["epochs"] == 30
        assert item["train"]["pretrained"] is False
    left = dict(by_code["AF2FFAPR0"]["af2_ffa"])
    right = dict(by_code["AF2FFAPR1"]["af2_ffa"])
    left.pop("conditioning")
    right.pop("conditioning")
    assert left == right
    assert by_code["AF2FFAPR0"]["train"] == by_code["AF2FFAPR1"]["train"]


def test_parent_freeze_keeps_only_adapters_trainable_and_bn_eval():
    _, candidate = _models()
    names = adapter_parameter_names(candidate)
    assert names
    assert all(".adapters." in name for name in names)
    assert sum(p.numel() for p in candidate.parameters() if p.requires_grad) < sum(
        p.numel() for p in candidate.parameters()
    ) * 0.01
    candidate.train()
    assert all(
        not module.training
        for module in candidate.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )


def test_parent_residual_is_exact_identity_then_has_live_adapter_gradient():
    source, candidate = _models()
    candidate.eval()
    features = [
        torch.rand(1, adapter.channels, size, size)
        for adapter, size in zip(candidate.model[-1].adapters, (16, 8, 4))
    ]
    with torch.inference_mode():
        parent = source.model[-1]([item.clone() for item in features])
        wrapped = candidate.model[-1]([item.clone() for item in features])
    assert torch.equal(parent[1]["one2one"]["boxes"], wrapped[1]["one2one"]["boxes"])
    assert torch.allclose(
        parent[1]["one2one"]["scores"],
        wrapped[1]["one2one"]["scores"],
        atol=1.0e-7,
        rtol=0.0,
    )

    candidate.train()
    head = candidate.model[-1]
    candidate.zero_grad(set_to_none=True)
    scores = head._classification_scores(0, features[0], head.one2many["cls_head"][0])
    scores.square().mean().backward()
    assert all(
        parameter.grad is None
        for name, parameter in candidate.named_parameters()
        if ".adapters." not in name
    )
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for adapter in head.adapters
        for parameter in adapter.parameters()
    )


def test_parent_static_audit_passes_on_synthetic_af2_checkpoint(tmp_path: Path):
    source, _ = _models()
    checkpoint = tmp_path / "af2_parent.pt"
    torch.save({"model": source, "train_args": {"seed": 42}}, checkpoint)
    result = run_af2_ffa_parent_preserving_audit(
        checkpoint, tmp_path / "audit.json", device="cpu", image_size=64
    )
    assert result["decision"] == "PASS"
    assert result["training_authorized"] is True
    assert result["test_access_authorized"] is False
    assert result["global_gates"]["trainable_fraction_under_one_percent"]
    assert result["records"]["AF2FFAPR1"]["gates"]["candidate_has_live_adapter_gradient"]
    assert result["records"]["AF2FFAPR1"]["gates"]["parent_receives_no_gradient"]


def _write_decision_fixture(root: Path, candidate_gain: float):
    parents, candidates = [], []
    for seed in (42, 123, 2026):
        parent = {
            "format": "coffee_detector.af2_ffa.from_start_arm_result.v1",
            "arm": "AF2FS",
            "seed": seed,
            "checkpoint_sha256": f"parent-{seed}",
            "evaluation_split": "val",
            "test_images_accessed": False,
            "metrics": {
                "macro_map50_95": 0.87,
                "bottom3_class_map50_95": 0.78,
                "worst_class_map50_95": 0.74,
            },
        }
        candidate = {
            "format": "coffee_detector.af2_ffa.parent_preserving_arm_result.v1",
            "arm": "AF2FFAPR1",
            "seed": seed,
            "parent_frozen": True,
            "trainable_scope": "ffab_adapters_only",
            "parent_checkpoint_sha256": f"parent-{seed}",
            "evaluation_split": "val",
            "test_images_accessed": False,
            "metrics": {
                "macro_map50_95": 0.87 + candidate_gain,
                "bottom3_class_map50_95": 0.78 + candidate_gain,
                "worst_class_map50_95": 0.74 + candidate_gain,
            },
        }
        p = root / f"parent_{seed}.json"
        c = root / f"candidate_{seed}.json"
        p.write_text(json.dumps(parent), encoding="utf-8")
        c.write_text(json.dumps(candidate), encoding="utf-8")
        parents.append(p)
        candidates.append(c)
    return parents, candidates


def test_parent_decision_requires_original_strict_upgrade_gate(tmp_path: Path):
    parents, candidates = _write_decision_fixture(tmp_path, 0.006)
    passed = run_parent_decision(parents, candidates, tmp_path / "pass.json")
    assert passed["decision"] == "PASS"
    assert passed["test_opened"] is False

    fail_root = tmp_path / "fail"
    fail_root.mkdir()
    parents, candidates = _write_decision_fixture(fail_root, 0.003)
    rejected = run_parent_decision(parents, candidates, fail_root / "reject.json")
    assert rejected["decision"] == "REJECT"
    assert rejected["criteria"]["macro_mean_gain_at_least_0_5pp"] is False
