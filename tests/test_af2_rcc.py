import json
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_rcc import (
    AF2RCCConfig,
    AF2RCCDetectionModel,
    RecoveredCueClassCalibration,
    freeze_for_rcc,
    load_af2_rcc_weights,
    run_af2_rcc_static_audit,
)
from coffee_detector.afab import AFABConfig
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.experiments.run_faruq_v3_af2_rcc_arm import (
    run_faruq_v3_af2_rcc_arm,
)
from coffee_detector.experiments.run_faruq_v3_af2_rcc_decision import (
    run_faruq_v3_af2_rcc_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc=21):
    afab = AFABConfig(mode="af2")
    torch.manual_seed(23)
    source = AFABDetectionModel(
        str(MODEL_YAML), nc=nc, verbose=False, afab=afab
    ).eval()
    candidate = AF2RCCDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        afab=afab,
        af2_rcc=AF2RCCConfig(),
    ).eval()
    load_af2_rcc_weights(candidate, source)
    return source, candidate


def test_calibration_is_identity_bounded_and_gradient_matched():
    module = RecoveredCueClassCalibration(21, AF2RCCConfig(gain_cap=0.10))
    scores = torch.rand(2, 21, 8, 8)
    cue = torch.rand(2, 3, 64, 64)
    assert torch.equal(module(scores, cue), scores)
    gradient = torch.autograd.grad(module.bounded_weight().sum(), module.weight)[0]
    assert torch.equal(gradient, torch.ones_like(gradient))
    with torch.no_grad():
        module.weight.fill_(100.0)
    correction = module(scores, torch.ones_like(cue)) - scores
    assert float(correction.detach().abs().max()) <= 0.10 + 1.0e-6


def test_zero_control_hides_cue_with_same_parameter_schema():
    zero = RecoveredCueClassCalibration(
        5, AF2RCCConfig(conditioning="zero")
    )
    recovered = RecoveredCueClassCalibration(
        5, AF2RCCConfig(conditioning="recovered")
    )
    assert zero.state_dict().keys() == recovered.state_dict().keys()
    with torch.no_grad():
        zero.weight.fill_(0.05)
        recovered.weight.fill_(0.05)
    scores = torch.rand(1, 5, 4, 4)
    cue = torch.rand(1, 3, 32, 32)
    assert torch.equal(zero(scores, cue), scores)
    assert not torch.equal(recovered(scores, cue), scores)


def test_full_model_starts_bitwise_af2_and_recovers_once():
    source, candidate = _models()
    calls = 0
    original = candidate.afab.recover

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    candidate.afab.recover = counted
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        native = source(image)
        calibrated = candidate(image)
    assert calls == 1
    assert torch.equal(
        native[1]["one2one"]["boxes"], calibrated[1]["one2one"]["boxes"]
    )
    assert torch.equal(
        native[1]["one2one"]["scores"], calibrated[1]["one2one"]["scores"]
    )


def test_active_calibration_changes_scores_but_never_boxes():
    _, candidate = _models()
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        identity = candidate(image)
    with torch.no_grad():
        for calibrator in candidate.model[-1].calibrators:
            calibrator.weight.fill_(0.05)
    with torch.inference_mode():
        active = candidate(image)
    assert torch.equal(
        identity[1]["one2one"]["boxes"], active[1]["one2one"]["boxes"]
    )
    assert not torch.equal(
        identity[1]["one2one"]["scores"], active[1]["one2one"]["scores"]
    )


def test_freeze_policy_allows_only_189_calibration_parameters():
    _, candidate = _models()
    policy = freeze_for_rcc(candidate)
    names = [name for name, value in candidate.named_parameters() if value.requires_grad]
    assert policy["trainable"] == 189
    assert len(names) == 3
    assert all("calibrators" in name and name.endswith("weight") for name in names)
    candidate.train()
    image = torch.rand(1, 3, 64, 64)
    loss = candidate(image)["one2many"]["scores"].square().mean()
    loss.backward()
    trainable = [p for p in candidate.parameters() if p.requires_grad]
    frozen = [p for p in candidate.parameters() if not p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert all(p.grad is None for p in frozen)


def test_configs_and_protocol_are_frozen_and_capacity_matched():
    paths = sorted((ROOT / "configs/af2_rcc").glob("AF2RCC*.yaml"))
    assert len(paths) == 2
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    assert {item["code"] for item in payloads} == {"AF2RCC0", "AF2RCC1"}
    assert len({json.dumps(item["afab"], sort_keys=True) for item in payloads}) == 1
    assert len({json.dumps(item["train"], sort_keys=True) for item in payloads}) == 1
    assert payloads[0]["train"]["epochs"] == 20
    assert {item["af2_rcc"]["conditioning"] for item in payloads} == {
        "zero",
        "recovered",
    }
    protocol = (
        ROOT
        / "docs/FARUQ_V3_AF2_RECOVERED_CUE_CALIBRATION_PROTOCOL_2026-08-22.md"
    ).read_text(encoding="utf-8")
    assert "Status: frozen before training" in protocol
    assert "189 weights" in protocol
    assert "Test is unavailable and locked" in protocol


def test_static_audit_passes_on_af2_checkpoint(tmp_path: Path):
    source, _ = _models()
    checkpoint = tmp_path / "af2.pt"
    torch.save({"model": source, "train_args": {"seed": 42}}, checkpoint)
    result = run_af2_rcc_static_audit(
        checkpoint, tmp_path / "static.json", device="cpu", image_size=64
    )
    assert result["decision"] == "PASS"
    assert result["added_parameters"] == 189
    assert result["gates"]["one_af2_recovery_call"]
    assert result["gates"]["no_additional_fft"]
    assert result["gates"]["classification_path_only"]


def test_decision_gate_requires_tail_and_target_preservation(tmp_path: Path):
    names = [f"class_{index}" for index in range(20)] + [
        "kulit_tanduk_ukuran_kecil"
    ]
    baseline = {
        "macro_map50_95": 0.88,
        "bottom3_class_map50_95": 0.79,
        "worst_class_map50_95": 0.76,
        "map50_95_by_class": {name: 0.80 for name in names},
    }
    candidate = json.loads(json.dumps(baseline))
    candidate.update(
        macro_map50_95=0.882,
        bottom3_class_map50_95=0.80,
        worst_class_map50_95=0.759,
    )
    payload = {
        "format": "coffee_detector.af2_rcc.arm_result.v1",
        "arm": "AF2RCC1",
        "seed": 42,
        "baseline_metrics": baseline,
        "metrics": candidate,
        "test_images_accessed": False,
    }
    source = tmp_path / "result.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    result = run_faruq_v3_af2_rcc_decision(source, tmp_path / "decision.json")
    assert result["decision"] == "PASS"
    assert result["improved_headline_metrics"] == 2


def test_runner_fails_before_artifact_access_without_authorization(tmp_path: Path):
    try:
        run_faruq_v3_af2_rcc_arm(
            tmp_path,
            tmp_path / "grouped.json",
            tmp_path / "af2.pt",
            tmp_path / "static.json",
            tmp_path / "output",
        )
    except RuntimeError as error:
        assert "belum diotorisasi" in str(error)
    else:
        raise AssertionError("Runner lolos tanpa otorisasi training")


def test_notebook_is_resumable_sparse_and_test_locked():
    notebook = ROOT / "notebooks/Faruq_V3_AF2_Recovered_Cue_Calibration_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(notebook), "exec")
    assert "run_faruq_v3_af2_rcc_arm" in source
    assert "--authorize-training" in source
    assert "last.pt" not in source  # runner owns resume; notebook does not overwrite it
    assert "time.sleep(60)" in source
    assert "split=test" not in source.lower()
