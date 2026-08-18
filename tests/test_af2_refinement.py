"""Executable contracts for the frozen AF2 radial-wavelet follow-up."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_refinement import (
    TRAIN_ARMS,
    AF2RefinementInputEnhancer,
    frozen_refinement_config,
)
from coffee_detector.af2_spectral import SpectralInputEnhancer, frozen_arm_config
from coffee_detector.afab import AFABConfig, AFABInputEnhancer
from coffee_detector.experiments.run_faruq_v3_af2_refinement_decision import (
    run_af2_refinement_decision,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def test_af2c_remains_bitwise_legacy_af2():
    torch.manual_seed(18)
    image = torch.rand(2, 3, 64, 64)
    legacy = AFABInputEnhancer(
        AFABConfig(
            mode="af2",
            patch_size=32,
            overlap=0.50,
            gamma=0.10,
            angular_bins=360,
        )
    )
    followup = AF2RefinementInputEnhancer(frozen_refinement_config("AF2C"))
    assert torch.equal(followup(image), legacy(image))


def test_af2rad_preserves_legacy_360_angle_map_and_only_adds_radial_index():
    legacy = AFABInputEnhancer(
        AFABConfig(mode="af2", patch_size=32, overlap=0.50, gamma=0.10, angular_bins=360)
    )
    followup = AF2RefinementInputEnhancer(frozen_refinement_config("AF2RAD"))
    assert followup.radial.config.angular_bins == 360
    assert torch.equal(followup.radial.angle_bin, legacy.angle_bin)
    assert set(followup.radial.radial_bin.unique().tolist()) == {0, 1, 2}
    combined = (
        followup.radial.radial_bin * followup.radial.config.angular_bins
        + followup.radial.angle_bin
    )
    assert combined.shape == legacy.angle_bin.shape
    assert combined.min().item() >= 0
    assert combined.max().item() < 3 * 360


def test_af2wav_reuses_completed_wav1_cue_bitwise():
    torch.manual_seed(19)
    image = torch.rand(1, 3, 65, 63)
    reference = SpectralInputEnhancer(frozen_arm_config("WAV1")).recover(image)
    followup = AF2RefinementInputEnhancer(frozen_refinement_config("AF2WAV"))
    assert torch.equal(followup.wavelet_cue(image), reference)


def test_max_fusion_never_attenuates_af2_cue():
    torch.manual_seed(20)
    image = torch.rand(1, 3, 64, 64)
    for arm in ("AF2WAV", "AF2RADWAV"):
        frontend = AF2RefinementInputEnhancer(frozen_refinement_config(arm))
        base = frontend.base_cue(image)
        wave = frontend.wavelet_cue(image)
        fused = frontend.fused_cue(image)
        assert torch.equal(fused, torch.maximum(base, wave))
        assert torch.all(fused >= base)


def test_all_followup_frontends_are_parameter_free_finite_and_differentiable():
    for index, arm in enumerate(TRAIN_ARMS):
        torch.manual_seed(30 + index)
        image = torch.rand(1, 3, 64, 64, requires_grad=True)
        frontend = AF2RefinementInputEnhancer(frozen_refinement_config(arm))
        output = frontend(image)
        assert output.shape == image.shape
        assert output.dtype == image.dtype
        assert torch.isfinite(output).all()
        assert not frontend.state_dict()
        assert not list(frontend.parameters())
        output.mean().backward()
        assert image.grad is not None and torch.isfinite(image.grad).all()


def test_configs_are_schedule_matched_and_mechanisms_are_frozen():
    payloads = {
        path.stem.split("_")[0]: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (ROOT / "configs/af2_refinement").glob("*.yaml")
    }
    assert set(payloads) == set(TRAIN_ARMS)
    assert len({json.dumps(payload["train"], sort_keys=True) for payload in payloads.values()}) == 1
    assert {payload["model"] for payload in payloads.values()} == {
        "configs/coffee_fg/models/yolo26n-p3.yaml"
    }
    for arm, payload in payloads.items():
        refinement = payload["refinement"]
        assert refinement["arm"] == arm
        assert refinement["patch_size"] == 32
        assert refinement["overlap"] == 0.50
        assert refinement["gamma"] == 0.10
        assert refinement["angular_bins"] == 360
        assert refinement["radial_bands"] == 3
        assert refinement["wavelet_levels"] == 2
        assert refinement["fusion"] == "max"


def _write_result(root: Path, arm: str, values: tuple[float, float, float]) -> None:
    reports = root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{arm}_seed42_result.json").write_text(
        json.dumps(
            {
                "format": "coffee_detector.af2_refinement.arm_result.v1",
                "arm": arm,
                "seed": 42,
                "metrics": dict(zip(METRICS, values)),
                "latency": {"median_ms": 20.0},
                "evaluation_split": "val",
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )


def test_decision_keeps_original_plus_0_5_macro_gate(tmp_path):
    baseline = tmp_path / "af2.json"
    baseline.write_text(
        json.dumps(
            {
                "candidate": {"AF2": dict(zip(METRICS, (0.88, 0.80, 0.79)))},
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    _write_result(tmp_path, "AF2RAD", (0.8849, 0.84, 0.84))
    _write_result(tmp_path, "AF2WAV", (0.8850, 0.81, 0.80))
    _write_result(tmp_path, "AF2RADWAV", (0.8860, 0.80, 0.785))

    decision = run_af2_refinement_decision(tmp_path, baseline)

    assert decision["candidates"]["AF2RAD"]["decision"] == "REJECT"
    assert decision["candidates"]["AF2WAV"]["decision"] == "RETAIN"
    assert decision["candidates"]["AF2RADWAV"]["decision"] == "RETAIN"
    assert decision["test_opened"] is False


def test_protocol_and_notebook_lock_the_followup_contract():
    protocol = (ROOT / "docs/FARUQ_V3_AF2_RAD_WAVELET_REFINEMENT_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "frozen before follow-up training" in protocol
    assert "AF2RAD" in protocol and "AF2WAV" in protocol and "AF2RADWAV" in protocol
    assert "+0.5 percentage point" in protocol
    assert "Faruq locked test remains closed" in protocol

    notebook = ROOT / "notebooks/Faruq_V3_AF2_RAD_Wavelet_Refinement_Sequential_Kaggle.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(notebook), "exec")
    assert "agent/af2-rad-wavelet-refinement" in source
    assert "run_af2_refinement_static_audit" in source
    assert "for arm in TRAIN_ARMS: run_arm(arm)" in source
    assert "run_af2_refinement_decision" in source
    assert "test_images_accessed" in source
    assert "shutil.rmtree(DATA)" in source
