"""Executable contracts for the frozen AF2 spectral-factorization study."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from coffee_detector.af2_spectral import SpectralFrontendConfig, SpectralInputEnhancer
from coffee_detector.af2_spectral.config import ARMS, frozen_arm_config
from coffee_detector.af2_spectral.operator import (
    haar_dwt2,
    haar_idwt2,
    soft_direction_weight,
)
from coffee_detector.experiments.run_faruq_v3_af2_spectral_decision import (
    run_spectral_decision,
)
from coffee_detector.experiments.prepare_af2_spectral_kaggle import (
    restore_spectral_kaggle_run,
)
from coffee_detector.afab import AFABConfig, AFABInputEnhancer


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def test_af2c_is_bitwise_legacy_af2():
    torch.manual_seed(4)
    image = torch.rand(2, 3, 64, 64)
    legacy = AFABInputEnhancer(
        AFABConfig(mode="af2", patch_size=32, overlap=0.5, gamma=0.1, angular_bins=360)
    )
    control = SpectralInputEnhancer(frozen_arm_config("AF2C"))
    assert torch.equal(control(image), legacy(image))


def test_hann_patch_frontend_is_finite_and_retains_constant_raw_boundary():
    image = torch.full((1, 3, 67, 79), 0.4, requires_grad=True)
    frontend = SpectralInputEnhancer(frozen_arm_config("AF2WIN"))
    output = frontend(image)
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert torch.isfinite(output).all()
    assert (output + 1.0e-7 >= image).all()
    output.mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_orientation_bins_are_modulo_pi_for_af2ori():
    frontend = SpectralInputEnhancer(frozen_arm_config("AF2ORI"))
    center = frontend.config.patch_size // 2
    # Opposite vectors have equal magnitude-spectrum orientation modulo pi.
    assert frontend.angle_bin[center, center + 5] == frontend.angle_bin[center, center - 5]
    assert frontend.angle_bin[center + 7, center + 3] == frontend.angle_bin[
        center - 7, center - 3
    ]


def test_polar_geometry_covers_each_orientation_and_radial_band():
    frontend = SpectralInputEnhancer(frozen_arm_config("AF2POL"))
    combinations = frontend.radial_bin * frontend.config.angular_bins + frontend.angle_bin
    assert combinations.numel() == frontend.config.patch_size**2
    assert set(frontend.radial_bin.unique().tolist()) == {0, 1, 2}
    assert frontend.angle_bin.min() == 0
    assert frontend.angle_bin.max() == 15


def test_soft_gate_is_monotonic_and_has_nonzero_transition_gradient():
    density = torch.tensor([[[0.1, 0.3, 0.5, 0.7, 0.9]]], requires_grad=True)
    threshold = torch.tensor([[0.5]])
    weight = soft_direction_weight(density, threshold, 0.02)
    assert torch.all(weight[..., 1:] >= weight[..., :-1])
    weight[..., 2].backward()
    assert density.grad is not None and density.grad[..., 2].abs().item() > 0


def test_luminance_arm_shares_the_identical_spectral_gate_across_rgb():
    frontend = SpectralInputEnhancer(frozen_arm_config("AF2LUM"))
    frequency = torch.randn(3, 3, 32, 32, dtype=torch.complex64)
    weight = frontend._direction_weight(frequency)
    assert torch.equal(weight[:, 0], weight[:, 1])
    assert torch.equal(weight[:, 1], weight[:, 2])


@pytest.mark.parametrize("arm", ("PCG1", "WAV1"))
def test_non_fft_controls_are_finite_and_active_with_finite_gradients(arm):
    torch.manual_seed(3)
    image = torch.rand(1, 3, 65, 63, requires_grad=True)
    frontend = SpectralInputEnhancer(frozen_arm_config(arm))
    output = frontend(image)
    assert output.shape == image.shape
    assert torch.isfinite(output).all()
    assert not torch.equal(output.detach(), image.detach())
    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_haar_reconstruction_and_constant_detail_response():
    torch.manual_seed(8)
    image = torch.rand(2, 3, 13, 15)
    bands, shape = haar_dwt2(image)
    reconstructed = haar_idwt2(bands, shape)
    assert torch.allclose(reconstructed, image, atol=2.0e-6, rtol=2.0e-6)
    constant_bands, _ = haar_dwt2(torch.ones(1, 1, 16, 16))
    assert constant_bands[:, :, 1:].abs().max().item() < 1.0e-6


def test_all_frontends_keep_dtype_shape_and_no_persistent_state():
    image = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    for arm in ARMS:
        frontend = SpectralInputEnhancer(frozen_arm_config(arm))
        output = frontend(image)
        assert output.shape == image.shape
        assert output.dtype == image.dtype
        assert torch.isfinite(output).all()
        assert not frontend.state_dict()


def test_new_configs_are_schedule_matched_and_frozen():
    payloads = {
        path.stem.split("_")[0]: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (ROOT / "configs/af2_spectral").glob("*.yaml")
    }
    assert set(payloads) == set(ARMS[1:])
    assert {json.dumps(payload["train"], sort_keys=True) for payload in payloads.values()}.__len__() == 1
    assert {payload["model"] for payload in payloads.values()} == {
        "configs/coffee_fg/models/yolo26n-p3.yaml"
    }
    assert payloads["AF2POL"]["spectral"] == payloads["AF2SOFT"]["spectral"] | {
        "arm": "AF2POL"
    }
    assert payloads["AF2LUM"]["spectral"]["angular_bins"] == 16
    assert payloads["AF2LUM"]["spectral"]["radial_bands"] == 3


def _write_arm(root: Path, arm: str, values: tuple[float, float, float]) -> None:
    (root / "val_reports").mkdir(parents=True, exist_ok=True)
    (root / "val_reports" / f"{arm}_seed42_result.json").write_text(
        json.dumps(
            {
                "format": "coffee_detector.af2_spectral.arm_result.v1",
                "arm": arm,
                "seed": 42,
                "metrics": dict(zip(METRICS, values)),
                "latency": {"median_ms": 4.0},
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )


def test_stage1_decision_requires_predeclared_lower_tail_gate(tmp_path):
    baseline = tmp_path / "af2.json"
    baseline.write_text(
        json.dumps({"candidate": {"AF2": dict(zip(METRICS, (0.88, 0.80, 0.79)))}, "test_images_accessed": False}),
        encoding="utf-8",
    )
    for arm in ("AF2WIN", "AF2ORI", "AF2POL", "AF2SOFT", "AF2LUM"):
        _write_arm(tmp_path, arm, (0.886, 0.80, 0.785))
    _write_arm(tmp_path, "AF2LUM", (0.889, 0.81, 0.80))

    decision = run_spectral_decision(tmp_path, baseline, stage="stage1")

    assert decision["decision"] == "PASS"
    assert decision["winner"] == "AF2LUM"
    assert decision["test_opened"] is False


def test_protocol_and_kaggle_notebooks_freeze_the_staged_contract():
    protocol = (ROOT / "docs/FARUQ_V3_AF2_SPECTRAL_FACTORIZATION_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "Status: **frozen before training**" in protocol
    assert "AF2WIN" in protocol and "PCG1" in protocol and "WAV1" in protocol
    assert "Faruq locked test is not reopened" in protocol
    expected = (
        "AF2WIN",
        "AF2ORI",
        "AF2POL",
        "AF2SOFT",
        "AF2LUM",
        "PCG1",
        "WAV1",
    )
    for arm in expected:
        path = ROOT / f"notebooks/Faruq_V3_AF2_Spectral_{arm}_Kaggle.ipynb"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["cells"][1]["source"][0].endswith("\n")
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        for cell in payload["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), str(path), "exec")
        assert arm in source
        assert "prepare_af2_spectral_kaggle_input" in source
        assert "restore_spectral_kaggle_run" in source
        assert "DOWNLOAD SEBELUM STOP SESSION" in source
        assert "rglob('best.pt')" not in source
    global_payload = json.loads(
        (ROOT / "notebooks/Faruq_V3_AF2_Spectral_Global_Decision_Kaggle.ipynb").read_text(
            encoding="utf-8"
        )
    )
    global_source = "\n".join("".join(cell.get("source", [])) for cell in global_payload["cells"])
    assert "run_spectral_decision" in global_source
    assert "test_opened" in global_source
    sequential = json.loads(
        (ROOT / "notebooks/Faruq_V3_AF2_Spectral_Stage1_Sequential_Kaggle.ipynb").read_text(
            encoding="utf-8"
        )
    )
    sequential_source = "\n".join(
        "".join(cell.get("source", [])) for cell in sequential["cells"]
    )
    assert "for arm in ARMS: run_arm(arm)" in sequential_source
    assert "restore_spectral_kaggle_run" in sequential_source
    assert "time.sleep(120)" in sequential_source
    bundle = json.loads(
        (ROOT / "notebooks/Faruq_V3_AF2_Spectral_Kaggle_Bundle_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    bundle_source = "\n".join("".join(cell.get("source", [])) for cell in bundle["cells"])
    assert "build_af2_spectral_kaggle_bundle" in bundle_source
    assert "D0 checkpoints" in bundle_source
    assert "datasets','version" in bundle_source


def test_kaggle_restore_requires_an_exact_arm_seed_and_sha_contract(tmp_path):
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    prior = input_root / "prior-arm"
    prior.mkdir(parents=True)
    checkpoint, config = tmp_path / "D0_seed42_best.pt", tmp_path / "AF2WIN.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("code: AF2WIN\n", encoding="utf-8")
    import hashlib

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    contract = {
        "format": "coffee_detector.af2_spectral.run_contract.v1",
        "arm": "AF2WIN",
        "seed": 42,
        "config_sha256": digest(config),
        "d0_checkpoint_sha256": digest(checkpoint),
        "epochs": 50,
        "test_images_accessed": False,
    }
    (prior / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (prior / "weights.txt").write_text("not a generic checkpoint lookup", encoding="utf-8")

    restored = restore_spectral_kaggle_run(
        input_root, output_root, arm="AF2WIN", seed=42, d0_checkpoint=checkpoint, config=config
    )

    assert restored == output_root / "AF2WIN/AF2WIN_seed42"
    assert (restored / "weights.txt").is_file()
    assert restore_spectral_kaggle_run(
        input_root, output_root, arm="AF2ORI", seed=42, d0_checkpoint=checkpoint, config=config
    ) is None
