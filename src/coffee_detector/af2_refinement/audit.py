from __future__ import annotations

import json
from pathlib import Path

import torch

from coffee_detector.af2_spectral import SpectralInputEnhancer, frozen_arm_config
from coffee_detector.af2_spectral.audit import sha256
from coffee_detector.afab import AFABConfig, AFABInputEnhancer

from .config import TRAIN_ARMS, frozen_refinement_config
from .operator import AF2RefinementInputEnhancer


REPEAT_ATOL = 1.0e-6
REPEAT_RTOL = 1.0e-6


def run_af2_refinement_static_audit(
    d0_checkpoint: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    """Verify frozen mechanics before any AF2 refinement training is allowed."""

    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model.cpu().eval()
    nc = int(getattr(model.model[-1], "nc", -1))
    checkpoint_gate = nc == 21

    torch.manual_seed(42018)
    cpu_probe = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    legacy = AFABInputEnhancer(
        AFABConfig(
            mode="af2",
            patch_size=32,
            overlap=0.50,
            gamma=0.10,
            angular_bins=360,
        )
    )
    control = AF2RefinementInputEnhancer(frozen_refinement_config("AF2C"))
    control_bitwise = torch.equal(control(cpu_probe), legacy(cpu_probe))

    radial = AF2RefinementInputEnhancer(frozen_refinement_config("AF2RAD"))
    radial_geometry = {
        "angular_bins": radial.radial.config.angular_bins,
        "radial_bands": radial.radial.radial_bands,
        "radial_values": sorted(radial.radial.radial_bin.unique().tolist()),
        "angle_min": int(radial.radial.angle_bin.min().item()),
        "angle_max": int(radial.radial.angle_bin.max().item()),
    }
    radial_gate = (
        radial_geometry["angular_bins"] == 360
        and radial_geometry["radial_bands"] == 3
        and radial_geometry["radial_values"] == [0, 1, 2]
        and radial_geometry["angle_min"] == 0
        and radial_geometry["angle_max"] == 359
    )

    torch.manual_seed(42019)
    wav_probe = torch.rand(1, 3, 65, 63, dtype=torch.float32)
    reference_wav = SpectralInputEnhancer(frozen_arm_config("WAV1")).recover(wav_probe)
    refinement_wav = AF2RefinementInputEnhancer(
        frozen_refinement_config("AF2WAV")
    ).wavelet_cue(wav_probe)
    wavelet_equivalent = torch.equal(reference_wav, refinement_wav)

    base = AF2RefinementInputEnhancer(frozen_refinement_config("AF2WAV")).base_cue(
        wav_probe
    )
    fused = AF2RefinementInputEnhancer(frozen_refinement_config("AF2WAV")).fused_cue(
        wav_probe
    )
    max_fusion_gate = bool(torch.all(fused + 1.0e-8 >= base).item())

    torch_device = torch.device(device)
    arms: dict[str, dict] = {}
    previous_det = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        for index, arm in enumerate(TRAIN_ARMS):
            torch.manual_seed(42100 + index)
            probe = torch.rand(
                1, 3, 64, 64, device=torch_device, dtype=torch.float32, requires_grad=True
            )
            frontend = AF2RefinementInputEnhancer(
                frozen_refinement_config(arm)
            ).to(torch_device)
            first = frontend(probe)
            second = frontend(probe)
            max_difference = float((first.detach() - second.detach()).abs().max().item())
            repeat = torch.allclose(
                first.detach(),
                second.detach(),
                atol=REPEAT_ATOL,
                rtol=REPEAT_RTOL,
            )
            finite_output = bool(torch.isfinite(first).all().item())
            first.mean().backward()
            finite_gradient = probe.grad is not None and bool(
                torch.isfinite(probe.grad).all().item()
            )
            no_trainable_parameters = not any(
                parameter.requires_grad for parameter in frontend.parameters()
            )
            no_persistent_state = len(frontend.state_dict()) == 0
            failed = [
                name
                for name, passed in {
                    "repeatable": repeat,
                    "finite_output": finite_output,
                    "finite_gradient": finite_gradient,
                    "no_trainable_parameters": no_trainable_parameters,
                    "no_persistent_state": no_persistent_state,
                }.items()
                if not passed
            ]
            arms[arm] = {
                "repeatable": repeat,
                "max_repeat_difference": max_difference,
                "repeat_atol": REPEAT_ATOL,
                "repeat_rtol": REPEAT_RTOL,
                "finite_output": finite_output,
                "finite_gradient": finite_gradient,
                "no_trainable_parameters": no_trainable_parameters,
                "no_persistent_state": no_persistent_state,
                "failed_gates": failed,
            }
    finally:
        torch.use_deterministic_algorithms(previous_det)

    gates = {
        "d0_is_sni21": checkpoint_gate,
        "legacy_af2c_bitwise_equal": control_bitwise,
        "radial_is_3x360": radial_gate,
        "wav1_cue_bitwise_equal": wavelet_equivalent,
        "max_fusion_never_attenuates_af2_cue": max_fusion_gate,
        "all_arm_gates_pass": all(not entry["failed_gates"] for entry in arms.values()),
        "test_accessed": False,
    }
    decision = "PASS" if all(value for key, value in gates.items() if key != "test_accessed") and gates["test_accessed"] is False else "FAIL"
    result = {
        "format": "coffee_detector.af2_refinement.static_audit.v1",
        "d0_checkpoint": str(checkpoint),
        "d0_checkpoint_sha256": sha256(checkpoint),
        "d0_nc": nc,
        "device": str(torch_device),
        "gates": gates,
        "radial_geometry": radial_geometry,
        "arms": arms,
        "arm_authorization": {
            arm: decision == "PASS" and not entry["failed_gates"]
            for arm, entry in arms.items()
        },
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result
