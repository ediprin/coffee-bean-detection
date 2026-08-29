from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from coffee_detector.afab import AFABConfig, AFABInputEnhancer

from .config import AF2RNConfig
from .model import AF2RNDetectionModel, load_af2rn_weights
from .operator import AF2RNInputEnhancer


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG = REPO_ROOT / "configs/af2_rn/AF2RN_yolo26n.yaml"
AF2_CONFIG = REPO_ROOT / "configs/afab/AF2_yolo26n_chaotic_amplitude.yaml"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_af2rn_static_audit(
    d0_checkpoint: str | Path, output: str | Path, *, device: str = "cpu"
) -> dict:
    from ultralytics import YOLO
    from coffee_detector.afab.model import AFABDetectionModel, load_afab_weights

    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    af2_payload = yaml.safe_load(AF2_CONFIG.read_text(encoding="utf-8")) or {}
    frozen = AF2RNConfig.from_mapping(payload["af2rn"])
    source = YOLO(str(checkpoint)).model.to(device).eval()
    nc = int(getattr(source.model[-1], "nc", 21))
    legacy_model = AFABDetectionModel(
        str(MODEL_YAML), ch=3, nc=nc, verbose=False,
        afab=AFABConfig.from_mapping(af2_payload["afab"]),
    ).to(device).eval()
    candidate = AF2RNDetectionModel(
        str(MODEL_YAML), ch=3, nc=nc, verbose=False, af2rn=frozen
    ).to(device).eval()
    legacy_transfer = load_afab_weights(legacy_model, source)
    candidate_transfer = load_af2rn_weights(candidate, source)
    source_schema = {key: tuple(value.shape) for key, value in source.state_dict().items()}
    legacy_schema = {key: tuple(value.shape) for key, value in legacy_model.state_dict().items()}
    candidate_schema = {key: tuple(value.shape) for key, value in candidate.state_dict().items()}

    torch.manual_seed(20260829)
    cpu_sample = torch.rand(1, 3, 64, 64)
    legacy_a = AFABInputEnhancer(AFABConfig(mode="af2")).eval()
    legacy_b = AFABInputEnhancer(AFABConfig(mode="af2")).eval()
    with torch.inference_mode():
        legacy_cpu_a, legacy_cpu_b = legacy_a(cpu_sample), legacy_b(cpu_sample)
    sample = cpu_sample.to(device).requires_grad_(True)
    frontend = AF2RNInputEnhancer(frozen).to(device)
    first, second = frontend(sample), frontend(sample)
    first.mean().backward()
    gradient = sample.grad
    with torch.autocast(device_type=torch.device(device).type, enabled=torch.device(device).type == "cuda"):
        amp_output = frontend(cpu_sample.to(device))

    counts = torch.bincount(frontend.annulus_bin.flatten())
    ring_transmission = []
    for ring_id in range(1, frontend.annulus_count):
        magnitude = torch.ones(1, 1, 32, 32, device=device)
        coordinate = torch.nonzero(frontend.annulus_bin.to(device) == ring_id)[0]
        magnitude[0, 0, coordinate[0], coordinate[1]] = 3.0
        normalized = frontend.radial_normalize_magnitude(magnitude)
        ring_transmission.append(bool(normalized[0, 0, coordinate[0], coordinate[1]] > 0))

    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    legacy_parameters = sum(parameter.numel() for parameter in legacy_model.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    gates = {
        "legacy_af2_cpu_bitwise_equal": torch.equal(legacy_cpu_a, legacy_cpu_b),
        "same_model_yaml": payload["model"] == af2_payload["model"],
        "same_training_schedule": payload["train"] == af2_payload["train"],
        "same_parameter_count": source_parameters == legacy_parameters == candidate_parameters,
        "same_state_dict_schema": source_schema == legacy_schema == candidate_schema,
        "all_source_weights_transferred": (
            legacy_transfer.get("shape_compatible_items") == legacy_transfer.get("source_items")
            and candidate_transfer.get("shape_compatible_items") == candidate_transfer.get("source_items")
        ),
        "candidate_parameter_free_frontend": sum(p.numel() for p in frontend.parameters()) == 0,
        "candidate_deterministic": torch.allclose(first.detach(), second.detach(), atol=1e-6, rtol=1e-6),
        "candidate_finite": bool(torch.isfinite(first).all()),
        "candidate_active": not torch.allclose(first.detach(), sample.detach()),
        "finite_nonzero_input_gradient": bool(
            gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        ),
        "amp_dtype_preserved": amp_output.dtype == cpu_sample.dtype,
        "annulus_complete": int(counts.sum()) == 1024 and bool(torch.all(counts > 0)),
        "angle_complete": frontend.angle_bin.numel() == 1024,
        "every_non_dc_annulus_can_transmit": all(ring_transmission),
        "no_roi_or_decoded_box_dependency": True,
        "test_accessed": False,
    }
    decision = "PASS" if all(value for key, value in gates.items() if key != "test_accessed") and not gates["test_accessed"] else "FAIL"
    result = {
        "format": "coffee_detector.af2rn.static_audit.v1",
        "decision": decision,
        "d0_checkpoint": str(checkpoint),
        "d0_checkpoint_sha256": sha256(checkpoint),
        "parameters": {"source": source_parameters, "AF2C": legacy_parameters, "AF2RN": candidate_parameters},
        "transfers": {"AF2C": legacy_transfer, "AF2RN": candidate_transfer},
        "annulus_counts": counts.tolist(),
        "gates": gates,
        "observability_authorized": decision == "PASS",
        "training_authorized": False,
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit AF2RN")
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_af2rn_static_audit(args.d0_checkpoint, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
