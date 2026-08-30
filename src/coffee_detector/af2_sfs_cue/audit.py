from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from coffee_detector.experiments.run_faruq_v3_af2_direct import (
    AF2_CONFIG,
    MODEL_YAML,
    OFFICIAL_YOLO26N_SHA256,
    _build_initialized_detector,
    _load_yaml,
    _require_official_pretrained,
)
from coffee_detector.afab.operator import AFABConfig

from .config import AF2SFSCUEConfig
from .model import (
    AF2SFSCUEDetectHead,
    AF2SFSCUEDetectionModel,
    canonical_native_state,
    factorized_dual_cue_loss,
    load_af2_sfs_cue_weights,
)


def _normalize_device(value: str | int | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        return torch.device(f"cuda:{value}")
    return torch.device(str(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _flatten(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result: list[torch.Tensor] = []
        for key in sorted(value):
            result.extend(_flatten(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return []


def _max_abs(left: Any, right: Any) -> float:
    first, second = _flatten(left), _flatten(right)
    if len(first) != len(second):
        return float("inf")
    return max(
        (float((a.float() - b.float()).abs().max().item()) for a, b in zip(first, second)),
        default=0.0,
    )


def _build_candidate(
    checkpoint: Path, afab: AFABConfig, combo: AF2SFSCUEConfig, seed: int
) -> tuple[AF2SFSCUEDetectionModel, dict[str, int]]:
    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = AF2SFSCUEDetectionModel(
            str(MODEL_YAML), nc=21, verbose=False, afab=afab, sfs_cue=combo
        )
        transfer = load_af2_sfs_cue_weights(model, source)
    return model, transfer


def run_af2_sfs_cue_direct_static_audit(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    device: str | int | torch.device = "cpu",
) -> dict:
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    if seed != 42:
        raise ValueError("Screen pertama dikunci pada seed 42")
    config_path = (
        Path(__file__).resolve().parents[3]
        / "configs/af2_sfs_cue_direct/AF2SFSCUE1_yolo26n.yaml"
    )
    payload = _load_yaml(config_path)
    af2_payload = _load_yaml(AF2_CONFIG)
    if payload.get("afab") != af2_payload.get("afab"):
        raise RuntimeError("AF2 kandidat berbeda dari AF2 direct yang dibekukan")
    if payload.get("train") != af2_payload.get("train"):
        raise RuntimeError("Schedule kandidat berbeda dari AF2 direct")
    afab = AFABConfig.from_mapping(payload["afab"])
    combo = AF2SFSCUEConfig.from_mapping(payload["sfs_cue"])

    reference, _ = _build_initialized_detector(
        use_af2=True,
        pretrained_checkpoint=checkpoint,
        af2_config=afab,
        seed=seed,
        verbose=False,
    )
    candidate, transfer = _build_candidate(checkpoint, afab, combo, seed)
    reference_state = reference.state_dict()
    candidate_native = canonical_native_state(candidate)
    native_keys_exact = list(reference_state) == list(candidate_native)
    native_tensors_exact = bool(
        native_keys_exact
        and all(
            torch.equal(reference_state[key].cpu(), candidate_native[key].cpu())
            for key in reference_state
        )
    )
    native_fingerprint = _state_fingerprint(candidate_native)

    target_device = _normalize_device(device)
    reference = reference.to(target_device).eval()
    candidate = candidate.to(target_device).eval()
    sample = torch.linspace(
        0.0, 1.0, 3 * 64 * 64, device=target_device, dtype=torch.float32
    ).reshape(1, 3, 64, 64)
    with torch.inference_mode():
        expected = reference(sample)
        observed = candidate(sample)
    initial_diff = _max_abs(expected, observed)

    head = candidate.model[-1]
    if not isinstance(head, AF2SFSCUEDetectHead):
        raise TypeError("Head kandidat salah")
    training_added = sum(parameter.numel() for parameter in head.parameters()) - sum(
        parameter.numel() for parameter in head.base_head.parameters()
    )
    inference_added = sum(parameter.numel() for parameter in head.adapter.parameters())

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    _ = candidate(sample)
    predictions = head.last_auxiliary_predictions
    target = candidate.last_af2_gate_target
    raw = candidate.last_raw_input_target
    signal = candidate.last_af2_signal_target
    if predictions is None or target is None or raw is None or signal is None:
        raise RuntimeError("Factorized CUE/SPDS tidak aktif saat training")
    cue_reads_pre_adapter = head.last_pre_adapter_features is not None
    auxiliary, gate_loss, signal_loss = factorized_dual_cue_loss(
        predictions,
        target,
        raw,
        signal,
        signal_mix=combo.signal_mix,
    )
    auxiliary.backward()
    decoder_gradients = [p.grad for p in head.decoders.parameters()]
    cue_gradients_valid = bool(
        decoder_gradients
        and all(g is not None and torch.isfinite(g).all() for g in decoder_gradients)
        and any(torch.count_nonzero(g) > 0 for g in decoder_gradients if g is not None)
    )

    candidate.eval()
    original_output_weight = head.adapter.output.weight.detach().clone()
    with torch.no_grad():
        torch.nn.init.constant_(head.adapter.output.weight, 0.01)
    with torch.inference_mode():
        active_output = candidate(sample)
    active_diff = _max_abs(observed, active_output)
    with torch.no_grad():
        head.adapter.output.weight.copy_(original_output_weight)
    with torch.inference_mode():
        _ = candidate(sample)
    cue_inactive_at_inference = head.last_auxiliary_predictions is None

    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "same_af2_frontend_as_direct": payload.get("afab") == af2_payload.get("afab"),
        "same_50_epoch_schedule_as_direct": payload.get("train") == af2_payload.get("train"),
        "native_state_keys_exact": native_keys_exact,
        "native_state_tensors_exact": native_tensors_exact,
        "initial_detector_output_exact": initial_diff == 0.0,
        "sfs_identity_initialized": bool(torch.count_nonzero(head.adapter.output.weight) == 0),
        "sfs_active_changes_detector_output": active_diff > 0.0,
        "cue_target_is_pure_normalized_gate": tuple(target.shape) == tuple(sample.shape),
        "spds_target_equals_raw_times_gate": bool(torch.equal(signal, raw * target)),
        "gate_and_signal_losses_finite": bool(
            torch.isfinite(gate_loss) and torch.isfinite(signal_loss)
        ),
        "single_decoder_factorization": len(head.decoders) == 3,
        "cue_reads_pre_adapter_features": cue_reads_pre_adapter,
        "cue_gradients_finite_nonzero": cue_gradients_valid,
        "cue_inactive_during_inference": cue_inactive_at_inference,
        "test_not_accessed": True,
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    result = {
        "format": "coffee_detector.af2_sfs_cue_direct.static_audit.v1",
        "protocol": "faruq-v3-af2-sfs-cue-direct-seed42-v1",
        "seed": seed,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_images_accessed": False,
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": pretrained_sha,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "native_initial_state_sha256": native_fingerprint,
        "reference_parameter_count": sum(p.numel() for p in reference.parameters()),
        "candidate_training_parameter_count": sum(p.numel() for p in candidate.parameters()),
        "training_only_added_parameters": training_added - inference_added,
        "inference_added_parameters": inference_added,
        "initial_output_max_abs_diff": initial_diff,
        "active_output_max_abs_diff": active_diff,
        "weight_transfer": transfer,
        "gates": gates,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
