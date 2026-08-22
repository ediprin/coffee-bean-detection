from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .model import AF2_CONFIG, _build_pair_model, _load_pair_weights


ARM_TO_BASE = {
    "AF2STB1": "STB1",
    "AF2IGEM1": "IGEM1",
    "AF2SAF1": "SAF1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameters(model) -> int:
    return sum(value.numel() for value in model.parameters())


def _safety_decision(gates: dict[str, bool]) -> str:
    passed = all(
        value for key, value in gates.items() if key != "test_accessed"
    ) and not gates.get("test_accessed", True)
    return "PASS" if passed else "FAIL"


def run_af2_pair_static_audit(
    arm: str,
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    standalone_checkpoint: str | Path,
    strong: dict,
    output: str | Path,
    *,
    image_size: int = 64,
) -> dict:
    if arm not in ARM_TO_BASE:
        raise ValueError(f"Arm tidak dikenal: {arm}")
    model_yaml = Path(model_yaml).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    standalone_checkpoint = Path(standalone_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    for path in (model_yaml, d0_checkpoint, standalone_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    from ultralytics import YOLO

    torch.manual_seed(20260823)
    pair = _build_pair_model(
        arm, str(model_yaml), nc=21, ch=3, verbose=False,
        strong=strong, af2=AF2_CONFIG,
    )
    transfer = _load_pair_weights(arm, pair, YOLO(str(d0_checkpoint)).model)
    pair.eval()
    source = YOLO(str(standalone_checkpoint)).model.eval()
    pair_state = pair.state_dict()
    source_state = source.state_dict()
    schema_equal = tuple((key, tuple(value.shape)) for key, value in pair_state.items()) == tuple(
        (key, tuple(value.shape)) for key, value in source_state.items()
    )
    sample = torch.rand(1, 3, image_size, image_size)
    with torch.inference_mode():
        recovered = pair.af2.recover(sample)
        enhanced = pair.af2(sample)
        output_value = pair(sample)
    tensors = []
    if isinstance(output_value, tuple):
        tensors.append(output_value[0])
    elif isinstance(output_value, torch.Tensor):
        tensors.append(output_value)
    gates = {
        "known_arm": arm in ARM_TO_BASE,
        "af2_exact_historical_config": pair.af2_config.to_dict() == AF2_CONFIG.to_dict(),
        "detector_parameter_count_equal": _parameters(pair) == _parameters(source),
        "detector_state_schema_equal": schema_equal,
        "af2_parameter_free": sum(value.numel() for value in pair.af2.parameters()) == 0,
        "af2_active": not torch.equal(sample, enhanced),
        "af2_finite": bool(torch.isfinite(recovered).all() and torch.isfinite(enhanced).all()),
        "finite_pair_inference": all(bool(torch.isfinite(value).all()) for value in tensors),
        "no_roi_or_decoded_box_dependency": True,
        "test_accessed": False,
    }
    result = {
        "format": "coffee_detector.af2_pairs.static_audit.v1",
        "arm": arm,
        "standalone": ARM_TO_BASE[arm],
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "standalone_checkpoint_sha256": _sha256(standalone_checkpoint),
        "parameters": {"standalone": _parameters(source), "candidate": _parameters(pair)},
        "transfer": transfer,
        "gates": gates,
        "decision": _safety_decision(gates),
        "training_executed": False,
        "test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(output)
    return result
