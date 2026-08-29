from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer


DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
EXPECTED_AF2 = {
    "mode": "af2",
    "patch_size": 32,
    "overlap": 0.50,
    "radius_ratio": 0.05,
    "gamma": 0.10,
    "angular_bins": 360,
    "chunk_size": 128,
    "eps": 1.0e-8,
}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _prepare_dfine_import(dfine_root: Path) -> None:
    # D-FINE exposes its top-level package as `src`. Put the pinned checkout first.
    root_text = str(dfine_root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    importlib.invalidate_caches()


def build_loaded_model(config_path: Path, checkpoint: Path, seed: int) -> torch.nn.Module:
    from src.core import YAMLConfig
    from src.solver import TASKS

    # This reset is essential: the 21-class score heads are shape-incompatible with
    # the COCO checkpoint and therefore remain freshly initialized by D-FINE.
    torch.manual_seed(int(seed))
    cfg = YAMLConfig(str(config_path), tuning=str(checkpoint), device="cpu", use_amp=False)
    model = cfg.model
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.model = model
    solver.load_tuning_state(str(checkpoint))
    return model


def run_preflight(
    *,
    dfine_root: str | Path,
    native_config: str | Path,
    af2_config: str | Path,
    pretrained_checkpoint: str | Path,
    preparation_report: str | Path,
    output: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    root = Path(dfine_root).expanduser().resolve()
    native_cfg = Path(native_config).expanduser().resolve()
    candidate_cfg = Path(af2_config).expanduser().resolve()
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    prep_path = Path(preparation_report).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()

    for path in (native_cfg, candidate_cfg, checkpoint, prep_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    head = git_head(root)
    prep = load_json(prep_path)
    dataset = prep.get("dataset", {})
    splits = dataset.get("splits", {})

    test_absent = True
    for split_name in ("train", "val"):
        images_dir = Path(splits[split_name]["images_dir"])
        # Grouped development archive convention: sibling test must not exist.
        if (images_dir.parent / "test").exists():
            test_absent = False

    _prepare_dfine_import(root)

    # Build each arm independently from the same RNG state and checkpoint.
    native = build_loaded_model(native_cfg, checkpoint, seed)
    candidate = build_loaded_model(candidate_cfg, checkpoint, seed)

    native_state = native.state_dict()
    candidate_state = candidate.state_dict()
    keys_exact = list(native_state.keys()) == list(candidate_state.keys())
    tensors_exact = bool(
        keys_exact
        and all(
            torch.equal(native_state[key].detach().cpu(), candidate_state[key].detach().cpu())
            for key in native_state
        )
    )

    native_params = parameter_count(native)
    candidate_params = parameter_count(candidate)

    frontend = AFABInputEnhancer(AFABConfig.from_mapping(EXPECTED_AF2))
    frontend_params = parameter_count(frontend)
    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64, dtype=torch.float32).reshape(1, 3, 64, 64)
    with torch.inference_mode():
        enhanced = frontend(probe)
    probe_ok = {
        "shape_preserved": tuple(enhanced.shape) == tuple(probe.shape),
        "finite": bool(torch.isfinite(enhanced).all().item()),
        "nonzero_change": float((enhanced - probe).abs().max().item()) > 0.0,
        "max_abs_change": float((enhanced - probe).abs().max().item()),
    }

    checkpoint_sha = sha256(checkpoint)
    train_manifest = Path(splits["train"]["path"])
    val_manifest = Path(splits["val"]["path"])

    gates = {
        "dfine_commit_exact": head == DFINE_COMMIT,
        "seed_is_frozen_42": int(seed) == 42,
        "pretrained_checkpoint_present": checkpoint.is_file(),
        "same_dataset_manifest_contract": (
            sha256(train_manifest) == splits["train"]["sha256"]
            and sha256(val_manifest) == splits["val"]["sha256"]
        ),
        "dataset_counts_exact": (
            splits["train"]["images"] == 1665
            and splits["train"]["annotations"] == 2986
            and splits["val"]["images"] == 294
            and splits["val"]["annotations"] == 526
        ),
        "test_absent": test_absent,
        "detector_state_keys_exact": keys_exact,
        "detector_state_tensors_exact": tensors_exact,
        "detector_parameter_count_exact": native_params == candidate_params,
        "af2_learned_parameters_zero": frontend_params == 0,
        "af2_probe_shape_preserved": probe_ok["shape_preserved"],
        "af2_probe_finite": probe_ok["finite"],
        "af2_probe_live": probe_ok["nonzero_change"],
    }
    decision = "PASS" if all(gates.values()) else "FAIL"

    payload = {
        "format": "coffee_detector.af2_dfine_transfer.static_preflight.v1",
        "protocol": "faruq-v3-af2-dfine-n-transfer-seed42-v1",
        "decision": decision,
        "training_authorized": decision == "PASS",
        "seed": int(seed),
        "test_images_accessed": False,
        "dfine_root": str(root),
        "dfine_commit": head,
        "native_config": str(native_cfg),
        "native_config_sha256": sha256(native_cfg),
        "af2_config": str(candidate_cfg),
        "af2_config_sha256": sha256(candidate_cfg),
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": checkpoint_sha,
        "native_parameter_count": native_params,
        "candidate_parameter_count": candidate_params,
        "common_initialized_detector_state_sha256": state_fingerprint(native),
        "candidate_initialized_detector_state_sha256": state_fingerprint(candidate),
        "af2_learned_parameter_count": frontend_params,
        "af2_probe": probe_ok,
        "dataset_train_manifest_sha256": sha256(train_manifest),
        "dataset_val_manifest_sha256": sha256(val_manifest),
        "gates": gates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict AF2 × D-FINE-N seed-42 static preflight")
    p.add_argument("--dfine-root", required=True)
    p.add_argument("--native-config", required=True)
    p.add_argument("--af2-config", required=True)
    p.add_argument("--pretrained-checkpoint", required=True)
    p.add_argument("--preparation-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    result = run_preflight(
        dfine_root=args.dfine_root,
        native_config=args.native_config,
        af2_config=args.af2_config,
        pretrained_checkpoint=args.pretrained_checkpoint,
        preparation_report=args.preparation_report,
        output=args.output,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if result["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
