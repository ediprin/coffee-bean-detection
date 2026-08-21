"""Build and validate the Kaggle input contract for top-control confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from coffee_detector.experiments.prepare_faruq_v3_kaggle import (
    ARCHIVE_BYTES,
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    prepare_faruq_v3_kaggle_input,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_NAME = "top_controls_kaggle_manifest.json"
MANIFEST_FORMAT = "coffee_detector.top_controls.kaggle_manifest.v1"
CORE_MANIFEST_NAME = "top_controls_canonical_core_manifest.json"
CORE_MANIFEST_FORMAT = "coffee_detector.top_controls.canonical_kaggle_core.v1"
ARTIFACTS = {
    "STB1_seed123_best.pt": "experiments/faruq-v3-stb-paired-confirmation-v1/STB1/STB1_seed123/weights/best.pt",
    "STB1_seed2026_best.pt": "experiments/faruq-v3-stb-paired-confirmation-v1/STB1/STB1_seed2026/weights/best.pt",
    "AF2_seed123_best.pt": "experiments/faruq-v3-af2-igem-paired-confirmation-v1/AF2/AF2_seed123/weights/best.pt",
    "AF2_seed2026_best.pt": "experiments/faruq-v3-af2-igem-paired-confirmation-v1/AF2/AF2_seed2026/weights/best.pt",
    "stb_capacity_paired_confirmation.json": "experiments/faruq-v3-stb-paired-confirmation-v1/val_reports/stb_capacity_paired_confirmation.json",
    "af2_igem_paired_confirmation.json": "experiments/faruq-v3-af2-igem-paired-confirmation-v1/val_reports/af2_igem_paired_confirmation.json",
    "af2_continuation_paired_confirmation.json": "experiments/faruq-v3-af2-continuation-paired-v1/val_reports/af2_continuation_paired_confirmation.json",
    "FCT0_seed42_val.json": "experiments/faruq-v3-fcstb-distillation-v1/val_reports/FCT0_seed42_val.json",
}
CHECKPOINT_SEEDS = {
    "STB1_seed123_best.pt": 123,
    "STB1_seed2026_best.pt": 2026,
    "AF2_seed123_best.pt": 123,
    "AF2_seed2026_best.pt": 2026,
}
CONFIGS = {
    "FCT0": REPO_ROOT / "configs/fcstb/FCT0_stb_joint_control.yaml",
    "AF2R0": REPO_ROOT / "configs/af2r/AF2R0_yolo26n_zero_control.yaml",
    "AF2R1": REPO_ROOT / "configs/af2r/AF2R1_yolo26n_illumination_gate.yaml",
    "AF2CAL3": REPO_ROOT / "configs/af2cal/AF2CAL3_yolo26n.yaml",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_seed(path: Path) -> int:
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(train_args, dict) or "seed" not in train_args:
        raise RuntimeError(f"Checkpoint tidak merekam seed: {path}")
    return int(train_args["seed"])


def _validate_yolo_checkpoint(path: Path, expected_seed: int) -> dict:
    from ultralytics import YOLO

    if path.suffix.lower() != ".pt" or path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Checkpoint tidak valid/terlalu kecil: {path}")
    actual_seed = _checkpoint_seed(path)
    if actual_seed != expected_seed:
        raise RuntimeError(
            f"Checkpoint {path.name} bukan seed {expected_seed}: {actual_seed}"
        )
    try:
        model = YOLO(str(path)).model.cpu().eval()
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint tidak dapat di-load Ultralytics: {path}: {exc}"
        ) from exc
    nc = int(getattr(model.model[-1], "nc", -1))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if nc != 21 or parameters <= 0:
        raise RuntimeError(
            f"Checkpoint bukan detektor SNI-21 valid: nc={nc}, params={parameters}"
        )
    return {
        "loadable_by_ultralytics": True,
        "seed": actual_seed,
        "nc": nc,
        "parameters": parameters,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_top_controls_kaggle_bundle(
    project_root: str | Path, output_root: str | Path
) -> dict:
    project_root = Path(project_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, relative in ARTIFACTS.items():
        source = project_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Artefak proyek tidak ditemukan: {source}")
        target = output_root / name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        records[name] = {
            "source": relative,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        }
    manifest = {
        "format": MANIFEST_FORMAT,
        "artifacts": records,
        "checkpoint_seeds": CHECKPOINT_SEEDS,
        "test_images_included": False,
    }
    path = output_root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(path)
    return manifest


def build_top_controls_canonical_kaggle_core(
    project_root: str | Path, output_root: str | Path
) -> dict:
    """Build the canonical single Kaggle core used by a fresh account.

    The immutable grouped archive stays opaque as ``.tar.bin``. Completed
    training runs are deliberately excluded; resume state belongs to Kaggle
    Saved Version outputs, not to the core dataset.
    """

    project_root = Path(project_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    top_controls = build_top_controls_kaggle_bundle(project_root, output_root)

    archive_source = project_root / "bundles/faruq-development-v3-grouped.tar"
    if not archive_source.is_file():
        raise FileNotFoundError(f"Arsip Faruq-v3 tidak ditemukan: {archive_source}")
    if (
        archive_source.stat().st_size != ARCHIVE_BYTES
        or sha256(archive_source) != ARCHIVE_SHA256
    ):
        raise RuntimeError("Arsip Faruq-v3 berbeda dari kontrak canonical")
    archive_target = output_root / ARCHIVE_NAME
    shutil.copy2(archive_source, archive_target)
    if (
        archive_target.stat().st_size != ARCHIVE_BYTES
        or sha256(archive_target) != ARCHIVE_SHA256
    ):
        raise RuntimeError("Copy arsip opaque mengubah ukuran/SHA")

    checkpoint_validation = {
        name: _validate_yolo_checkpoint(output_root / name, expected_seed)
        for name, expected_seed in CHECKPOINT_SEEDS.items()
    }
    manifest_path = output_root / MANIFEST_NAME
    core = {
        "format": CORE_MANIFEST_FORMAT,
        "canonical_dataset_slug": "faruq-v3-experiment-core-v1",
        "archive": {
            "name": ARCHIVE_NAME,
            "bytes": ARCHIVE_BYTES,
            "sha256": ARCHIVE_SHA256,
            "opaque": True,
        },
        "top_controls_manifest": {
            "name": MANIFEST_NAME,
            "sha256": sha256(manifest_path),
            "format": top_controls["format"],
        },
        "checkpoint_validation": checkpoint_validation,
        "completed_training_runs_included": False,
        "resume_source": "kaggle_saved_version_only",
        "test_images_included": False,
    }
    path = output_root / CORE_MANIFEST_NAME
    path.write_text(json.dumps(core, indent=2) + "\n", encoding="utf-8")
    core["manifest"] = str(path)
    return core


def _validate_summaries(artifacts: dict[str, Path]) -> None:
    summaries = {
        name: json.loads(artifacts[name].read_text(encoding="utf-8"))
        for name in (
            "stb_capacity_paired_confirmation.json",
            "af2_igem_paired_confirmation.json",
            "af2_continuation_paired_confirmation.json",
        )
    }
    expected = {
        "stb_capacity_paired_confirmation.json": (
            "protocol",
            "faruq-v3-stb-capacity-paired-confirmation-v1",
        ),
        "af2_igem_paired_confirmation.json": (
            "protocol",
            "faruq-v3-af2-igem-paired-validation-confirmation-v1",
        ),
        "af2_continuation_paired_confirmation.json": (
            "format",
            "coffee_detector.af2_continuation.paired_confirmation.v1",
        ),
    }
    for name, payload in summaries.items():
        key, value = expected[name]
        if (
            payload.get(key) != value
            or payload.get("seeds") != [42, 123, 2026]
            or payload.get("test_images_accessed") is not False
            or payload.get("test_opened") is not False
        ):
            raise RuntimeError(f"Summary Kaggle tidak kompatibel/test-lock gagal: {name}")
    fct0 = json.loads(artifacts["FCT0_seed42_val.json"].read_text(encoding="utf-8"))
    metrics = fct0.get("metrics", {})
    if metrics.get("classes_without_ground_truth") or not all(
        key in metrics
        for key in (
            "macro_map50_95",
            "bottom3_class_map50_95",
            "worst_class_map50_95",
        )
    ):
        raise RuntimeError("Evidence FCT0 seed 42 tidak lengkap")


def prepare_top_controls_kaggle_input(
    input_root: str | Path, work_root: str | Path
) -> tuple[Path, dict[str, Path], dict]:
    input_root = Path(input_root).expanduser().resolve()
    work_root = Path(work_root).expanduser().resolve()
    matches = sorted(input_root.rglob(MANIFEST_NAME))
    if len(matches) != 1:
        raise FileNotFoundError(f"Harus ada satu {MANIFEST_NAME}; ditemukan {matches}")
    manifest = json.loads(matches[0].read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise RuntimeError("Format manifest top-controls Kaggle tidak kompatibel")
    if manifest.get("test_images_included") is not False:
        raise RuntimeError("Bundle top-controls tidak mempertahankan test lock")
    if manifest.get("checkpoint_seeds") != CHECKPOINT_SEEDS:
        raise RuntimeError("Kontrak checkpoint seed tidak lengkap")

    artifacts = {}
    for name, contract in manifest.get("artifacts", {}).items():
        candidates = sorted(input_root.rglob(name))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Harus ada tepat satu {name}; ditemukan {candidates}")
        path = candidates[0]
        if path.stat().st_size != int(contract["bytes"]) or sha256(path) != contract["sha256"]:
            raise RuntimeError(f"Kontrak SHA/ukuran gagal: {name}")
        artifacts[name] = path
    if set(artifacts) != set(ARTIFACTS):
        raise RuntimeError("Bundle top-controls tidak memuat seluruh artefak")
    for name, expected_seed in CHECKPOINT_SEEDS.items():
        if _checkpoint_seed(artifacts[name]) != expected_seed:
            raise RuntimeError(f"Checkpoint {name} bukan seed {expected_seed}")
    _validate_summaries(artifacts)

    data_root, dataset_contract = prepare_faruq_v3_kaggle_input(input_root, work_root)
    contract = {
        "format": "coffee_detector.top_controls.kaggle_input.v1",
        "dataset": dataset_contract,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "manifest_sha256": sha256(matches[0]),
        "test_images_accessed": False,
        "decision": "PASS",
    }
    path = work_root / "top_controls_kaggle_input_contract.json"
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    return data_root, artifacts, contract


def run_directory(output_root: str | Path, arm: str, seed: int) -> Path:
    output_root = Path(output_root).expanduser().resolve()
    if arm == "FCT0":
        return output_root / f"FCT0_seed{seed}"
    if arm in {"AF2R0", "AF2R1", "AF2CAL3"}:
        return output_root / arm / f"{arm}_seed{seed}"
    raise ValueError(f"Arm tidak dikenal: {arm}")


def expected_run_contract(
    arm: str, seed: int, source_checkpoint: str | Path
) -> dict:
    if arm not in CONFIGS or seed not in (123, 2026):
        raise ValueError(f"Kontrak arm/seed tidak valid: {arm}/{seed}")
    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    return {
        "format": "coffee_detector.top_controls.kaggle_run_contract.v1",
        "arm": arm,
        "seed": seed,
        "config_sha256": sha256(CONFIGS[arm]),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "epochs": int(payload["train"]["epochs"]),
        "test_images_accessed": False,
    }


def ensure_run_contract(
    output_root: str | Path,
    *,
    arm: str,
    seed: int,
    source_checkpoint: str | Path,
) -> Path:
    destination = run_directory(output_root, arm, seed)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "run_contract.json"
    expected = expected_run_contract(arm, seed, source_checkpoint)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError(f"Run directory berisi kontrak berbeda: {destination}")
    path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return path


def restore_top_control_kaggle_run(
    input_root: str | Path,
    output_root: str | Path,
    *,
    arm: str,
    seed: int,
    source_checkpoint: str | Path,
) -> Path | None:
    input_root = Path(input_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    expected = expected_run_contract(arm, seed, source_checkpoint)
    matches = []
    for path in input_root.rglob("run_contract.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload == expected:
            matches.append(path.parent)
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Saved Version {arm}/{seed} ambigu: {matches}")
    destination = run_directory(output_root, arm, seed)
    if destination.exists():
        existing = destination / "run_contract.json"
        if not existing.is_file() or json.loads(existing.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"Destination memiliki kontrak berbeda: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(matches[0], destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/validate top-controls Kaggle bundle")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--project-root", required=True)
    build.add_argument("--output-root", required=True)
    build_core = sub.add_parser("build-core")
    build_core.add_argument("--project-root", required=True)
    build_core.add_argument("--output-root", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--input-root", required=True)
    validate.add_argument("--work-root", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_top_controls_kaggle_bundle(args.project_root, args.output_root)
    elif args.command == "build-core":
        result = build_top_controls_canonical_kaggle_core(
            args.project_root, args.output_root
        )
    else:
        _, _, result = prepare_top_controls_kaggle_input(args.input_root, args.work_root)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
