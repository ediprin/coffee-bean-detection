"""Post-hoc external-domain evaluation of frozen Faruq-v3 D0FT/ACMC1 pairs.

The evaluator materializes only Adrian validation images from the already
audited combined A0 archive.  Checkpoint hashes must match the completed
Faruq-v3 locked-test summary, so this module cannot be used to tune or replace
the frozen models after test access.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import statistics
from collections import Counter
from pathlib import Path

import yaml

from coffee_detector.dataset import IMAGE_SUFFIXES, parse_label
from coffee_detector.evaluate import _classwise_summary
from coffee_detector.prepare_sni_fullscene import (
    SNI21_CLASSES,
    canonical_source_identity,
)


FROZEN_SEEDS = (42, 123, 2026)
ARMS = ("D0FT", "ACMC1")
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
ADRIAN_PREFIX = "adrian_detection__"


def _load_json(path: str | Path, label: str) -> dict | list:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _roboflow_derivative_id(name: str) -> str | None:
    stem = Path(name).stem.lower()
    marker = ".rf."
    if marker not in stem:
        return None
    value = stem.rsplit(marker, 1)[1]
    return value or None


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _validate_safe_validation_restore(combined_root: Path) -> dict:
    candidates = (
        combined_root / "validation_restore.json",
        combined_root / "development_restore.json",
    )
    marker = next((path for path in candidates if path.is_file()), None)
    if marker is None:
        raise FileNotFoundError(
            "Marker restore development/validation A0 tidak ditemukan"
        )
    payload = _load_json(marker, "Marker restore A0")
    if payload.get("test_files_extracted") != 0 or payload.get(
        "test_images_accessed"
    ) is not False:
        raise RuntimeError("Restore A0 tidak menjaga test tetap terkunci")
    if (combined_root / "test").exists():
        raise RuntimeError("Split test A0 tidak boleh tersedia")
    return payload


def prepare_adrian_external_validation(
    combined_root: str | Path,
    faruq_manifest: str | Path,
    output_root: str | Path,
) -> dict:
    """Materialize Adrian validation only and audit cross-source identities."""

    combined_root = Path(combined_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    source_images = combined_root / "val/images"
    source_labels = combined_root / "val/labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("A0 validation belum direstore")
    _validate_safe_validation_restore(combined_root)

    faruq_rows = _load_json(faruq_manifest, "Manifest Faruq-v3")
    if not isinstance(faruq_rows, list) or not faruq_rows:
        raise RuntimeError("Manifest Faruq-v3 kosong/tidak dikenal")
    faruq_train = [
        row for row in faruq_rows if str(row.get("output_split")) == "train"
    ]
    if not faruq_train:
        raise RuntimeError("Manifest Faruq-v3 tidak memiliki train identities")
    faruq_parents = {str(row["source_parent_id"]) for row in faruq_train}
    faruq_source_hashes = {str(row["source_sha256"]) for row in faruq_train}
    faruq_derivatives = {
        derivative
        for row in faruq_train
        if (derivative := _roboflow_derivative_id(str(row["input_image"])))
    }

    summary_path = output_root / "adrian_external_validation_summary.json"
    manifest_path = output_root / "adrian_external_validation_manifest.json"
    if summary_path.is_file() and manifest_path.is_file():
        cached = _load_json(summary_path, "Cached Adrian preparation")
        if (
            cached.get("status") == "complete"
            and cached.get("test_images_accessed") is False
            and cached.get("source_dataset") == "adrian_detection"
            and cached.get("gates")
            and all(cached["gates"].values())
        ):
            print(f"REUSE ADRIAN VALIDATION: {output_root}", flush=True)
            return cached
        raise RuntimeError("Cache Adrian validation tidak aman/kompatibel")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output Adrian parsial: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    for kind in ("images", "labels"):
        (output_root / "train" / kind).mkdir(parents=True, exist_ok=True)
        (output_root / "val" / kind).mkdir(parents=True, exist_ok=True)

    valid_ids = set(range(len(SNI21_CLASSES)))
    class_counts: Counter[int] = Counter()
    manifest = []
    adrian_parents: set[str] = set()
    adrian_hashes: set[str] = set()
    adrian_derivatives: set[str] = set()
    all_validation_images = sorted(
        path
        for path in source_images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in all_validation_images:
        if not image_path.name.startswith(ADRIAN_PREFIX):
            continue
        relative = image_path.relative_to(source_images)
        label_path = (source_labels / relative).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Label Adrian hilang: {label_path}")
        boxes = parse_label(label_path, valid_ids)
        if not boxes:
            raise RuntimeError(f"Gambar Adrian tanpa target: {image_path}")
        clean_name = image_path.name[len(ADRIAN_PREFIX) :]
        parent_id = canonical_source_identity(clean_name)
        image_hash = _sha256(image_path)
        label_hash = _sha256(label_path)
        derivative_id = _roboflow_derivative_id(clean_name)
        adrian_parents.add(parent_id)
        adrian_hashes.add(image_hash)
        if derivative_id:
            adrian_derivatives.add(derivative_id)
        class_counts.update(box.class_id for box in boxes)
        image_target = output_root / "val/images" / relative
        label_target = (output_root / "val/labels" / relative).with_suffix(".txt")
        _link_or_copy(image_path, image_target)
        _link_or_copy(label_path, label_target)
        manifest.append(
            {
                "source_dataset": "adrian_detection",
                "source_split": "val",
                "file_name": relative.as_posix(),
                "source_parent_id": parent_id,
                "roboflow_derivative_id": derivative_id,
                "image_sha256": image_hash,
                "label_sha256": label_hash,
                "boxes": len(boxes),
            }
        )

    if not manifest:
        raise RuntimeError("Tidak ada identity Adrian pada A0 validation")
    parent_overlap = sorted(adrian_parents & faruq_parents)
    hash_overlap = sorted(adrian_hashes & faruq_source_hashes)
    derivative_overlap = sorted(adrian_derivatives & faruq_derivatives)
    supported_ids = sorted(class_counts)
    missing_ids = sorted(valid_ids - set(supported_ids))
    names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output_root),
                "train": "train/images",
                "val": "val/images",
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gates = {
        "source_tag_is_adrian_only": all(
            row["source_dataset"] == "adrian_detection" for row in manifest
        ),
        "zero_parent_overlap_with_faruq_train": not parent_overlap,
        "zero_derivative_id_overlap_with_faruq_train": not derivative_overlap,
        "zero_cross_manifest_hash_overlap_with_faruq_train": not hash_overlap,
        "test_not_materialized": not (output_root / "test").exists(),
    }
    payload = {
        "format": "coffee_detector.faruq_v3_acmc_adrian_external_setup.v1",
        "status": "complete" if all(gates.values()) else "failed",
        "source_dataset": "adrian_detection",
        "source_split": "val",
        "images": len(manifest),
        "boxes": sum(row["boxes"] for row in manifest),
        "independent_parent_ids": len(adrian_parents),
        "class_support": {
            SNI21_CLASSES[class_id]: int(class_counts[class_id])
            for class_id in supported_ids
        },
        "supported_classes": [SNI21_CLASSES[index] for index in supported_ids],
        "classes_without_ground_truth": [
            SNI21_CLASSES[index] for index in missing_ids
        ],
        "parent_overlap": parent_overlap,
        "derivative_id_overlap": derivative_overlap,
        "cross_manifest_hash_overlap": hash_overlap,
        "gates": gates,
        "manifest_sha256": _sha256(manifest_path),
        "test_images_accessed": False,
        "training_executed": False,
        "development_only": True,
        "claim_limit": (
            "Adrian validation is an independent-source, post-hoc development "
            "evaluation. Its small number of parent identities limits uncertainty "
            "claims and it does not replace the completed Faruq locked test."
        ),
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not all(gates.values()):
        raise RuntimeError(f"Audit external Adrian gagal: {gates}")
    return payload


def _runtime_yaml(data_root: Path, output: Path) -> Path:
    payload = {
        "path": str(data_root),
        # Ultralytics validates schema keys; train is an alias only and no
        # training API is called by this module.
        "train": "val/images",
        "val": "val/images",
        "names": {index: name for index, name in enumerate(SNI21_CLASSES)},
        "external_development_only": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output


def _evaluate_checkpoint(
    checkpoint: Path,
    checkpoint_hash: str,
    data_root: Path,
    dataset_manifest_hash: str,
    output: Path,
    *,
    device: str | None,
) -> dict:
    if output.is_file():
        cached = _load_json(output, "Cached Adrian report")
        if (
            cached.get("checkpoint_sha256") == checkpoint_hash
            and cached.get("dataset_manifest_sha256") == dataset_manifest_hash
            and cached.get("complete") is True
        ):
            print(f"REUSE {output.name}", flush=True)
            return cached
        raise RuntimeError(f"Cache Adrian tidak kompatibel: {output}")
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error
    runtime_yaml = _runtime_yaml(data_root, output.parent / "runtime_data.yaml")
    kwargs = {
        "data": str(runtime_yaml),
        "split": "val",
        "imgsz": 640,
        "batch": 16,
        "workers": 2,
        "max_det": 500,
        "conf": 0.001,
        "iou": 0.7,
        "plots": False,
        "verbose": True,
        "project": str(output.parent / "ultralytics"),
        "name": "validation",
        "exist_ok": True,
    }
    if device is not None:
        kwargs["device"] = device
    model = YOLO(str(checkpoint))
    metrics = model.val(**kwargs)
    box = getattr(metrics, "box", None)
    if box is None or getattr(box, "ap", None) is None:
        raise RuntimeError("Evaluator Adrian tidak menghasilkan box AP")
    results = {
        key: float(value) for key, value in metrics.results_dict.items()
    }
    results.update(
        _classwise_summary(
            box, {index: name for index, name in enumerate(SNI21_CLASSES)}
        )
    )
    payload = {
        "format": "coffee_detector.faruq_v3_acmc_adrian_external_report.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "split": "val",
        "metrics": results,
        "training_executed": False,
        "test_images_accessed": False,
        "complete": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model, metrics
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
    return payload


def _validate_frozen_checkpoints(
    locked_summary: dict,
    checkpoint_map: dict[str, tuple[Path, ...]],
) -> dict[str, tuple[str, ...]]:
    if (
        locked_summary.get("status") != "complete"
        or tuple(locked_summary.get("seeds", [])) != FROZEN_SEEDS
        or locked_summary.get("training_executed") is not False
        or locked_summary.get("test_opened") is not True
        or locked_summary.get("further_tuning_authorized") is not False
    ):
        raise RuntimeError("Locked-test summary tidak membekukan model yang sah")
    hashes = {
        arm: tuple(_sha256(path) for path in checkpoint_map[arm]) for arm in ARMS
    }
    expected = locked_summary.get("checkpoint_hashes", {})
    for arm in ARMS:
        if list(hashes[arm]) != list(expected.get(arm, [])):
            raise RuntimeError(f"Hash checkpoint {arm} berbeda dari locked test")
    return hashes


def run_faruq_v3_acmc_adrian_external(
    combined_root: str | Path,
    faruq_manifest: str | Path,
    locked_test_summary: str | Path,
    adrian_root: str | Path,
    output_root: str | Path,
    d0ft_checkpoints: tuple[str | Path, ...],
    acmc_checkpoints: tuple[str | Path, ...],
    *,
    seeds: tuple[int, ...] = FROZEN_SEEDS,
    device: str | None = "0",
) -> dict:
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != FROZEN_SEEDS:
        raise ValueError(f"External evaluation dikunci pada seed {FROZEN_SEEDS}")
    if len(d0ft_checkpoints) != 3 or len(acmc_checkpoints) != 3:
        raise ValueError("Diperlukan tiga pasangan checkpoint beku")
    checkpoint_map = {
        "D0FT": tuple(Path(path).expanduser().resolve() for path in d0ft_checkpoints),
        "ACMC1": tuple(Path(path).expanduser().resolve() for path in acmc_checkpoints),
    }
    for paths in checkpoint_map.values():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    locked = _load_json(locked_test_summary, "Faruq locked-test summary")
    checkpoint_hashes = _validate_frozen_checkpoints(locked, checkpoint_map)
    adrian_root = Path(adrian_root).expanduser().resolve()
    setup = prepare_adrian_external_validation(
        combined_root, faruq_manifest, adrian_root
    )
    output_root = Path(output_root).expanduser().resolve()
    manifest_hash = str(setup["manifest_sha256"])
    per_seed = {}
    for index, seed in enumerate(frozen_seeds):
        results = {}
        for arm in ARMS:
            print(f"ADRIAN EXTERNAL {arm} seed={seed}", flush=True)
            report = _evaluate_checkpoint(
                checkpoint_map[arm][index],
                checkpoint_hashes[arm][index],
                adrian_root,
                manifest_hash,
                output_root / "reports" / f"{arm}_seed{seed}_adrian_val.json",
                device=device,
            )
            if set(report["metrics"].get("classes_without_ground_truth", [])) != set(
                setup["classes_without_ground_truth"]
            ):
                raise RuntimeError("Coverage kelas evaluator berbeda dari setup Adrian")
            results[arm] = report["metrics"]
        per_seed[str(seed)] = {
            "results": results,
            "head_deltas_acmc1_vs_d0ft": {
                metric: float(results["ACMC1"][metric])
                - float(results["D0FT"][metric])
                for metric in METRICS
            },
        }

    aggregate = {}
    for metric in METRICS:
        left = [
            float(per_seed[str(seed)]["results"]["D0FT"][metric])
            for seed in frozen_seeds
        ]
        right = [
            float(per_seed[str(seed)]["results"]["ACMC1"][metric])
            for seed in frozen_seeds
        ]
        deltas = [candidate - baseline for baseline, candidate in zip(left, right)]
        aggregate[metric] = {
            "d0ft_mean": statistics.mean(left),
            "d0ft_std": statistics.stdev(left),
            "acmc1_mean": statistics.mean(right),
            "acmc1_std": statistics.stdev(right),
            "head_delta_mean": statistics.mean(deltas),
            "head_delta_std": statistics.stdev(deltas),
            "head_delta_min": min(deltas),
            "head_improved_seeds": sum(value > 0.0 for value in deltas),
            "per_seed_deltas": dict(zip(map(str, frozen_seeds), deltas)),
        }
    criteria = {
        "macro_delta_positive": aggregate["macro_map50_95"]["head_delta_mean"] > 0,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"][
            "head_improved_seeds"
        ]
        >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"][
            "head_delta_mean"
        ]
        >= 0,
        "worst_mean_drop_no_more_than_1_point": aggregate[
            "worst_class_map50_95"
        ]["head_delta_mean"]
        >= -0.01,
    }
    directional_status = (
        "SUPPORTS_EXTERNAL_DIRECTION"
        if all(criteria.values())
        else "DOES_NOT_SUPPORT_EXTERNAL_DIRECTION"
    )
    payload = {
        "format": "coffee_detector.faruq_v3_acmc_adrian_external_summary.v1",
        "status": "complete",
        "directional_status": directional_status,
        "seeds": list(frozen_seeds),
        "checkpoint_hashes": {
            arm: list(values) for arm, values in checkpoint_hashes.items()
        },
        "locked_test_summary_sha256": _sha256(locked_test_summary),
        "adrian_setup": setup,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "further_tuning_authorized": False,
        "locked_test_conclusion_changed": False,
        "claim_limit": (
            "Post-hoc external-source development evidence only. It cannot "
            "override the Faruq locked-test NOT_CONFIRMED conclusion."
        ),
        "next_action": "REPORT_EXTERNAL_DIRECTION_WITHOUT_MODEL_TUNING",
    }
    summary_path = output_root / "adrian_external_summary.json"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary_path"] = str(summary_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen three-seed ACMC evaluation on Adrian validation"
    )
    parser.add_argument("--combined-root", required=True)
    parser.add_argument("--faruq-manifest", required=True)
    parser.add_argument("--locked-test-summary", required=True)
    parser.add_argument("--adrian-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--d0ft-checkpoints", nargs=3, required=True)
    parser.add_argument("--acmc-checkpoints", nargs=3, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, default=list(FROZEN_SEEDS))
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run_faruq_v3_acmc_adrian_external(
        args.combined_root,
        args.faruq_manifest,
        args.locked_test_summary,
        args.adrian_root,
        args.output_root,
        tuple(args.d0ft_checkpoints),
        tuple(args.acmc_checkpoints),
        seeds=tuple(args.seeds),
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
