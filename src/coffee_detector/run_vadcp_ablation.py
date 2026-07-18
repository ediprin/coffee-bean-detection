from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .audit_dataset import audit_dataset
from .audit_vadcp import audit_vadcp_dataset
from .dataset import IMAGE_SUFFIXES, discover_layout
from .evaluate import evaluate
from .hf_sync import HuggingFaceSync
from .run_baseline import is_training_complete, load_verified_audit
from .run_visual_audit import run_visual_audit
from .train import load_experiment, train_experiment


DEFAULT_CONFIGS = {
    "A0": "configs/A0_yolo26n_real.yaml",
    "A1": "configs/A1_yolo26n_naive_copy_paste.yaml",
    "A2": "configs/A2_yolo26n_vadcp.yaml",
}


def _parse_pairs(values: list[str], label: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} harus CODE=PATH: {value}")
        code, path = value.split("=", 1)
        code = code.strip()
        if not code or not path.strip():
            raise ValueError(f"{label} tidak valid: {value}")
        if code in result:
            raise ValueError(f"{label} duplikat: {code}")
        result[code] = path.strip()
    return result


def _metric(metrics: dict, needle: str) -> float | None:
    wanted = needle.lower().replace(" ", "")
    for key, value in metrics.items():
        normalized = key.lower().replace(" ", "")
        if wanted == "map50" and "map50-95" in normalized:
            continue
        if wanted in normalized and isinstance(value, (int, float)):
            return float(value)
    return None


def _audit_arm(code: str, data_root: Path, reports: Path) -> Path:
    output = reports / f"{code}_dataset_audit.json"
    if (data_root / "metadata" / "instances_synthetic_train.json").is_file():
        report = audit_vadcp_dataset(data_root, output)
    else:
        report = audit_dataset(data_root, output, near_threshold=-1)
    if not report["safe_for_training"]:
        raise RuntimeError(f"Dataset arm {code} belum aman: {output}")
    return output


def _count_images(root: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in root.rglob("*")
    )


def build_combined_training_view(
    code: str,
    real_root: str | Path,
    synthetic_root: str | Path,
    output_root: str | Path,
) -> Path:
    """Create a no-copy Ultralytics view for real + synthetic train data."""
    if code == "A0":
        raise ValueError("A0 tidak memerlukan combined training view")
    real = discover_layout(real_root)
    synthetic = discover_layout(synthetic_root)
    if real.names != synthetic.names:
        raise ValueError(f"Nama/urutan kelas {code} berbeda dari A0")
    missing_real = sorted({"train", "val", "test"} - set(real.splits))
    if missing_real:
        raise FileNotFoundError(
            "A0 harus memiliki train/val/test: " + ", ".join(missing_real)
        )
    metadata_path = (
        synthetic.root / "metadata" / "generation_manifest.json"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Manifest synthetic {code} tidak ditemukan: {metadata_path}")
    generation = json.loads(metadata_path.read_text(encoding="utf-8"))
    if generation.get("include_real_train"):
        raise RuntimeError(
            f"Synthetic arm {code} sudah memuat real train; kombinasi akan menduplikasi A0"
        )

    output_root = Path(output_root).expanduser().resolve()
    view_root = output_root / "dataset_views" / code
    for split in ("train", "val", "test"):
        (view_root / split / "images").mkdir(parents=True, exist_ok=True)
        (view_root / split / "labels").mkdir(parents=True, exist_ok=True)
    real_train_images = real.splits["train"][0]
    synthetic_train_images = synthetic.splits["train"][0]
    payload = {
        "path": str(view_root),
        "train": [str(real_train_images), str(synthetic_train_images)],
        "val": str(real.splits["val"][0]),
        "test": str(real.splits["test"][0]),
        "names": real.names,
    }
    (view_root / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    manifest = {
        "format": "coffee_detector.combined_training_view.v1",
        "code": code,
        "real_root": str(real.root),
        "synthetic_root": str(synthetic.root),
        "real_train_images": _count_images(real_train_images),
        "synthetic_train_images": _count_images(synthetic_train_images),
        "val_root": str(real.splits["val"][0]),
        "test_root": str(real.splits["test"][0]),
        "files_copied": 0,
    }
    (view_root / "combined_view_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"TRAIN VIEW {code}: {manifest['real_train_images']} real + "
        f"{manifest['synthetic_train_images']} synthetic | tanpa copy",
        flush=True,
    )
    return view_root


def run_vadcp_ablation(
    arms: dict[str, str | Path],
    output_root: str | Path,
    *,
    configs: dict[str, str | Path] | None = None,
    seeds: tuple[int, ...] = (42,),
    device: str | None = None,
    resume: bool = True,
    count_audit: bool = True,
    count_confidence: float = 0.25,
    verified_audits: dict[str, str | Path] | None = None,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "vadcp-ablation-screen-v10",
    hf_private: bool = True,
) -> dict:
    if "A0" not in arms:
        raise ValueError("Arm A0 real-only wajib sebagai baseline")
    if not seeds:
        raise ValueError("Minimal satu seed diperlukan")
    configs = {**DEFAULT_CONFIGS, **(configs or {})}
    missing = sorted(set(arms) - set(configs))
    if missing:
        raise ValueError("Config belum diberikan untuk arm: " + ", ".join(missing))
    output_root = Path(output_root).expanduser().resolve()
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    resolved_arms = {
        code: Path(path).expanduser().resolve() for code, path in arms.items()
    }
    verified_audits = verified_audits or {}
    hub = (
        HuggingFaceSync(
            hf_repo_id,
            path_prefix=hf_path_prefix,
            private=hf_private,
        )
        if hf_repo_id
        else None
    )
    audit_paths = {}
    for code, data_root in resolved_arms.items():
        if code in verified_audits:
            path = Path(verified_audits[code]).expanduser().resolve()
            load_verified_audit(path, data_root)
            print(f"REUSE AUDIT {code}: {path}", flush=True)
        else:
            path = _audit_arm(code, data_root, reports)
        audit_paths[code] = str(path)

    real_test_root = resolved_arms["A0"]
    training_roots = {"A0": real_test_root}
    for code in sorted(set(resolved_arms) - {"A0"}):
        training_roots[code] = build_combined_training_view(
            code, real_test_root, resolved_arms[code], output_root
        )
    runs: dict[str, dict[str, dict]] = defaultdict(dict)

    for seed in seeds:
        for code, data_root in training_roots.items():
            config_path = Path(configs[code]).expanduser().resolve()
            config = load_experiment(config_path)
            if str(config["code"]) != code:
                raise ValueError(
                    f"Code config {config_path} adalah {config['code']}, bukan {code}"
                )
            run_dir = output_root / f"{code}_seed{seed}"
            if not is_training_complete(run_dir):
                action = "RESUME" if (run_dir / "weights" / "last.pt").is_file() else "START"
                print(f"\n{action} TRAINING: {code} | seed {seed}", flush=True)
                run_dir = train_experiment(
                    config_path,
                    data_root,
                    output_root,
                    seed,
                    device=device,
                    resume=resume,
                    on_checkpoint=(
                        (lambda path, epoch: hub.sync_run(path, epoch))
                        if hub is not None
                        else None
                    ),
                )
            else:
                print(f"SKIP TRAINING: {code} seed {seed} sudah lengkap", flush=True)
            checkpoint = run_dir / "weights" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"best.pt tidak ditemukan: {checkpoint}")
            evaluation_path = reports / f"{code}_seed{seed}_test.json"
            evaluation = evaluate(
                checkpoint,
                real_test_root,
                evaluation_path,
                split="test",
                device=device,
            )
            count_payload = None
            if count_audit:
                count_payload = run_visual_audit(
                    checkpoint,
                    real_test_root,
                    output_root / "count_audit" / f"{code}_seed{seed}",
                    samples=0,
                    seed=seed,
                    device=device,
                    confidence=count_confidence,
                )
            runs[code][str(seed)] = {
                "data_root": str(data_root),
                "evaluation_data_root": str(real_test_root),
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "evaluation": str(evaluation_path),
                "metrics": evaluation["metrics"],
                "count_metrics": (
                    {
                        key: count_payload[key]
                        for key in (
                            "exact_count_match_rate",
                            "mean_absolute_count_error",
                            "mean_count_bias",
                            "over_count_rate",
                            "under_count_rate",
                        )
                        if key in count_payload
                    }
                    if count_payload
                    else None
                ),
            }
            if hub is not None:
                hub.sync_output(output_root, f"{code} seed {seed} evaluated")

    aggregate = {}
    tracked = (
        "mAP50-95",
        "mAP50",
        "precision",
        "recall",
        "worst_class_map50_95",
    )
    for code, seed_rows in runs.items():
        aggregate[code] = {}
        for metric_name in tracked:
            values = [
                value
                for row in seed_rows.values()
                if (value := _metric(row["metrics"], metric_name)) is not None
            ]
            if values:
                aggregate[code][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }
        for count_name in (
            "exact_count_match_rate",
            "mean_absolute_count_error",
            "mean_count_bias",
            "over_count_rate",
            "under_count_rate",
        ):
            values = [
                float(row["count_metrics"][count_name])
                for row in seed_rows.values()
                if row["count_metrics"] and count_name in row["count_metrics"]
            ]
            if values:
                aggregate[code][count_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }

    comparisons = {}
    baseline = aggregate["A0"]
    for code in sorted(set(aggregate) - {"A0"}):
        comparisons[f"{code}_vs_A0"] = {}
        for metric_name in set(baseline) & set(aggregate[code]):
            comparisons[f"{code}_vs_A0"][metric_name] = (
                aggregate[code][metric_name]["mean"] - baseline[metric_name]["mean"]
            )
    payload = {
        "format": "coffee_detector.vadcp_ablation.v1",
        "arms": {code: str(path) for code, path in resolved_arms.items()},
        "training_roots": {
            code: str(path) for code, path in training_roots.items()
        },
        "audits": audit_paths,
        "seeds": list(seeds),
        "runs": dict(runs),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "huggingface": (
            {
                "repo_id": hub.repo_id,
                "repo_type": hub.repo_type,
                "path_prefix": hub.path_prefix,
            }
            if hub is not None
            else None
        ),
    }
    summary_path = reports / "vadcp_ablation_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(summary_path)
    if hub is not None:
        hub.sync_output(output_root, "ablation summary complete")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation real-only vs naive copy-paste vs VA-DCP."
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Ulangi sebagai CODE=/path/dataset, misalnya A0=/data/real.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Override config sebagai CODE=/path/config.yaml.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-count-audit", action="store_true")
    parser.add_argument("--count-confidence", type=float, default=0.25)
    parser.add_argument("--hf-repo-id")
    parser.add_argument(
        "--hf-path-prefix", default="vadcp-ablation-screen-v10"
    )
    parser.add_argument(
        "--hf-public", action="store_true", help="Buat repo Hub publik; default privat."
    )
    parser.add_argument(
        "--verified-audit",
        action="append",
        default=[],
        help="Audit yang sudah ada sebagai CODE=/path/audit.json.",
    )
    args = parser.parse_args()
    result = run_vadcp_ablation(
        _parse_pairs(args.arm, "--arm"),
        args.output_root,
        configs=_parse_pairs(args.config, "--config") if args.config else None,
        seeds=tuple(args.seeds),
        device=args.device,
        resume=not args.no_resume,
        count_audit=not args.skip_count_audit,
        count_confidence=args.count_confidence,
        verified_audits=(
            _parse_pairs(args.verified_audit, "--verified-audit")
            if args.verified_audit
            else None
        ),
        hf_repo_id=args.hf_repo_id,
        hf_path_prefix=args.hf_path_prefix,
        hf_private=not args.hf_public,
    )
    print("\n=== VA-DCP ABLATION ===")
    for code, metrics in result["aggregate"].items():
        print(code, json.dumps(metrics, ensure_ascii=False))
    print("DELTA", json.dumps(result["comparisons"], indent=2, ensure_ascii=False))
    print(f"SAVED: {result['summary']}")


if __name__ == "__main__":
    main()
