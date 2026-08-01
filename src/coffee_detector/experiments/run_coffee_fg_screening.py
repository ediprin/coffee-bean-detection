from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.hf_sync import HuggingFaceSync
from coffee_detector.run_baseline import is_training_complete, load_verified_audit
from coffee_detector.train import load_experiment, train_experiment


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIGS = {
    "D0Q": REPO_ROOT / "configs/coffee_fg/D0Q_yolo26n_p3_quick10.yaml",
    "D1Q": REPO_ROOT / "configs/coffee_fg/D1Q_yolo26n_p2_quick10.yaml",
    "R0Q": REPO_ROOT / "configs/coffee_fg/R0Q_yolo26n_p3_first_order_quick10.yaml",
    "R1Q": REPO_ROOT / "configs/coffee_fg/R1Q_yolo26n_p3_bilinear_quick10.yaml",
    "D0": REPO_ROOT / "configs/coffee_fg/D0_yolo26n_p3.yaml",
    "D1": REPO_ROOT / "configs/coffee_fg/D1_yolo26n_p2.yaml",
    "R0": REPO_ROOT / "configs/coffee_fg/R0_yolo26n_p3_first_order.yaml",
    "R1": REPO_ROOT / "configs/coffee_fg/R1_yolo26n_p3_bilinear.yaml",
    "R2": REPO_ROOT / "configs/coffee_fg/R2_yolo26n_p2_first_order.yaml",
    "R3": REPO_ROOT / "configs/coffee_fg/R3_yolo26n_p2_bilinear.yaml",
}

COMPARISONS = (
    ("D0Q", "D1Q", "quick-10 efek P2 tanpa refiner"),
    ("D0Q", "R0Q", "quick-10 efek first-order ROI pada P3-P5"),
    ("D0Q", "R1Q", "quick-10 efek bilinear ROI pada P3-P5"),
    ("R0Q", "R1Q", "quick-10 bilinear vs capacity-matched first-order"),
    ("D0", "D1", "efek P2 tanpa refiner"),
    ("D0", "R0", "efek first-order ROI pada P3-P5"),
    ("D0", "R1", "efek bilinear ROI pada P3-P5"),
    ("R0", "R1", "bilinear vs capacity-matched first-order tanpa P2"),
    ("D1", "R2", "efek first-order ROI pada P2-P5"),
    ("D1", "R3", "efek bilinear ROI pada P2-P5"),
    ("R2", "R3", "bilinear vs capacity-matched first-order dengan P2"),
    ("R0", "R2", "efek P2 pada first-order refiner"),
    ("R1", "R3", "efek P2 pada bilinear refiner"),
)


def _aggregate(
    reports: dict[str, dict[int, dict]],
    baseline: str,
    candidate: str,
) -> dict:
    seeds = sorted(set(reports[baseline]) & set(reports[candidate]))
    if not seeds:
        raise RuntimeError(f"Tidak ada seed bersama untuk {baseline} vs {candidate}")
    result = {"baseline": baseline, "candidate": candidate, "seeds": seeds, "metrics": {}}
    for metric in (
        "macro_map50_95",
        "bottom3_class_map50_95",
        "worst_class_map50_95",
    ):
        left = np.asarray([reports[baseline][seed][metric] for seed in seeds], dtype=float)
        right = np.asarray([reports[candidate][seed][metric] for seed in seeds], dtype=float)
        delta = right - left
        result["metrics"][metric] = {
            "baseline_mean": float(left.mean()),
            "candidate_mean": float(right.mean()),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            "deltas": {str(seed): float(value) for seed, value in zip(seeds, delta)},
            "improved_seeds": int((delta > 0).sum()),
        }
    macro = result["metrics"]["macro_map50_95"]["delta_mean"]
    bottom = result["metrics"]["bottom3_class_map50_95"]["delta_mean"]
    worst = result["metrics"]["worst_class_map50_95"]["delta_mean"]
    criteria = {
        "macro_improved": macro > 0,
        "bottom3_preserved": bottom >= 0,
        "worst_preserved": worst >= 0,
    }
    result["decision"] = "PASS" if all(criteria.values()) else "FAIL"
    result["criteria"] = criteria
    return result


def run_coffee_fg_screening(
    data_root: str | Path,
    output_root: str | Path,
    *,
    models: tuple[str, ...] = ("D0", "D1"),
    seeds: tuple[int, ...] = (42,),
    evaluation_split: str = "val",
    open_test: bool = False,
    device: str | None = None,
    resume: bool = True,
    verified_audit: str | Path | None = None,
    diagnostic_report: str | Path | None = None,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "coffee-fg-v2",
    hf_private: bool = True,
) -> dict:
    unknown = sorted(set(models) - set(DEFAULT_CONFIGS))
    if unknown:
        raise ValueError("Model CoffeeFG tidak dikenal: " + ", ".join(unknown))
    if not seeds:
        raise ValueError("Minimal satu seed diperlukan")
    if evaluation_split == "test" and not open_test:
        raise RuntimeError(
            "Test terkunci. Gunakan validation untuk screening; --open-test hanya setelah protokol membuka test."
        )
    refinement_models = sorted(
        set(models) & {"R0Q", "R1Q", "R0", "R1", "R2", "R3"}
    )
    diagnostic = None
    if refinement_models:
        if diagnostic_report is None:
            raise RuntimeError(
                "Training refiner dikunci sampai diagnostic proposal/headroom selesai. "
                "Berikan --diagnostic-report."
            )
        diagnostic_path = Path(diagnostic_report).expanduser().resolve()
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        decision = diagnostic.get("decision", {})
        if not decision.get("classification_refinement_rational", False):
            raise RuntimeError("Diagnostic menyatakan classification refiner tidak rasional")
        allowed = set(decision.get("recommended_refiners", []))
        refiner_family = {
            "R0Q": "R0",
            "R1Q": "R1",
            "R0": "R0",
            "R1": "R1",
            "R2": "R2",
            "R3": "R3",
        }
        outside = sorted(
            code for code in refinement_models if refiner_family[code] not in allowed
        )
        if outside:
            raise RuntimeError(
                "Model refiner tidak sesuai foundation hasil diagnostic: "
                + ", ".join(outside)
            )

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    reports_root = output_root / (
        "reports" if evaluation_split == "test" else "val_reports"
    )
    reports_root.mkdir(parents=True, exist_ok=True)

    if verified_audit:
        audit_path = Path(verified_audit).expanduser().resolve()
        load_verified_audit(audit_path, data_root)
    else:
        audit_path = reports_root / "dataset_audit.json"
        audit = audit_dataset(data_root, audit_path, near_threshold=-1)
        if not audit["safe_for_training"]:
            raise RuntimeError(f"Dataset belum aman untuk training: {audit_path}")

    hub = (
        HuggingFaceSync(
            hf_repo_id,
            path_prefix=hf_path_prefix,
            private=hf_private,
        )
        if hf_repo_id
        else None
    )
    collected: dict[str, dict[int, dict]] = defaultdict(dict)
    for code in models:
        config_path = Path(DEFAULT_CONFIGS[code]).resolve()
        config = load_experiment(config_path)
        if config["code"] != code:
            raise ValueError(f"Code config tidak cocok: {config_path}")
        for seed in seeds:
            run_dir = output_root / f"{code}_seed{seed}"
            if not is_training_complete(run_dir):
                action = "RESUME" if (run_dir / "weights" / "last.pt").is_file() else "START"
                print(f"\n{action} TRAINING: {code} | seed {seed}", flush=True)
                train_experiment(
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
                print(f"SKIP TRAINING: {code} seed {seed} lengkap", flush=True)

            checkpoint = run_dir / "weights" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
            report_path = reports_root / f"{code}_seed{seed}_{evaluation_split}.json"
            payload = evaluate(
                checkpoint,
                data_root,
                report_path,
                split=evaluation_split,
                device=device,
            )
            missing_classes = payload["metrics"].get(
                "classes_without_ground_truth", []
            )
            if missing_classes:
                raise RuntimeError(
                    f"Split {evaluation_split} tidak mencakup seluruh kelas: "
                    + ", ".join(missing_classes)
                )
            collected[code][seed] = payload["metrics"]
            if hub is not None:
                hub.sync_run(run_dir)

    comparisons = {}
    for baseline, candidate, label in COMPARISONS:
        if baseline not in collected or candidate not in collected:
            continue
        key = f"{baseline}_vs_{candidate}"
        comparisons[key] = _aggregate(collected, baseline, candidate)
        comparisons[key]["label"] = label

    mechanism_decisions = {
        name: comparisons[name]["decision"]
        for name in ("R0Q_vs_R1Q", "R0_vs_R1", "R2_vs_R3")
        if name in comparisons
    }
    final_decision = (
        "PASS"
        if mechanism_decisions
        and any(value == "PASS" for value in mechanism_decisions.values())
        else ("FAIL" if mechanism_decisions else "DIAGNOSTIC_REQUIRED")
    )
    summary = {
        "protocol": "coffee-fg-v2",
        "data_root": str(data_root),
        "audit": str(audit_path),
        "models": list(models),
        "seeds": list(seeds),
        "evaluation_split": evaluation_split,
        "test_opened": evaluation_split == "test",
        "diagnostic_report": (
            str(Path(diagnostic_report).expanduser().resolve())
            if diagnostic_report is not None
            else None
        ),
        "diagnostic_decision": diagnostic.get("decision") if diagnostic else None,
        "runs": {
            code: {str(seed): metrics for seed, metrics in values.items()}
            for code, values in collected.items()
        },
        "comparisons": comparisons,
        "mechanism_decisions": mechanism_decisions,
        "final_decision": final_decision,
    }
    summary_path = reports_root / "coffee_fg_decision.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if hub is not None:
        hub.sync_output(output_root, f"{evaluation_split} decision {final_decision}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-first CoffeeFG-YOLO26 screening with a locked test split."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(DEFAULT_CONFIGS),
        default=["D0", "D1"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verified-audit")
    parser.add_argument("--diagnostic-report")
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-path-prefix", default="coffee-fg-v2")
    parser.add_argument("--hf-public", action="store_true")
    args = parser.parse_args()
    result = run_coffee_fg_screening(
        args.data_root,
        args.output_root,
        models=tuple(args.models),
        seeds=tuple(args.seeds),
        evaluation_split=args.evaluation_split,
        open_test=args.open_test,
        device=args.device,
        resume=not args.no_resume,
        verified_audit=args.verified_audit,
        diagnostic_report=args.diagnostic_report,
        hf_repo_id=args.hf_repo_id,
        hf_path_prefix=args.hf_path_prefix,
        hf_private=not args.hf_public,
    )
    print("\n=== PUTUSAN COFFEEFG ===")
    for name, comparison in result["comparisons"].items():
        print(name, comparison["decision"], comparison["criteria"])
    print("FINAL:", result["final_decision"])
    print("Test dibuka:", result["test_opened"])


if __name__ == "__main__":
    main()
