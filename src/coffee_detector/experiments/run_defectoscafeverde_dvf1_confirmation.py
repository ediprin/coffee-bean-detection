"""Run fixed DVF1 fuser confirmation seeds from immutable cached evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    CONFIG_PATH,
    FORMAT as DVF1_FORMAT,
    _config,
    _device,
    _json,
    _sha256,
    _train_fuser,
    evaluate_fuser,
    run_static_audit,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf2 import (
    FORMAT as DVF2_FORMAT,
    REVIEW_FORMAT,
)


PROTOCOL = "defectoscafeverde-dvf1-fuser-confirmation-v1"
FORMAT = "coffee_detector.defectoscafeverde.dvf1_confirmation.arm_result.v1"
ALLOWED_SEEDS = (123, 2026)
METRICS = (
    "physical_pair_accuracy",
    "macro_class_accuracy",
    "bottom3_class_accuracy",
    "worst_class_accuracy",
)


def _validate_inputs(
    result_path: Path, review_path: Path, dvf2_path: Path
) -> tuple[dict, Path, Path]:
    result = _json(result_path, "DVF1 seed-42 result")
    review = _json(review_path, "DVF1 review")
    dvf2 = _json(dvf2_path, "DVF2 result")
    if (
        result.get("format") != DVF1_FORMAT
        or result.get("seed") != 42
        or result.get("decision") != "PASS"
        or result.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF1 seed-42 result tidak valid")
    if (
        review.get("format") != REVIEW_FORMAT
        or review.get("decision") != "AUTHORIZE_DVF1_FUSER_MULTISEED"
        or review.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF1 review tidak mengotorisasi confirmation")
    if (
        dvf2.get("format") != DVF2_FORMAT
        or dvf2.get("decision") != "FAIL"
        or dvf2.get("next") != "RETAIN_DVF1"
        or dvf2.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DVF2 belum selesai sebagai FAIL yang valid")
    if _sha256(result_path) != review["artifacts"]["result_sha256"]:
        raise RuntimeError("SHA DVF1 result berbeda dari review")
    if dvf2["run_contract"]["dvf1_result_sha256"] != _sha256(result_path):
        raise RuntimeError("DVF2 tidak memakai DVF1 result ini")
    if dvf2["run_contract"]["dvf1_review_sha256"] != _sha256(review_path):
        raise RuntimeError("DVF2 tidak memakai DVF1 review ini")
    if result["run_contract"]["config_sha256"] != _sha256(CONFIG_PATH):
        raise RuntimeError("Konfigurasi DVF1 historis berubah")
    root = result_path.parent
    train_cache, val_cache = root / "train_pair_cache.pt", root / "val_pair_cache.pt"
    for path in (train_cache, val_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(val_cache) != review["artifacts"]["validation_cache_sha256"]:
        raise RuntimeError("Validation cache berbeda dari review")
    return result, train_cache, val_cache


def run_confirmation_seed(
    dvf1_result: str | Path,
    dvf1_review: str | Path,
    dvf2_result: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "cpu",
    authorize_training: bool = False,
) -> dict:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Seed confirmation harus salah satu {ALLOWED_SEEDS}")
    if not authorize_training:
        raise PermissionError("Confirmation memerlukan --authorize-training")
    result_path = Path(dvf1_result).expanduser().resolve()
    review_path = Path(dvf1_review).expanduser().resolve()
    dvf2_path = Path(dvf2_result).expanduser().resolve()
    seed42, train_path, val_path = _validate_inputs(
        result_path, review_path, dvf2_path
    )
    payload, fuser_config = _config()
    train_config = {**payload["train"], "seed": seed}
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"DVF1_seed{seed}_result.json"
    contract = {
        "protocol": PROTOCOL,
        "seed": seed,
        "dvf1_config_sha256": _sha256(CONFIG_PATH),
        "dvf1_seed42_result_sha256": _sha256(result_path),
        "dvf1_review_sha256": _sha256(review_path),
        "dvf2_result_sha256": _sha256(dvf2_path),
        "train_cache_sha256": _sha256(train_path),
        "validation_cache_sha256": _sha256(val_path),
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    if destination.is_file():
        result = _json(destination, "Cached confirmation result")
        if result.get("format") != FORMAT or result.get("run_contract") != contract:
            raise RuntimeError("Cached confirmation result stale")
        result["training_executed_this_call"] = False
        print(f"REUSE COMPLETE DVF1 SEED {seed}: {destination}", flush=True)
        return result
    static = run_static_audit(output_root / f"static_audit_seed{seed}.json")
    if static["decision"] != "PASS":
        raise RuntimeError(f"Static audit DVF1 gagal: {static['gates']}")
    torch_device = _device(device)
    train_cache = torch.load(train_path, map_location="cpu", weights_only=False)
    val_cache = torch.load(val_path, map_location="cpu", weights_only=False)
    if train_cache.get("split") != "train" or val_cache.get("split") != "val":
        raise RuntimeError("Split cache tidak valid")
    if set(train_cache["group_ids"]) & set(val_cache["group_ids"]):
        raise RuntimeError("Train/validation physical groups tumpang tindih")
    train_sha_before, val_sha_before = _sha256(train_path), _sha256(val_path)
    checkpoint_path = output_root / f"DVF1_seed{seed}_last.pt"
    fuser, history, trained = _train_fuser(
        train_cache,
        fuser_config,
        train_config,
        checkpoint_path,
        contract,
        torch_device,
    )
    values = evaluate_fuser(fuser, val_cache, torch_device)
    values["DVF1_CANDIDATE"] = values.pop("DVF1")
    calibration = {
        endpoint: {
            metric: values[endpoint][metric] - seed42["values"][endpoint][metric]
            for metric in METRICS
        }
        for endpoint in ("PAPER_MAX_CONFIDENCE", "PAIR_ORACLE")
    }
    gates = {
        "static_audit_pass": static["decision"] == "PASS",
        "paper_and_oracle_endpoints_exact": all(
            value == 0.0
            for endpoint in calibration.values()
            for value in endpoint.values()
        ),
        "train_validation_groups_disjoint": not (
            set(train_cache["group_ids"]) & set(val_cache["group_ids"])
        ),
        "train_cache_unchanged": _sha256(train_path) == train_sha_before,
        "validation_cache_unchanged": _sha256(val_path) == val_sha_before,
        "all_12_validation_classes_present": all(
            not row["classes_without_ground_truth"] for row in values.values()
            if isinstance(row, dict)
        ),
        "detector_not_loaded_or_trained": True,
        "test_not_opened": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Confirmation runtime gate gagal: {gates}")
    result = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "seed": seed,
        "values": values,
        "endpoint_calibration": calibration,
        "gates": gates,
        "status": "COMPLETE",
        "history": history,
        "artifacts": {
            "fuser_checkpoint": str(checkpoint_path),
            "fuser_checkpoint_sha256": _sha256(checkpoint_path),
            "static_audit": str(output_root / f"static_audit_seed{seed}.json"),
        },
        "run_contract": contract,
        "detector_training_executed": False,
        "fuser_training_executed": True,
        "training_executed_this_call": trained,
        "test_images_accessed": False,
    }
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DVF1 fixed fuser confirmation seed")
    parser.add_argument("--dvf1-result", required=True)
    parser.add_argument("--dvf1-review", required=True)
    parser.add_argument("--dvf2-result", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=ALLOWED_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_confirmation_seed(
        args.dvf1_result,
        args.dvf1_review,
        args.dvf2_result,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
