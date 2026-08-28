from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
import torch

from coffee_detector.af2_scaffold import (
    AF2ScaffoldConfig,
    make_af2_scaffold_trainer,
    strip_training_scaffold,
)
from coffee_detector.afab import AFABConfig
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import _exclusive_training_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_scaffold/AF2MTS1_yolo26n.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(run_dir: Path, epochs: int) -> bool:
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights/best.pt"
    if not marker.is_file() or not best.is_file():
        return False
    payload = _read(marker, "Training marker")
    return bool(payload.get("trainer_returned")) and payload.get("epochs_requested") == epochs


def _headline(metrics: dict) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in (
            "macro_map50_95",
            "bottom3_class_map50_95",
            "worst_class_map50_95",
        )
    }


def _flatten(value) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result: list[torch.Tensor] = []
        for key in sorted(value):
            result.extend(_flatten(value[key]))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return []


def _torch_device(value: str) -> torch.device:
    return torch.device(f"cuda:{value}" if str(value).isdigit() else value)


def _raw_export_exact(wrapped_path: Path, native_path: Path, device: str) -> bool:
    from ultralytics import YOLO

    resolved = _torch_device(device)
    wrapped = YOLO(str(wrapped_path)).model.to(resolved).eval()
    native = YOLO(str(native_path)).model.to(resolved).eval()
    torch.manual_seed(20260828)
    sample = torch.rand(1, 3, 64, 64, device=resolved)
    with torch.inference_mode():
        wrapped_output = _flatten(wrapped(sample.clone()))
        native_output = _flatten(native(sample.clone()))
    return len(wrapped_output) == len(native_output) and all(
        torch.equal(left, right) for left, right in zip(wrapped_output, native_output)
    )


def _export_native_checkpoint(source: Path, output: Path) -> Path:
    from ultralytics import YOLO

    wrapped = YOLO(str(source))
    strip_training_scaffold(wrapped.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapped.save(str(output))
    if not output.is_file():
        raise RuntimeError(f"Export native gagal: {output}")
    return output


def run_faruq_v3_af2_scaffold_arm(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("Kill-gate dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    grouped_summary = Path(grouped_summary).expanduser().resolve()
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    static_audit = Path(static_audit).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not (data_root / "data.yaml").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Dataset development atau checkpoint AF2 tidak lengkap")
    _read(grouped_summary, "Grouped summary")
    audit = _read(static_audit, "Static audit")
    if audit.get("format") != "coffee_detector.af2_scaffold.static_audit.v1":
        raise RuntimeError("Schema static audit salah")
    if audit.get("decision") != "PASS" or not audit.get("training_authorized"):
        raise RuntimeError("Static audit belum PASS")
    if audit.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("Checkpoint AF2 berbeda dari static audit")
    if audit.get("test_access_authorized") is not False:
        raise RuntimeError("Test lock tidak dipertahankan")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "AF2MTS1":
        raise RuntimeError("Config AF2MTS1 tidak konsisten")
    afab = AFABConfig.from_mapping(payload["afab"])
    scaffold = AF2ScaffoldConfig.from_mapping(payload["scaffold"])
    epochs = int(payload["train"]["epochs"])
    if epochs != scaffold.total_epochs:
        raise RuntimeError("Epoch config dan scaffold tidak cocok")
    run_dir = output_root / "AF2MTS1" / f"AF2MTS1_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    native = run_dir / "weights/best_native.pt"
    training_executed = False

    if not _complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_af2_scaffold_trainer(afab, scaffold, initial_checkpoint=checkpoint)
        with _exclusive_training_lock(output_root, lock_name=f"AF2MTS1_seed{seed}.training.lock"):
            if last.is_file():
                print("RESUME AF2MTS1 seed 42 dari last.pt", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                print("START AF2MTS1 seed 42 dari AF2 parent", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / "AF2MTS1"),
                    name=f"AF2MTS1_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                    device=device,
                )
            model.train(trainer=trainer, **args)
        training_executed = True
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "training_complete.json").write_text(
            json.dumps(
                {
                    "trainer_returned": True,
                    "epochs_requested": epochs,
                    "seed": seed,
                    "arm": "AF2MTS1",
                    "initial_checkpoint_sha256": _sha256(checkpoint),
                    "config_sha256": _sha256(CONFIG),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    if not _complete(run_dir, epochs):
        raise RuntimeError(f"Run belum lengkap: {run_dir}")

    wrapped_report_path = output_root / "val_reports/AF2MTS1_seed42_wrapped_bypass_val.json"
    wrapped_report = evaluate(best, data_root, wrapped_report_path, split="val", device=device)
    if wrapped_report["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    _export_native_checkpoint(best, native)
    raw_export_exact = _raw_export_exact(best, native, device)
    if not raw_export_exact:
        raise RuntimeError("Export native tidak bitwise-identik dengan bypass")
    native_report_path = output_root / "val_reports/AF2MTS1_seed42_native_val.json"
    native_report = evaluate(native, data_root, native_report_path, split="val", device=device)
    if native_report["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation export kehilangan kelas")
    wrapped_values = _headline(wrapped_report["metrics"])
    native_values = _headline(native_report["metrics"])
    export_deltas = {
        key: native_values[key] - wrapped_values[key] for key in native_values
    }
    if any(abs(value) > 1e-6 for value in export_deltas.values()):
        raise RuntimeError(f"Export native mengubah metrik: {export_deltas}")

    result = {
        "format": "coffee_detector.af2_scaffold.arm_result.v1",
        "arm": "AF2MTS1",
        "seed": seed,
        "metrics": native_report["metrics"],
        "wrapped_bypass_metrics": wrapped_report["metrics"],
        "native_export_deltas": export_deltas,
        "native_export_raw_output_bitwise_exact": raw_export_exact,
        "training_checkpoint": str(best),
        "native_checkpoint": str(native),
        "native_checkpoint_sha256": _sha256(native),
        "initial_af2_checkpoint": str(checkpoint),
        "initial_af2_checkpoint_sha256": _sha256(checkpoint),
        "config": str(CONFIG),
        "config_sha256": _sha256(CONFIG),
        "static_audit": str(static_audit),
        "training_executed_this_call": training_executed,
        "test_images_accessed": False,
    }
    result_path = output_root / "val_reports/AF2MTS1_seed42_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AF2MTS1 seed-42 kill gate arm")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_scaffold_arm(
        args.data_root,
        args.grouped_summary,
        args.af2_checkpoint,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
