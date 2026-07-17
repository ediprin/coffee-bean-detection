from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .dataset import discover_layout
from .models.local_hbp import make_local_hbp_trainer


def load_experiment(path: str | Path) -> dict:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"code", "variant", "model", "train"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Config {path} belum memiliki: {', '.join(missing)}")
    if payload["variant"] not in {"baseline", "local_hbp"}:
        raise ValueError("variant harus baseline atau local_hbp")
    return payload


def train_experiment(
    config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    seed: int,
    device: str | None = None,
    resume: bool = False,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError("Ultralytics belum terpasang. Jalankan `pip install -e .`.") from error

    config_path = Path(config_path).resolve()
    config = load_experiment(config_path)
    layout = discover_layout(data_root)
    output_root = Path(output_root).resolve()
    run_name = f"{config['code']}_seed{seed}"
    run_dir = output_root / run_name
    train_args = dict(config["train"])
    train_args.update(
        {
            "data": str(layout.yaml_path),
            "project": str(output_root),
            "name": run_name,
            "exist_ok": True,
            "seed": int(seed),
            "deterministic": True,
            "plots": True,
            "verbose": True,
        }
    )
    if device is not None:
        train_args["device"] = device

    last_checkpoint = run_dir / "weights" / "last.pt"
    if last_checkpoint.is_file() and not resume:
        raise FileExistsError(
            f"Run sudah memiliki checkpoint: {last_checkpoint}. Gunakan --resume atau output baru."
        )
    model = YOLO(str(last_checkpoint if resume and last_checkpoint.is_file() else config["model"]))
    if resume and last_checkpoint.is_file():
        train_args = {"resume": True}
        if device is not None:
            train_args["device"] = device
    if config["variant"] == "local_hbp":
        rank = int(config.get("local_hbp", {}).get("rank", 64))
        trainer = make_local_hbp_trainer(rank)
        model.train(trainer=trainer, **train_args)
    else:
        model.train(**train_args)

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(config_path),
        "code": config["code"],
        "variant": config["variant"],
        "model": config["model"],
        "data": str(layout.root),
        "data_yaml": str(layout.yaml_path),
        "seed": seed,
        "train": train_args,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline YOLO atau local-HBP secara terisolasi.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = train_experiment(
        args.config,
        args.data_root,
        args.output_root,
        args.seed,
        device=args.device,
        resume=args.resume,
    )
    print(f"SELESAI: {run_dir}")


if __name__ == "__main__":
    main()
