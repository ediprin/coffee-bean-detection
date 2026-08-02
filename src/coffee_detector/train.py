from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import yaml

from .dataset import discover_layout
from .coffee_fg import make_coffee_fg_trainer
from .hong_transfer import make_hong_transfer_trainer
from .models.local_hbp import make_local_hbp_trainer
from .multilevel_head import make_multilevel_head_trainer
from .ontology_marginal import make_ontology_marginal_trainer


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def _resolve_model_reference(value: str | Path, repo_root: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return str(path.resolve())
    candidate = repo_root / path
    return str(candidate.resolve()) if candidate.exists() else str(value)


def load_experiment(path: str | Path) -> dict:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"code", "variant", "model", "train"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Config {path} belum memiliki: {', '.join(missing)}")
    if payload["variant"] not in {
        "baseline",
        "local_hbp",
        "coffee_fg",
        "hong_transfer",
        "ontology_marginal",
        "multilevel_head",
    }:
        raise ValueError(
            "variant harus baseline, local_hbp, coffee_fg, hong_transfer, "
            "ontology_marginal, atau multilevel_head"
        )
    if payload["variant"] == "coffee_fg" and not isinstance(payload.get("coffee_fg"), dict):
        raise ValueError("variant coffee_fg memerlukan mapping coffee_fg")
    if payload["variant"] == "hong_transfer" and not isinstance(
        payload.get("hong_transfer"), dict
    ):
        raise ValueError("variant hong_transfer memerlukan mapping hong_transfer")
    if payload["variant"] == "ontology_marginal" and not isinstance(
        payload.get("ontology_marginal"), dict
    ):
        raise ValueError("variant ontology_marginal memerlukan mapping ontology_marginal")
    if payload["variant"] == "multilevel_head" and not isinstance(
        payload.get("multilevel_head"), dict
    ):
        raise ValueError("variant multilevel_head memerlukan mapping multilevel_head")
    return payload


def recover_completed_training_manifest(
    config_path: str | Path,
    data_root: str | Path,
    run_dir: str | Path,
    seed: int,
) -> bool:
    """Finalize a run whose last epoch survived but post-train metadata did not.

    Colab can disconnect after Ultralytics saves its final checkpoints and CSV
    but before this package writes ``experiment_manifest.json``.  Recovery is
    intentionally strict: both checkpoints must exist and the CSV must contain
    at least the configured number of epoch rows.  Partial runs still resume.
    """

    config_path = Path(config_path).resolve()
    config = load_experiment(config_path)
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "experiment_manifest.json"
    if manifest_path.is_file():
        return True

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results = run_dir / "results.csv"
    if not (best.is_file() and last.is_file() and results.is_file()):
        return False

    with results.open("r", encoding="utf-8-sig", newline="") as handle:
        completed_epochs = sum(1 for _ in csv.DictReader(handle))
    expected_epochs = int(config["train"].get("epochs", 0))
    if expected_epochs <= 0 or completed_epochs < expected_epochs:
        return False

    layout = discover_layout(data_root)
    manifest = {
        "config": str(config_path),
        "code": config["code"],
        "variant": config["variant"],
        "model": config["model"],
        "weights": config.get("weights"),
        "coffee_fg": config.get("coffee_fg"),
        "hong_transfer": config.get("hong_transfer"),
        "ontology_marginal": config.get("ontology_marginal"),
        "multilevel_head": config.get("multilevel_head"),
        "data": str(layout.root),
        "data_yaml": str(layout.yaml_path),
        "seed": int(seed),
        "train": dict(config["train"]),
        "completed_epochs": completed_epochs,
        "recovered_after_runtime_disconnect": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return True


def train_experiment(
    config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    seed: int,
    device: str | None = None,
    resume: bool = False,
    on_checkpoint: Callable[[Path, int], None] | None = None,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - runtime-only dependency
        raise RuntimeError("Ultralytics belum terpasang. Jalankan `pip install -e .`.") from error

    config_path = Path(config_path).resolve()
    config = load_experiment(config_path)
    repo_root = _repository_root(config_path.parent)
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
    model_reference = _resolve_model_reference(config["model"], repo_root)
    model = YOLO(str(last_checkpoint if resume and last_checkpoint.is_file() else model_reference))
    if (
        not (resume and last_checkpoint.is_file())
        and config.get("weights")
        and str(config["model"]).lower().endswith((".yaml", ".yml"))
    ):
        model.load(_resolve_model_reference(config["weights"], repo_root))
    if on_checkpoint is not None:
        def _persist_checkpoint(trainer) -> None:
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            on_checkpoint(Path(trainer.save_dir), epoch)

        model.add_callback("on_model_save", _persist_checkpoint)
    if resume and last_checkpoint.is_file():
        train_args = {"resume": True}
        if device is not None:
            train_args["device"] = device
    if config["variant"] == "local_hbp":
        rank = int(config.get("local_hbp", {}).get("rank", 64))
        trainer = make_local_hbp_trainer(rank)
        model.train(trainer=trainer, **train_args)
    elif config["variant"] == "coffee_fg":
        trainer = make_coffee_fg_trainer(config["coffee_fg"])
        model.train(trainer=trainer, **train_args)
    elif config["variant"] == "hong_transfer":
        trainer = make_hong_transfer_trainer(config["hong_transfer"])
        model.train(trainer=trainer, **train_args)
    elif config["variant"] == "ontology_marginal":
        trainer = make_ontology_marginal_trainer(config["ontology_marginal"])
        model.train(trainer=trainer, **train_args)
    elif config["variant"] == "multilevel_head":
        trainer = make_multilevel_head_trainer(config["multilevel_head"])
        model.train(trainer=trainer, **train_args)
    else:
        model.train(**train_args)

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(config_path),
        "code": config["code"],
        "variant": config["variant"],
        "model": config["model"],
        "weights": config.get("weights"),
        "coffee_fg": config.get("coffee_fg"),
        "hong_transfer": config.get("hong_transfer"),
        "ontology_marginal": config.get("ontology_marginal"),
        "multilevel_head": config.get("multilevel_head"),
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
