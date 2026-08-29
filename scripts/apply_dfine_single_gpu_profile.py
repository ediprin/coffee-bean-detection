from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


PROFILE = {
    "train_dataloader": {
        "total_batch_size": 16,
        "num_workers": 2,
    },
    "val_dataloader": {
        "total_batch_size": 16,
        "num_workers": 2,
    },
    "optimizer": {
        "type": "AdamW",
        "params": [
            {
                "params": "^(?=.*backbone)(?!.*norm|bn).*$",
                "lr": 0.00005,
            },
            {
                "params": "^(?=.*backbone)(?=.*norm|bn).*$",
                "lr": 0.00005,
                "weight_decay": 0.0,
            },
            {
                "params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$",
                "weight_decay": 0.0,
            },
        ],
        "lr": 0.0001,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
    },
    "ema": {
        "type": "ModelEMA",
        "decay": 0.9999875,
        "warmups": 8000,
        "start": 0,
    },
    "lr_warmup_scheduler": {
        "type": "LinearWarmup",
        "warmup_duration": 4000,
    },
    "epochs": 220,
}


def merge_dict(target: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_dict(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def apply(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    payload = merge_dict(payload, PROFILE)
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply frozen single-GPU D-FINE-N profile")
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    if len(args.configs) != 2:
        raise SystemExit("Exactly two paired config paths are required")
    updated = [apply(path) for path in args.configs]
    print("Applied frozen single-GPU profile to:")
    for path in updated:
        print(path)


if __name__ == "__main__":
    main()
