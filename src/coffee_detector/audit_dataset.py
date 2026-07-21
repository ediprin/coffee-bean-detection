from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import build_audit, discover_layout, write_json


def audit_dataset(data_root: str | Path, output: str | Path, near_threshold: int = 4) -> dict:
    audit = build_audit(
        discover_layout(data_root), near_threshold=near_threshold, progress=True
    )
    write_json(audit, output)
    return audit


def _print_summary(audit: dict) -> None:
    print("=== AUDIT DATASET YOLO ===")
    print(f"Root                  : {audit['dataset_root']}")
    print(f"Kelas                 : {len(audit['classes'])}")
    print(f"Gambar per split      : {audit['images_by_split']}")
    print(f"Box per split         : {audit['boxes_by_split']}")
    print(f"Gambar kosong         : {audit['empty_images_by_split']}")
    print(f"Komponen duplikat     : {audit['duplicate_components']}")
    print(f"Duplikat lintas split : {audit['cross_split_duplicate_components']}")
    print(f"Konflik anotasi       : {len(audit['exact_image_annotation_conflicts'])}")
    print(f"Error                 : {len(audit['errors'])}")
    print(f"AMAN TRAINING         : {'YA' if audit['safe_for_training'] else 'BELUM'}")
    print("\nBOX PER KELAS")
    for name, count in audit["boxes_by_class"].items():
        print(f"{name:24s}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset YOLO tanpa mengubah data sumber.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--near-threshold", type=int, default=4)
    args = parser.parse_args()
    audit = audit_dataset(args.data_root, args.output, args.near_threshold)
    _print_summary(audit)
    print(f"\nSAVED: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
