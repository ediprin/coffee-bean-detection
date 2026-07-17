from __future__ import annotations

import argparse
from pathlib import Path

from .vadcp.library import prepare_classification_library, prepare_yolo_library


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat pustaka cutout RGBA untuk eksperimen VA-DCP."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--classification-root")
    source.add_argument("--data-root", help="Dataset YOLO; hanya split train yang diekstrak.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--allowed-splits",
        nargs="+",
        default=["train", "unspecified"],
        help="Hanya untuk --classification-root.",
    )
    parser.add_argument("--mask-threshold", type=float, default=24.0)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--max-assets-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.classification_root:
        result = prepare_classification_library(
            args.classification_root,
            args.output_root,
            allowed_splits=tuple(args.allowed_splits),
            mask_threshold=args.mask_threshold,
            padding=args.padding,
            max_assets_per_class=args.max_assets_per_class,
            seed=args.seed,
        )
    else:
        result = prepare_yolo_library(
            args.data_root,
            args.output_root,
            source_split=args.source_split,
            mask_threshold=args.mask_threshold,
            padding=args.padding,
            max_assets_per_class=args.max_assets_per_class,
            seed=args.seed,
        )
    audit = result["audit"]
    print("=== OBJECT LIBRARY ===")
    print(f"Output       : {Path(args.output_root).expanduser().resolve()}")
    print(f"Assets       : {audit['assets']}")
    print(f"Classes      : {audit['classes']}")
    print(f"By class     : {audit['assets_by_class']}")
    print(f"Duplicate rm : {audit['duplicate_assets_removed']}")
    print(f"Failures     : {audit['failures']}")
    print("Periksa object_library.json dan audit visual cutout sebelum generasi data.")


if __name__ == "__main__":
    main()
