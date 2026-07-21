from __future__ import annotations

import argparse
import json

from .dataset import discover_layout
from .vadcp.profile import (
    build_scene_calibration,
    calibration_summary,
    save_scene_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kalibrasi geometri, skala, dan background VA-DCP dari train nyata."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-samples", type=int, default=64)
    args = parser.parse_args()

    layout = discover_layout(args.data_root)
    calibration = build_scene_calibration(
        layout,
        split="train",
        seed=args.seed,
        background_samples=args.background_samples,
    )
    path = save_scene_calibration(calibration, args.output)
    print("=== REAL-SCENE CALIBRATION ===")
    print(json.dumps(calibration_summary(calibration), indent=2, ensure_ascii=False))
    print("SAVED:", path)


if __name__ == "__main__":
    main()
