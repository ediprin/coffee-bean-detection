from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PatchScaleReport:
    label_files: int
    boxes: int
    imgsz: int
    q25_equivalent_side_px: float
    median_equivalent_side_px: float
    q75_equivalent_side_px: float
    candidates: tuple[int, ...]
    selected_patch_size: int
    selection_rule: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidates"] = list(self.candidates)
        return payload


def _iter_label_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".txt":
        yield root
        return
    if not root.exists():
        raise FileNotFoundError(root)
    yield from sorted(path for path in root.rglob("*.txt") if path.is_file())


def collect_equivalent_box_sides(label_root: str | Path, *, imgsz: int = 640) -> tuple[np.ndarray, int]:
    """Read YOLO normalized xywh labels and express bbox equivalent side at training resolution.

    For a normalized box width w and height h, the equivalent square side is
    imgsz * sqrt(w*h).  This avoids assuming that the original image files are square while
    matching the detector's fixed training canvas closely enough for scale calibration.
    """

    root = Path(label_root).expanduser().resolve()
    values: list[float] = []
    files = 0
    for path in _iter_label_files(root):
        files += 1
        for raw in path.read_text(encoding="utf-8").splitlines():
            fields = raw.strip().split()
            if len(fields) < 5:
                continue
            try:
                width = float(fields[3])
                height = float(fields[4])
            except ValueError:
                continue
            if width <= 0.0 or height <= 0.0:
                continue
            values.append(float(imgsz) * math.sqrt(width * height))
    if not values:
        raise ValueError(f"tidak ada bbox YOLO valid di {root}")
    return np.asarray(values, dtype=np.float64), files


def choose_patch_size(median_equivalent_side_px: float, candidates: Sequence[int] = (16, 32, 64)) -> int:
    """Frozen domain-calibration rule.

    Choose the largest candidate not exceeding the median equivalent bean side.  The intent is
    to maximize local context while keeping one spectral patch within the footprint of a typical
    bean rather than arbitrarily inheriting m=32 from the aircraft domain.  If every candidate is
    larger than the median object, the smallest candidate is used.
    """

    unique = tuple(sorted({int(v) for v in candidates if int(v) > 1}))
    if not unique:
        raise ValueError("candidates harus berisi ukuran patch >1")
    eligible = [value for value in unique if value <= median_equivalent_side_px]
    return max(eligible) if eligible else min(unique)


def calibrate_patch_size(
    label_root: str | Path,
    *,
    imgsz: int = 640,
    candidates: Sequence[int] = (16, 32, 64),
) -> PatchScaleReport:
    values, files = collect_equivalent_box_sides(label_root, imgsz=imgsz)
    q25, median, q75 = np.quantile(values, (0.25, 0.50, 0.75)).tolist()
    frozen_candidates = tuple(sorted({int(v) for v in candidates}))
    selected = choose_patch_size(median, frozen_candidates)
    return PatchScaleReport(
        label_files=files,
        boxes=int(values.size),
        imgsz=int(imgsz),
        q25_equivalent_side_px=float(q25),
        median_equivalent_side_px=float(median),
        q75_equivalent_side_px=float(q75),
        candidates=frozen_candidates,
        selected_patch_size=int(selected),
        selection_rule="largest_candidate_not_exceeding_median_equivalent_bbox_side",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate CAFR patch size from YOLO coffee labels")
    parser.add_argument("--labels", required=True, help="label .txt file or directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--candidates", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()
    report = calibrate_patch_size(args.labels, imgsz=args.imgsz, candidates=args.candidates)
    text = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
