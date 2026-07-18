from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import IMAGE_SUFFIXES, discover_layout, parse_label


def _pairwise_scene_metrics(boxes: list[tuple[float, float, float, float]]) -> tuple[list[float], float]:
    if len(boxes) < 2:
        return [], 0.0
    array = np.asarray(boxes[:128], dtype=np.float64)
    centers = array[:, :2]
    sizes = array[:, 2:]
    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(delta**2, axis=2))
    np.fill_diagonal(distances, np.inf)
    long_sides = np.maximum(sizes[:, 0], sizes[:, 1])
    normalizer = (long_sides[:, None] + long_sides[None, :]) / 2.0
    nearest = np.min(distances / np.maximum(normalizer, 1e-8), axis=1)

    left = centers[:, 0] - sizes[:, 0] / 2
    right = centers[:, 0] + sizes[:, 0] / 2
    top = centers[:, 1] - sizes[:, 1] / 2
    bottom = centers[:, 1] + sizes[:, 1] / 2
    inter_w = np.maximum(0.0, np.minimum(right[:, None], right[None, :]) - np.maximum(left[:, None], left[None, :]))
    inter_h = np.maximum(0.0, np.minimum(bottom[:, None], bottom[None, :]) - np.maximum(top[:, None], top[None, :]))
    intersection = inter_w * inter_h
    upper = np.triu_indices(len(array), 1)
    overlap_rate = float(np.mean(intersection[upper] > 0)) if upper[0].size else 0.0
    return [float(value) for value in nearest], overlap_rate


def _append_scene(profile: dict[str, list[float]], boxes: list[tuple[float, float, float, float]]) -> None:
    profile["labeled_density"].append(float(len(boxes)))
    if not boxes:
        return
    nearest, overlap_rate = _pairwise_scene_metrics(boxes)
    profile["nearest_center_normalized"].extend(nearest)
    profile["pair_overlap_rate"].append(overlap_rate)
    profile["summed_bbox_coverage"].append(float(sum(width * height for _, _, width, height in boxes)))
    for x, y, width, height in boxes:
        profile["long_side_fraction"].append(max(width, height))
        profile["bbox_area_fraction"].append(width * height)
        profile["absolute_aspect_ratio"].append(max(width / height, height / width))
        touches = (
            x - width / 2 <= 0.002
            or y - height / 2 <= 0.002
            or x + width / 2 >= 0.998
            or y + height / 2 >= 0.998
        )
        profile["border_touch"].append(float(touches))


def _real_profile(data_root: str | Path) -> dict[str, list[float]]:
    layout = discover_layout(data_root)
    image_root, label_root = layout.splits["train"]
    profile: dict[str, list[float]] = defaultdict(list)
    valid_ids = set(layout.names)
    image_paths = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for index, image_path in enumerate(image_paths, 1):
        relative = image_path.relative_to(image_root)
        rows = parse_label((label_root / relative).with_suffix(".txt"), valid_ids)
        boxes = [(row.x_center, row.y_center, row.width, row.height) for row in rows]
        _append_scene(profile, boxes)
        if index % 2000 == 0 or index == len(image_paths):
            print(f"  realism real: {index}/{len(image_paths)} gambar", flush=True)
    return dict(profile)


def build_realism_reference(
    real_data_root: str | Path,
) -> dict[str, list[float]]:
    """Build one reusable train-only reference for several synthetic arms."""
    return _real_profile(real_data_root)


def _synthetic_profile(
    data_root: str | Path,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict]:
    data_root = Path(data_root).expanduser().resolve()
    metadata = json.loads(
        (data_root / "metadata" / "instances_synthetic_train.json").read_text(
            encoding="utf-8"
        )
    )
    images_by_id = {int(item["id"]): item for item in metadata["images"]}
    by_image: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    full_by_image: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    visibility = []
    ignored = 0
    focus = 0
    for row in metadata["annotations"]:
        visibility.append(float(row["visibility_ratio"]))
        ignored += int(row.get("ignore", 0))
        focus += int(bool(row.get("is_focus")))
        image = images_by_id[int(row["image_id"])]
        width, height = float(image["width"]), float(image["height"])
        if row.get("full_bbox") is not None:
            x, y, box_width, box_height = (
                float(value) for value in row["full_bbox"]
            )
            full_by_image[int(row["image_id"])].append(
                (
                    (x + box_width / 2) / width,
                    (y + box_height / 2) / height,
                    box_width / width,
                    box_height / height,
                )
            )
        if int(row.get("ignore", 0)) or row.get("bbox") is None:
            continue
        x, y, box_width, box_height = (float(value) for value in row["bbox"])
        by_image[int(row["image_id"])].append(
            (
                (x + box_width / 2) / width,
                (y + box_height / 2) / height,
                box_width / width,
                box_height / height,
            )
        )
    profile: dict[str, list[float]] = defaultdict(list)
    full_profile: dict[str, list[float]] = defaultdict(list)
    for image in metadata["images"]:
        image_id = int(image["id"])
        _append_scene(profile, by_image.get(image_id, []))
        _append_scene(full_profile, full_by_image.get(image_id, []))
    physics = {
        "generated_instances": len(metadata["annotations"]),
        "labeled_instances": len(metadata["annotations"]) - ignored,
        "ignored_instances": ignored,
        "ignored_rate": ignored / max(len(metadata["annotations"]), 1),
        "focus_instances": focus,
        "visibility": _summary(visibility),
    }
    return dict(profile), dict(full_profile), physics


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, (0.05, 0.25, 0.50, 0.75, 0.95))
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        **{
            name: float(value)
            for name, value in zip(("q05", "q25", "q50", "q75", "q95"), quantiles)
        },
    }


def _quantile_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    points = np.linspace(0.05, 0.95, 19)
    left, right = np.quantile(reference, points), np.quantile(candidate, points)
    iqr = max(float(np.quantile(reference, 0.75) - np.quantile(reference, 0.25)), 1e-8)
    return float(np.mean(np.abs(left - right)) / iqr)


def _comparison(
    real_values: list[float],
    synthetic_values: list[float],
    rng: random.Random,
    repeats: int = 20,
) -> dict:
    real = np.asarray(real_values, dtype=np.float64)
    synthetic = np.asarray(synthetic_values, dtype=np.float64)
    if not real.size or not synthetic.size:
        return {"status": "insufficient_data"}
    distance = _quantile_distance(real, synthetic)
    sample_size = min(real.size // 2, synthetic.size)
    null = []
    if sample_size >= 5:
        numpy_rng = np.random.default_rng(rng.getrandbits(64))
        for _ in range(repeats):
            indices = numpy_rng.choice(real.size, size=sample_size * 2, replace=False)
            left, right = indices[:sample_size], indices[sample_size:]
            null.append(_quantile_distance(real[left], real[right]))
    null_p95 = float(np.quantile(null, 0.95)) if null else None
    return {
        "normalized_quantile_distance": distance,
        "real_real_null_median": float(np.median(null)) if null else None,
        "real_real_null_p95": null_p95,
        "excess_ratio": distance / max(null_p95, 1e-8) if null_p95 else None,
        "status": (
            "within_real_sampling_variation"
            if null_p95 is not None and distance <= null_p95
            else "shifted"
        ),
    }


def audit_vadcp_realism(
    real_data_root: str | Path,
    synthetic_data_root: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    real_profile: dict[str, list[float]] | None = None,
) -> dict:
    real = (
        real_profile
        if real_profile is not None
        else build_realism_reference(real_data_root)
    )
    synthetic, synthetic_full, physics = _synthetic_profile(synthetic_data_root)
    rng = random.Random(seed)
    comparisons = {
        name: _comparison(real.get(name, []), synthetic.get(name, []), rng)
        for name in sorted(set(real) | set(synthetic))
    }
    for name, values in sorted(synthetic_full.items()):
        comparisons[f"full_{name}"] = _comparison(
            real.get(name, []), values, rng
        )
    critical = (
        "labeled_density",
        "full_long_side_fraction",
        "full_absolute_aspect_ratio",
    )
    critical_shifts = [
        name for name in critical if comparisons.get(name, {}).get("status") == "shifted"
    ]
    report = {
        "format": "coffee_detector.vadcp_realism_audit.v1",
        "real_data_root": str(Path(real_data_root).expanduser().resolve()),
        "synthetic_data_root": str(Path(synthetic_data_root).expanduser().resolve()),
        "profiles": {
            "real": {name: _summary(values) for name, values in sorted(real.items())},
            "synthetic": {
                name: _summary(values) for name, values in sorted(synthetic.items())
            },
            "synthetic_full": {
                name: _summary(values)
                for name, values in sorted(synthetic_full.items())
            },
        },
        "comparisons": comparisons,
        "synthetic_physics": physics,
        "critical_shifts": critical_shifts,
        "realism_status": "PASS_GEOMETRY" if not critical_shifts else "REVIEW",
        "note": "Status geometri tidak menggantikan audit visual atau background nyata.",
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_realism_summary(report: dict) -> None:
    real = report["profiles"]["real"]
    synthetic = report["profiles"]["synthetic"]
    synthetic_full = report["profiles"]["synthetic_full"]
    print("=== AUDIT REALISME VA-DCP ===")
    print("Status             :", report["realism_status"])
    print("Critical shifts    :", report["critical_shifts"])
    for name in ("labeled_density", "long_side_fraction", "absolute_aspect_ratio", "nearest_center_normalized", "pair_overlap_rate"):
        if name not in real or name not in synthetic:
            continue
        comparison = report["comparisons"][name]
        print(
            f"{name:25s}: real_q50={real[name].get('q50', 0):.4f} "
            f"synth_q50={synthetic[name].get('q50', 0):.4f} "
            f"distance={comparison.get('normalized_quantile_distance', 0):.3f} "
            f"[{comparison.get('status')}]"
        )
    if "long_side_fraction" in synthetic_full:
        comparison = report["comparisons"]["full_long_side_fraction"]
        print(
            "full_long_side_fraction  : "
            f"real_q50={real['long_side_fraction'].get('q50', 0):.4f} "
            f"synth_q50={synthetic_full['long_side_fraction'].get('q50', 0):.4f} "
            f"distance={comparison.get('normalized_quantile_distance', 0):.3f} "
            f"[{comparison.get('status')}]"
        )
    physics = report["synthetic_physics"]
    print(
        "Ignored            : "
        f"{physics['ignored_instances']}/{physics['generated_instances']} "
        f"({physics['ignored_rate']:.2%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bandingkan geometri synthetic VA-DCP dengan train nyata."
    )
    parser.add_argument("--real-data-root", required=True)
    parser.add_argument("--synthetic-data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = audit_vadcp_realism(
        args.real_data_root,
        args.synthetic_data_root,
        args.output,
        seed=args.seed,
    )
    print_realism_summary(report)


if __name__ == "__main__":
    main()
