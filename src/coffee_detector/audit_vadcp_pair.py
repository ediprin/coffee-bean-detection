from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_VISIBILITY_SENSITIVE_CLASSES = (
    "biji_muda",
    "biji_bertutul_tutul",
)


def _summary(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"n": 0}
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


def _ratio_close(left: object, right: object, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _load_arm(root: str | Path) -> dict:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "metadata" / "generation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Metadata simulator belum lengkap di {root / 'metadata'}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    synthetic_split = str(manifest.get("synthetic_split", "train"))
    metadata_path = (
        root / "metadata" / f"instances_synthetic_{synthetic_split}.json"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata simulator belum lengkap di {root / 'metadata'}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    categories = {
        int(row["id"]): str(row["name"]) for row in metadata["categories"]
    }
    images = {int(row["id"]): row for row in metadata["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for row in metadata["annotations"]:
        annotations[int(row["image_id"])].append(row)
    for rows in annotations.values():
        rows.sort(key=lambda item: int(item["z_order"]))
    scene_by_seed = {
        int(row["generation_seed"]): {
            "image": row,
            "annotations": annotations.get(image_id, []),
        }
        for image_id, row in images.items()
    }
    if len(scene_by_seed) != len(images):
        raise ValueError(f"generation_seed tidak unik pada {root}")
    return {
        "root": root,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
        "metadata": metadata,
        "manifest": manifest,
        "categories": categories,
        "images": images,
        "scene_by_seed": scene_by_seed,
    }


def _bbox_area_fraction(row: dict, image: dict, key: str = "bbox") -> float | None:
    bbox = row.get(key)
    if bbox is None:
        return None
    _, _, width, height = (float(value) for value in bbox)
    canvas = max(float(image["width"]) * float(image["height"]), 1.0)
    return width * height / canvas


def _arm_profile(arm: dict, visibility_thresholds: tuple[float, float]) -> dict:
    categories = arm["categories"]
    class_counts = Counter()
    labeled_class_counts = Counter()
    densities = []
    labeled_densities = []
    bbox_areas = []
    full_bbox_areas = []
    visibility = []
    ignored = 0
    per_class: dict[str, dict[str, list | int]] = defaultdict(
        lambda: {
            "visibility": [],
            "generated": 0,
            "labeled": 0,
            "ignored": 0,
        }
    )
    for scene in arm["scene_by_seed"].values():
        image = scene["image"]
        rows = scene["annotations"]
        densities.append(len(rows))
        labeled_densities.append(
            sum(not int(row.get("ignore", 0)) for row in rows)
        )
        for row in rows:
            class_name = categories[int(row["category_id"])]
            class_counts[class_name] += 1
            per_class[class_name]["generated"] += 1
            ratio = float(row["visibility_ratio"])
            visibility.append(ratio)
            per_class[class_name]["visibility"].append(ratio)
            full_area = _bbox_area_fraction(row, image, "full_bbox")
            if full_area is not None:
                full_bbox_areas.append(full_area)
            if int(row.get("ignore", 0)):
                ignored += 1
                per_class[class_name]["ignored"] += 1
                continue
            labeled_class_counts[class_name] += 1
            per_class[class_name]["labeled"] += 1
            area = _bbox_area_fraction(row, image)
            if area is not None:
                bbox_areas.append(area)

    moderate_threshold, severe_threshold = visibility_thresholds
    class_visibility = {}
    for class_name, row in sorted(per_class.items()):
        values = [float(value) for value in row.pop("visibility")]
        generated = int(row["generated"])
        class_visibility[class_name] = {
            **row,
            "visibility": _summary(values),
            "below_moderate_visibility": sum(
                value < moderate_threshold for value in values
            ),
            "below_moderate_visibility_rate": (
                sum(value < moderate_threshold for value in values)
                / max(generated, 1)
            ),
            "below_severe_visibility": sum(
                value < severe_threshold for value in values
            ),
            "below_severe_visibility_rate": (
                sum(value < severe_threshold for value in values)
                / max(generated, 1)
            ),
            "ignored_rate": int(row["ignored"]) / max(generated, 1),
        }
    generated = sum(class_counts.values())
    labeled = sum(labeled_class_counts.values())
    return {
        "mode": arm["manifest"].get("mode"),
        "preset": arm["manifest"].get("preset"),
        "scene_count": len(arm["images"]),
        "generated_instances": generated,
        "labeled_instances": labeled,
        "ignored_instances": ignored,
        "ignored_rate": ignored / max(generated, 1),
        "scene_density": _summary(densities),
        "labeled_scene_density": _summary(labeled_densities),
        "visible_bbox_area_fraction": _summary(bbox_areas),
        "full_bbox_area_fraction": _summary(full_bbox_areas),
        "visibility_ratio": _summary(visibility),
        "instances_by_class": dict(sorted(class_counts.items())),
        "labeled_instances_by_class": dict(sorted(labeled_class_counts.items())),
        "class_visibility": class_visibility,
    }


def _class_prior_distance(left: dict[str, int], right: dict[str, int]) -> dict:
    names = sorted(set(left) | set(right))
    left_array = np.asarray([left.get(name, 0) for name in names], dtype=np.float64)
    right_array = np.asarray([right.get(name, 0) for name in names], dtype=np.float64)
    left_array /= max(float(left_array.sum()), 1.0)
    right_array /= max(float(right_array.sum()), 1.0)
    midpoint = (left_array + right_array) / 2.0

    def kl(values: np.ndarray) -> float:
        active = values > 0
        return float(
            np.sum(values[active] * np.log2(values[active] / midpoint[active]))
        )

    return {
        "classes": names,
        "total_variation_distance": float(np.abs(left_array - right_array).sum() / 2),
        "jensen_shannon_divergence_bits": (kl(left_array) + kl(right_array)) / 2,
        "per_class_probability_delta_a2_minus_a1": {
            name: float(right_array[index] - left_array[index])
            for index, name in enumerate(names)
        },
    }


def _paired_contract(a1: dict, a2: dict) -> dict:
    a1_seeds = set(a1["scene_by_seed"])
    a2_seeds = set(a2["scene_by_seed"])
    common = sorted(a1_seeds & a2_seeds)
    mismatch_examples = []
    scene_annotation_count_matches = 0
    exact_asset_sequence_matches = 0
    exact_class_sequence_matches = 0
    target_geometry_sequence_matches = 0
    observable_full_area_matches = 0
    paired_instances = 0
    paired_visibility_delta = []
    per_class_visibility_delta: dict[str, list[float]] = defaultdict(list)

    for scene_seed in common:
        left = a1["scene_by_seed"][scene_seed]["annotations"]
        right = a2["scene_by_seed"][scene_seed]["annotations"]
        if len(left) == len(right):
            scene_annotation_count_matches += 1
        else:
            mismatch_examples.append(
                {
                    "scene_seed": scene_seed,
                    "field": "annotation_count",
                    "a1": len(left),
                    "a2": len(right),
                }
            )
        pair_count = min(len(left), len(right))
        asset_match = [
            str(left[index].get("source_asset_id"))
            == str(right[index].get("source_asset_id"))
            for index in range(pair_count)
        ]
        class_match = [
            int(left[index]["category_id"]) == int(right[index]["category_id"])
            for index in range(pair_count)
        ]
        geometry_match = [
            _ratio_close(
                left[index].get("target_bbox_ratio"),
                right[index].get("target_bbox_ratio"),
            )
            and _ratio_close(
                left[index].get("achieved_bbox_ratio"),
                right[index].get("achieved_bbox_ratio"),
            )
            for index in range(pair_count)
        ]
        full_area_match = [
            int(left[index].get("full_area", -1))
            == int(right[index].get("full_area", -2))
            for index in range(pair_count)
        ]
        exact_asset_sequence_matches += int(
            len(left) == len(right) and all(asset_match)
        )
        exact_class_sequence_matches += int(
            len(left) == len(right) and all(class_match)
        )
        target_geometry_sequence_matches += int(
            len(left) == len(right) and all(geometry_match)
        )
        observable_full_area_matches += sum(full_area_match)
        paired_instances += pair_count
        for index in range(pair_count):
            if not (
                asset_match[index] and class_match[index] and geometry_match[index]
            ) and len(mismatch_examples) < 50:
                mismatch_examples.append(
                    {
                        "scene_seed": scene_seed,
                        "z_order": index,
                        "field": "instance_plan",
                        "a1_asset": left[index].get("source_asset_id"),
                        "a2_asset": right[index].get("source_asset_id"),
                        "a1_class": left[index].get("category_id"),
                        "a2_class": right[index].get("category_id"),
                        "a1_target_ratio": left[index].get("target_bbox_ratio"),
                        "a2_target_ratio": right[index].get("target_bbox_ratio"),
                    }
                )
            if class_match[index]:
                delta = float(right[index]["visibility_ratio"]) - float(
                    left[index]["visibility_ratio"]
                )
                paired_visibility_delta.append(delta)
                class_name = a1["categories"][int(left[index]["category_id"])]
                per_class_visibility_delta[class_name].append(delta)

    common_count = len(common)
    selection_plan_valid = bool(
        common_count
        and a1_seeds == a2_seeds
        and scene_annotation_count_matches == common_count
        and exact_asset_sequence_matches == common_count
        and exact_class_sequence_matches == common_count
        and target_geometry_sequence_matches == common_count
    )
    return {
        "a1_scene_count": len(a1_seeds),
        "a2_scene_count": len(a2_seeds),
        "common_scene_count": common_count,
        "missing_scene_seeds_in_a2": sorted(a1_seeds - a2_seeds)[:50],
        "extra_scene_seeds_in_a2": sorted(a2_seeds - a1_seeds)[:50],
        "scene_annotation_count_matches": scene_annotation_count_matches,
        "exact_asset_sequence_matches": exact_asset_sequence_matches,
        "exact_class_sequence_matches": exact_class_sequence_matches,
        "target_geometry_sequence_matches": target_geometry_sequence_matches,
        "paired_instances": paired_instances,
        "observable_full_area_matches": observable_full_area_matches,
        "observable_full_area_match_rate": (
            observable_full_area_matches / max(paired_instances, 1)
        ),
        "selection_and_geometry_plan_valid": selection_plan_valid,
        "paired_visibility_delta_a2_minus_a1": _summary(
            paired_visibility_delta
        ),
        "paired_visibility_delta_by_class": {
            name: _summary(values)
            for name, values in sorted(per_class_visibility_delta.items())
        },
        "mismatch_examples": mismatch_examples[:50],
        "note": (
            "Full-area equality is diagnostic only: placement at a canvas edge can "
            "clip the full mask. The frozen code-level compositor test is the "
            "stronger invariant for identical transforms."
        ),
    }


def audit_vadcp_pair(
    a1_root: str | Path,
    a2_root: str | Path,
    output: str | Path,
    *,
    visibility_sensitive_classes: Iterable[str] = DEFAULT_VISIBILITY_SENSITIVE_CLASSES,
    moderate_visibility: float = 0.75,
    severe_visibility: float = 0.50,
    real_train_boxes: int | None = None,
    real_median_bbox_area: float | None = None,
) -> dict:
    if not 0 < severe_visibility < moderate_visibility <= 1:
        raise ValueError(
            "Threshold harus memenuhi 0 < severe < moderate <= 1"
        )
    a1 = _load_arm(a1_root)
    a2 = _load_arm(a2_root)
    if a1["categories"] != a2["categories"]:
        raise ValueError("Urutan/nama kelas A1 dan A2 berbeda")

    thresholds = (moderate_visibility, severe_visibility)
    a1_profile = _arm_profile(a1, thresholds)
    a2_profile = _arm_profile(a2, thresholds)
    contract = _paired_contract(a1, a2)
    prior = _class_prior_distance(
        a1_profile["instances_by_class"],
        a2_profile["instances_by_class"],
    )
    sensitive = tuple(dict.fromkeys(str(name) for name in visibility_sensitive_classes))
    missing_sensitive = [
        name for name in sensitive if name not in a2_profile["class_visibility"]
    ]
    sensitive_risk = {
        name: {
            "a1": a1_profile["class_visibility"][name],
            "a2": a2_profile["class_visibility"][name],
            "paired_visibility_delta_a2_minus_a1": contract[
                "paired_visibility_delta_by_class"
            ].get(name, {"n": 0}),
        }
        for name in sensitive
        if name in a2_profile["class_visibility"]
    }
    severe_sensitive_instances = sum(
        row["a2"]["below_severe_visibility"] for row in sensitive_risk.values()
    )
    sensitive_instances = sum(
        row["a2"]["generated"] for row in sensitive_risk.values()
    )
    label_risk_review = severe_sensitive_instances > 0

    dominance = None
    if real_train_boxes is not None:
        if real_train_boxes <= 0:
            raise ValueError("real_train_boxes harus positif")
        synthetic = a2_profile["labeled_instances"]
        dominance = {
            "real_train_boxes": int(real_train_boxes),
            "synthetic_labeled_boxes": int(synthetic),
            "synthetic_to_real_ratio": synthetic / real_train_boxes,
            "synthetic_share_of_mixed_train_boxes": synthetic
            / (synthetic + real_train_boxes),
        }
    scale_shift = None
    if real_median_bbox_area is not None:
        if real_median_bbox_area <= 0:
            raise ValueError("real_median_bbox_area harus positif")
        synthetic_median = a2_profile["visible_bbox_area_fraction"].get("q50")
        scale_shift = {
            "real_median_bbox_area_fraction": float(real_median_bbox_area),
            "a2_median_bbox_area_fraction": synthetic_median,
            "a2_to_real_ratio": (
                float(synthetic_median) / real_median_bbox_area
                if synthetic_median is not None
                else None
            ),
            "real_to_a2_ratio": (
                real_median_bbox_area / float(synthetic_median)
                if synthetic_median
                else None
            ),
        }

    if not contract["selection_and_geometry_plan_valid"]:
        status = "INVALID_PAIRING"
    elif label_risk_review:
        status = "REVIEW_LABEL_RISK"
    else:
        status = "INTERNALLY_VALID"
    report = {
        "format": "coffee_detector.vadcp_pair_audit.v1",
        "status": status,
        "a1_root": str(a1["root"]),
        "a2_root": str(a2["root"]),
        "claim_scope": (
            "Internal paired-simulator quality audit only. It does not establish "
            "photorealism, semantic visibility, or transfer to real dense scenes."
        ),
        "thresholds": {
            "moderate_visibility": moderate_visibility,
            "severe_visibility": severe_visibility,
        },
        "paired_contract": contract,
        "class_prior_comparison": prior,
        "arms": {"A1": a1_profile, "A2": a2_profile},
        "visibility_sensitive_classes": list(sensitive),
        "missing_visibility_sensitive_classes": missing_sensitive,
        "semantic_label_risk_proxy": {
            "review_required": label_risk_review,
            "sensitive_instances": sensitive_instances,
            "sensitive_instances_below_severe_visibility": severe_sensitive_instances,
            "sensitive_severe_visibility_rate": (
                severe_sensitive_instances / max(sensitive_instances, 1)
            ),
            "per_class": sensitive_risk,
            "interpretation": (
                "Overall visible-instance area is only a risk proxy. Without an "
                "annotation of the discriminative defect region, this audit cannot "
                "prove whether the class-defining cue remains visible."
            ),
        },
        "mixed_training_dominance": dominance,
        "real_to_synthetic_scale_shift": scale_shift,
        "real_dense_validation_available": False,
        "realism_decision": "NOT_ESTABLISHED",
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_pair_audit_summary(report: dict) -> None:
    contract = report["paired_contract"]
    a1, a2 = report["arms"]["A1"], report["arms"]["A2"]
    risk = report["semantic_label_risk_proxy"]
    print("=== AUDIT BERPASANGAN SIMULATOR A1/A2 ===")
    print("Status                 :", report["status"])
    print(
        "Scene A1/A2            :",
        f"{a1['scene_count']}/{a2['scene_count']}",
    )
    print(
        "Instance A1/A2         :",
        f"{a1['generated_instances']}/{a2['generated_instances']}",
    )
    print(
        "Selection plan valid   :",
        contract["selection_and_geometry_plan_valid"],
    )
    print(
        "Class prior TV distance:",
        f"{report['class_prior_comparison']['total_variation_distance']:.6f}",
    )
    print(
        "Visibility median A1/A2:",
        f"{a1['visibility_ratio'].get('q50', 0):.2%}/"
        f"{a2['visibility_ratio'].get('q50', 0):.2%}",
    )
    print(
        "Ignored rate A1/A2     :",
        f"{a1['ignored_rate']:.2%}/{a2['ignored_rate']:.2%}",
    )
    print(
        "Sensitive severe risk  :",
        f"{risk['sensitive_instances_below_severe_visibility']}/"
        f"{risk['sensitive_instances']} "
        f"({risk['sensitive_severe_visibility_rate']:.2%})",
    )
    if report["mixed_training_dominance"]:
        dominance = report["mixed_training_dominance"]
        print(
            "Synthetic box share    :",
            f"{dominance['synthetic_share_of_mixed_train_boxes']:.2%}",
        )
    if report["real_to_synthetic_scale_shift"]:
        shift = report["real_to_synthetic_scale_shift"]
        print(
            "A2/real median area    :",
            f"{shift['a2_to_real_ratio']:.3f}x",
        )
    print("Real dense realism     :", report["realism_decision"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit kontrak berpasangan, prior kelas, visibility, dan proxy risiko "
            "label simulator dense A1/A2 tanpa training."
        )
    )
    parser.add_argument("--a1-root", required=True)
    parser.add_argument("--a2-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--visibility-sensitive-class",
        action="append",
        dest="sensitive_classes",
        help=(
            "Kelas dengan ciri lokal yang perlu diaudit; dapat diulang. "
            "Default: biji_muda dan biji_bertutul_tutul."
        ),
    )
    parser.add_argument("--moderate-visibility", type=float, default=0.75)
    parser.add_argument("--severe-visibility", type=float, default=0.50)
    parser.add_argument("--real-train-boxes", type=int)
    parser.add_argument("--real-median-bbox-area", type=float)
    args = parser.parse_args()
    report = audit_vadcp_pair(
        args.a1_root,
        args.a2_root,
        args.output,
        visibility_sensitive_classes=(
            args.sensitive_classes
            if args.sensitive_classes
            else DEFAULT_VISIBILITY_SENSITIVE_CLASSES
        ),
        moderate_visibility=args.moderate_visibility,
        severe_visibility=args.severe_visibility,
        real_train_boxes=args.real_train_boxes,
        real_median_bbox_area=args.real_median_bbox_area,
    )
    print_pair_audit_summary(report)


if __name__ == "__main__":
    main()
