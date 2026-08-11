from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .vadcp.profile import SceneCalibration


COMPOSITION_POLICIES = ("source_empirical", "defect_enriched")


def _quantile_sample(values: list[float], limit: int = 512) -> tuple[float, ...]:
    if not values:
        return ()
    array = np.asarray(values, dtype=np.float64)
    if array.size <= limit:
        return tuple(float(value) for value in array)
    return tuple(
        float(value)
        for value in np.quantile(array, np.linspace(0.0, 1.0, limit))
    )


def build_sni_crop_calibration(
    dataset_root: str | Path,
    *,
    policy: str,
    source_split: str = "train",
    normal_class: str = "biji_normal",
    enriched_normal_fraction: float = 0.55,
    objects_min: int = 220,
    objects_max: int = 300,
) -> tuple[dict[int, str], SceneCalibration, dict]:
    """Build a calibration from one explicitly selected crop-manifest split.

    Crop manifests contain appearance, class, and bbox-aspect information, but
    no full-scene scale or 300 g mass calibration.  Consequently count and
    object scale remain explicit preview parameters rather than empirical
    physical claims.
    """
    if policy not in COMPOSITION_POLICIES:
        raise ValueError(f"Policy komposisi tidak dikenal: {policy}")
    source_split = str(source_split).strip().lower()
    if source_split not in {"train", "val"}:
        raise ValueError("source_split kalibrasi hanya boleh train atau val")
    if objects_min <= 0 or objects_min > objects_max:
        raise ValueError("Rentang objek tidak valid")
    if not 0.5 < enriched_normal_fraction < 1.0:
        raise ValueError(
            "enriched_normal_fraction harus > 0.5 dan < 1 agar normal tetap dominan"
        )
    dataset_root = Path(dataset_root).expanduser().resolve()
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest crop tidak ditemukan: {manifest_path}")

    class_counts = Counter()
    ratios_by_class: dict[str, list[float]] = defaultdict(list)
    parent_ids: set[str] = set()
    required = {
        "generated_split",
        "source_identity",
        "canonical_class",
        "bbox_width",
        "bbox_height",
    }
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError("Kolom manifest belum lengkap: " + ", ".join(missing))
        for row in reader:
            if str(row["generated_split"]).strip().lower() != source_split:
                continue
            class_name = str(row["canonical_class"])
            class_counts[class_name] += 1
            parent_ids.add(str(row["source_identity"]))
            try:
                width = float(row["bbox_width"])
                height = float(row["bbox_height"])
            except (TypeError, ValueError):
                continue
            if width > 0 and height > 0:
                ratios_by_class[class_name].append(width / height)
    if not class_counts:
        raise RuntimeError(f"Manifest tidak memiliki crop {source_split}")
    if normal_class not in class_counts:
        raise ValueError(f"Kelas normal tidak ditemukan: {normal_class}")

    class_names = sorted(class_counts)
    names = {index: name for index, name in enumerate(class_names)}
    class_ids = {name: index for index, name in names.items()}
    total = sum(class_counts.values())
    source_probabilities = {
        class_ids[name]: count / total
        for name, count in sorted(class_counts.items())
    }
    if policy == "source_empirical":
        probabilities = source_probabilities
    else:
        defect_total = total - class_counts[normal_class]
        probabilities = {}
        for class_name in class_names:
            class_id = class_ids[class_name]
            if class_name == normal_class:
                probabilities[class_id] = enriched_normal_fraction
            else:
                probabilities[class_id] = (
                    (1.0 - enriched_normal_fraction)
                    * class_counts[class_name]
                    / defect_total
                )

    all_ratios = [
        value for values in ratios_by_class.values() for value in values
    ]
    if not all_ratios:
        all_ratios = [1.4]
    ratio_by_id = {
        class_ids[class_name]: _quantile_sample(values)
        for class_name, values in ratios_by_class.items()
        if values
    }
    normal_fraction_source = class_counts[normal_class] / total
    calibration = SceneCalibration(
        scene_counts=(objects_min, objects_max),
        object_long_sides=(0.04,),
        bbox_width_height_ratios=_quantile_sample(all_ratios, 1024),
        scene_scale_medians=(0.04,),
        within_scene_scale_ratios=(1.0,),
        background_colors=((236.0, 235.0, 231.0),),
        background_gradient_std=(2.0,),
        background_sensor_std=(1.0,),
        source_images=len(parent_ids),
        source_boxes=total,
        bbox_width_height_ratios_by_class=ratio_by_id,
        canvas_width_height_ratios=(0.75,),
        class_probabilities=probabilities,
        split=f"{source_split}_crop_manifest",
    )
    report = {
        "policy": policy,
        "claim": (
            f"source_empirical adalah distribusi paket crop {source_split}, bukan estimasi "
            "prevalensi populasi kopi 300 g"
        ),
        "source_split": source_split,
        "normal_class": normal_class,
        "source_crops": total,
        "source_train_crops": total if source_split == "train" else None,
        "source_parent_images": len(parent_ids),
        "source_class_counts": dict(sorted(class_counts.items())),
        "source_normal_fraction": normal_fraction_source,
        "requested_normal_fraction": (
            normal_fraction_source
            if policy == "source_empirical"
            else enriched_normal_fraction
        ),
        "class_probabilities": {
            names[class_id]: value
            for class_id, value in sorted(probabilities.items())
        },
        "objects_per_scene": [objects_min, objects_max],
        "mass_claim": False,
    }
    return names, calibration, report
