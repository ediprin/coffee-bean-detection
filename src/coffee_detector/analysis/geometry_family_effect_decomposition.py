from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SOURCE_PROTOCOL = "faruq-v3-geometry-conditioning-paired-confirmation-v1"
PROTOCOL = "faruq-v3-geometry-family-effect-decomposition-v1"
SEEDS = (42, 123, 2026)
FAMILIES = {
    "kulit_kopi": (
        "kulit_kopi_ukuran_besar",
        "kulit_kopi_ukuran_kecil",
        "kulit_kopi_ukuran_sedang",
    ),
    "kulit_tanduk": (
        "kulit_tanduk_ukuran_besar",
        "kulit_tanduk_ukuran_kecil",
        "kulit_tanduk_ukuran_sedang",
    ),
    "tanah_batu_ranting": (
        "tanah_batu_ranting_besar",
        "tanah_batu_ranting_kecil",
        "tanah_batu_ranting_sedang",
    ),
}


def _load_source(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol") != SOURCE_PROTOCOL:
        raise RuntimeError(f"Protocol source tidak kompatibel: {payload.get('protocol')}")
    if tuple(int(seed) for seed in payload.get("seeds", ())) != SEEDS:
        raise RuntimeError("Source bukan paired confirmation seed 42/123/2026")
    if payload.get("evaluation_split") != "val":
        raise RuntimeError("Decomposition hanya boleh memakai validation")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError("Source menunjukkan locked test pernah dibuka")
    if payload.get("decision") != "PASS":
        raise RuntimeError("Decomposition dikunci pada paired confirmation PASS")
    return payload


def _size_maps(record: dict) -> tuple[dict[str, float], dict[str, float]]:
    results = record.get("results", {})
    control = results.get("GEO-C0", {}).get("size_map50_95_by_class", {})
    geometry = results.get("GEO1", {}).get("size_map50_95_by_class", {})
    if not isinstance(control, dict) or not isinstance(geometry, dict):
        raise RuntimeError("Per-class size AP tidak tersedia")
    expected = {name for classes in FAMILIES.values() for name in classes}
    missing = sorted(expected - set(control) | expected - set(geometry))
    if missing:
        raise RuntimeError(f"Kelas size-defined hilang: {missing}")
    return (
        {name: float(control[name]) for name in expected},
        {name: float(geometry[name]) for name in expected},
    )


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_delta": float(array.mean()),
        "min_delta": float(array.min()),
        "max_delta": float(array.max()),
        "positive_seeds": int((array > 0.0).sum()),
        "negative_seeds": int((array < 0.0).sum()),
        "zero_seeds": int((array == 0.0).sum()),
        "deltas_by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
    }


def _pattern(summary: dict) -> str:
    if summary["positive_seeds"] == len(SEEDS):
        return "positive_3_of_3"
    if summary["negative_seeds"] == len(SEEDS):
        return "negative_3_of_3"
    return "mixed_across_seeds"


def decompose_geometry_family_effects(source: str | Path, output: str | Path) -> dict:
    payload = _load_source(source)
    per_seed: dict[str, dict] = {}
    class_values: dict[str, list[float]] = {
        name: [] for classes in FAMILIES.values() for name in classes
    }
    family_values: dict[str, list[float]] = {family: [] for family in FAMILIES}

    for seed in SEEDS:
        record = payload["per_seed"].get(str(seed))
        if not isinstance(record, dict):
            raise RuntimeError(f"Record seed {seed} tidak tersedia")
        control, geometry = _size_maps(record)
        class_delta = {name: geometry[name] - control[name] for name in control}
        family_delta = {}
        family_detail = {}
        for family, classes in FAMILIES.items():
            values = [class_delta[name] for name in classes]
            family_delta[family] = float(np.mean(values))
            family_detail[family] = {
                "classes": list(classes),
                "class_deltas": {name: float(class_delta[name]) for name in classes},
                "mean_delta": family_delta[family],
                "positive_classes": int(sum(value > 0.0 for value in values)),
                "negative_classes": int(sum(value < 0.0 for value in values)),
                "within_family_range": float(max(values) - min(values)),
            }
            family_values[family].append(family_delta[family])
        for name, value in class_delta.items():
            class_values[name].append(float(value))
        per_seed[str(seed)] = {
            "class_deltas": {name: float(value) for name, value in sorted(class_delta.items())},
            "families": family_detail,
        }

    aggregate_classes = {}
    for family, classes in FAMILIES.items():
        for name in classes:
            summary = _summary(class_values[name])
            aggregate_classes[name] = {"family": family, **summary, "pattern": _pattern(summary)}

    aggregate_families = {}
    for family in FAMILIES:
        summary = _summary(family_values[family])
        aggregate_families[family] = {**summary, "pattern": _pattern(summary)}

    family_contrasts = {}
    pairs = (
        ("kulit_tanduk", "kulit_kopi"),
        ("tanah_batu_ranting", "kulit_kopi"),
        ("kulit_tanduk", "tanah_batu_ranting"),
    )
    for left, right in pairs:
        values = [
            family_values[left][index] - family_values[right][index]
            for index in range(len(SEEDS))
        ]
        summary = _summary(values)
        family_contrasts[f"{left}_minus_{right}"] = {
            "left": left,
            "right": right,
            **summary,
            "pattern": _pattern(summary),
        }

    result = {
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "source": str(Path(source).expanduser().resolve()),
        "seeds": list(SEEDS),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "analysis_status": "posthoc_descriptive_decomposition_no_new_gate",
        "per_seed": per_seed,
        "aggregate_families": aggregate_families,
        "aggregate_classes": aggregate_classes,
        "family_contrasts": family_contrasts,
        "guardrails": [
            "No new model training or inference was executed.",
            "This decomposition was defined after some per-class three-seed results were already inspected; it is descriptive, not a prospective confirmation gate.",
            "Three seeds are insufficient for a strong inferential significance claim; report paired effect sizes and sign consistency instead.",
            "Family-level means do not imply every class in a family benefits.",
            "No family-aware architecture is authorized by this audit alone.",
        ],
        "next_action": "REVIEW_FAMILY_HETEROGENEITY_BEFORE_ANY_FAMILY_AWARE_MODEL",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["output"] = str(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Posthoc family decomposition of three-seed GEO1 minus GEO-C0 size-class AP effects")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = decompose_geometry_family_effects(args.source, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
