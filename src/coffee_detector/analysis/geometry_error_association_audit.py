from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from coffee_detector.dataset import discover_layout, parse_label

EVENT_PROTOCOL = "faruq-v3-validation-object-events-v1"
SCALE_PROTOCOL = "faruq-v3-scale-identifiability-audit-v1"
PROTOCOL = "faruq-v3-geometry-error-association-audit-v1"
FEATURES = ("long_side_norm", "area_norm")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_event(path: str | Path, expected_model: str) -> dict:
    payload = _load_json(path)
    if payload.get("protocol") != EVENT_PROTOCOL:
        raise RuntimeError(f"Event protocol tidak kompatibel: {path}")
    if payload.get("model") != expected_model:
        raise RuntimeError(f"Model event salah: expected={expected_model}, got={payload.get('model')}")
    if int(payload.get("seed", -1)) != 42 or payload.get("evaluation_split") != "val":
        raise RuntimeError(f"Event bukan seed42 val: {path}")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError(f"Event menunjukkan akses test: {path}")
    return payload


def _load_scale(path: str | Path) -> dict:
    payload = _load_json(path)
    if payload.get("protocol") != SCALE_PROTOCOL:
        raise RuntimeError("Scale-identifiability JSON tidak kompatibel")
    if int(payload.get("seed", -1)) != 42 or payload.get("evaluation_split") != "val":
        raise RuntimeError("Scale audit bukan seed42 val")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError("Scale audit menunjukkan akses test")
    return payload


def _pair_name(a: str, b: str) -> str:
    return " <-> ".join(sorted((str(a), str(b))))


def _pair_error(row: dict, family: str) -> bool:
    return bool(
        row.get("matched")
        and not row.get("correct")
        and row.get("gt_class_name") is not None
        and row.get("pred_class_name") is not None
        and _pair_name(row["gt_class_name"], row["pred_class_name"]) == family
    )


def _feature_rows(data_root: Path, events: dict) -> dict[str, dict]:
    layout = discover_layout(data_root)
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit menolak data root yang mengekspos test")
    image_root, label_root = layout.splits["val"]
    cache = {}
    rows = {}
    for key, event in events.items():
        image_key = str(event["image"])
        if image_key not in cache:
            image_path = image_root / image_key
            image = cv2.imread(str(image_path))
            if image is None:
                raise OSError(f"Gagal baca gambar: {image_path}")
            h, w = image.shape[:2]
            label_path = (label_root / Path(image_key)).with_suffix(".txt")
            annotations = parse_label(label_path, set(layout.names))
            cache[image_key] = (h, w, annotations)
        h, w, annotations = cache[image_key]
        idx = int(event["target_index"])
        ann = annotations[idx]
        bw = float(ann.width)
        bh = float(ann.height)
        rows[key] = {
            "gt_class_name": str(event["gt_class_name"]),
            "long_side_norm": max(bw, bh),
            "area_norm": bw * bh,
        }
    return rows


def _geometry_prediction(value: float, threshold: float, low_class: str, high_class: str) -> str:
    return low_class if float(value) <= float(threshold) else high_class


def _geometry_state(feature_row: dict, pair: dict) -> dict:
    gt = feature_row["gt_class_name"]
    votes = {}
    for feature in FEATURES:
        threshold = float(pair["features"][feature]["best_threshold"])
        predicted = _geometry_prediction(
            feature_row[feature], threshold, pair["low_class"], pair["high_class"]
        )
        votes[feature] = predicted
    supports = sum(pred == gt for pred in votes.values())
    if supports == 2:
        state = "both_support_gt"
    elif supports == 0:
        state = "both_support_other"
    else:
        state = "mixed_geometry"
    return {"state": state, "votes": votes}


def _safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def run(cpe0_event, cir0_event, scale_json, data_root, output) -> dict:
    cpe = _load_event(cpe0_event, "CPE0")
    cir = _load_event(cir0_event, "CIR0")
    if set(cpe.get("events", {})) != set(cir.get("events", {})):
        raise RuntimeError("Target universe CPE0/CIR0 berbeda")
    scale = _load_scale(scale_json)
    data_root = Path(data_root).expanduser().resolve()
    features = _feature_rows(data_root, cpe["events"])

    pair_results = []
    global_by_model = {"CPE0": Counter(), "CIR0": Counter()}
    global_gt_states = Counter()

    for pair in scale.get("pairs", []):
        family = str(pair["family"])
        low_class = str(pair["low_class"])
        high_class = str(pair["high_class"])
        pair_keys = [
            key for key, row in features.items()
            if row["gt_class_name"] in (low_class, high_class)
        ]
        state_by_key = {}
        gt_state_counts = Counter()
        for key in pair_keys:
            state = _geometry_state(features[key], pair)["state"]
            state_by_key[key] = state
            gt_state_counts[state] += 1
            global_gt_states[state] += 1

        model_summaries = {}
        for model_name, payload in (("CPE0", cpe), ("CIR0", cir)):
            errors = [key for key in pair_keys if _pair_error(payload["events"][key], family)]
            error_states = Counter(state_by_key[key] for key in errors)
            for state, count in error_states.items():
                global_by_model[model_name][state] += count
            global_by_model[model_name]["pair_errors_total"] += len(errors)
            model_summaries[model_name] = {
                "pair_errors_total": len(errors),
                "pair_errors_by_geometry_state": dict(error_states),
                "pair_error_share_by_geometry_state": {
                    state: _safe_rate(error_states.get(state, 0), len(errors))
                    for state in ("both_support_gt", "mixed_geometry", "both_support_other")
                },
                "pair_error_rate_within_geometry_state": {
                    state: _safe_rate(error_states.get(state, 0), gt_state_counts.get(state, 0))
                    for state in ("both_support_gt", "mixed_geometry", "both_support_other")
                },
            }

        pair_results.append({
            "family": family,
            "low_class": low_class,
            "high_class": high_class,
            "gt_instances_in_pair": len(pair_keys),
            "geometry_state_counts": dict(gt_state_counts),
            "thresholds_reused_from_scale_audit": {
                feature: float(pair["features"][feature]["best_threshold"])
                for feature in FEATURES
            },
            "models": model_summaries,
        })

    global_summary = {}
    for model_name in ("CPE0", "CIR0"):
        counts = global_by_model[model_name]
        total = int(counts.get("pair_errors_total", 0))
        global_summary[model_name] = {
            "pair_errors_total_across_size_pairs": total,
            "pair_errors_by_geometry_state": {
                state: int(counts.get(state, 0))
                for state in ("both_support_gt", "mixed_geometry", "both_support_other")
            },
            "pair_error_share_by_geometry_state": {
                state: _safe_rate(int(counts.get(state, 0)), total)
                for state in ("both_support_gt", "mixed_geometry", "both_support_other")
            },
            "pair_error_rate_within_geometry_state": {
                state: _safe_rate(int(counts.get(state, 0)), int(global_gt_states.get(state, 0)))
                for state in ("both_support_gt", "mixed_geometry", "both_support_other")
            },
        }

    result = {
        "protocol": PROTOCOL,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "source_scale_audit": str(Path(scale_json).expanduser().resolve()),
        "method_note": (
            "Thresholds are reused from the prior validation-only scale-identifiability audit. "
            "This is a descriptive same-split association diagnostic, not an independent predictive test."
        ),
        "geometry_state_definition": {
            "both_support_gt": "long-side and area thresholds both predict the GT size class",
            "mixed_geometry": "one threshold predicts GT and one predicts the opposite class",
            "both_support_other": "both thresholds predict the opposite size class",
        },
        "global_geometry_state_counts_across_pair_memberships": dict(global_gt_states),
        "global_model_summary": global_summary,
        "pairs": pair_results,
        "screening_decision_remains": "STOP_CIRCLE_CPE",
        "interpretation_guardrail": (
            "A high share of errors in both_support_gt supports an available-but-underused geometry-cue hypothesis. "
            "A high share in both_support_other instead supports geometry/label/acquisition ambiguity. "
            "Neither pattern alone proves a causal mechanism or label error."
        ),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", destination)
    print("GLOBAL:", result["global_model_summary"])
    print("SCREENING DECISION REMAINS:", result["screening_decision_remains"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpe0-event", required=True)
    parser.add_argument("--cir0-event", required=True)
    parser.add_argument("--scale-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.cpe0_event, args.cir0_event, args.scale_json, args.data_root, args.output)


if __name__ == "__main__":
    main()
