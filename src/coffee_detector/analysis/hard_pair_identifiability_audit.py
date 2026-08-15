from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from coffee_detector.dataset import discover_layout, parse_label

EVENT_PROTOCOL = "faruq-v3-validation-object-events-v1"
CONSENSUS_PROTOCOL = "faruq-v3-cross-model-hard-confusion-consensus-v1"
PROTOCOL = "faruq-v3-hard-pair-identifiability-audit-v1"
MAX_PER_CATEGORY = 8
TILE_SIZE = 220
PADDING_SCALE = 1.8


def _load_json(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    return json.loads(source.read_text(encoding="utf-8"))


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
    if not isinstance(payload.get("events"), dict) or not payload["events"]:
        raise RuntimeError(f"Event kosong: {path}")
    return payload


def _load_frozen_pairs(path: str | Path) -> tuple[list[str], dict[str, int]]:
    payload = _load_json(path)
    if payload.get("protocol") != CONSENSUS_PROTOCOL:
        raise RuntimeError("Consensus JSON tidak kompatibel")
    if int(payload.get("seed", -1)) != 42 or payload.get("evaluation_split") != "val":
        raise RuntimeError("Consensus bukan seed42 val")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError("Consensus menunjukkan akses test")
    rows = [r for r in payload.get("undirected_families", []) if r.get("frozen_consensus_hard_family")]
    if len(rows) != 17:
        raise RuntimeError(f"Frozen undirected family harus 17, got={len(rows)}")
    pairs = [str(r["family"]) for r in rows]
    support = {str(r["family"]): int(r.get("total_support", 0)) for r in rows}
    return pairs, support


def _pair_name(a: str, b: str) -> str:
    return " <-> ".join(sorted((str(a), str(b))))


def _is_pair_error(row: dict, family: str) -> bool:
    return bool(
        row.get("matched")
        and not row.get("correct")
        and row.get("gt_class_name") is not None
        and row.get("pred_class_name") is not None
        and _pair_name(row["gt_class_name"], row["pred_class_name"]) == family
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _read_target(data_root: Path, event_row: dict):
    layout = discover_layout(data_root)
    image_root, label_root = layout.splits["val"]
    image_path = image_root / event_row["image"]
    relative = Path(event_row["image"])
    label_path = (label_root / relative).with_suffix(".txt")
    annotations = parse_label(label_path, set(layout.names))
    target_index = int(event_row["target_index"])
    if target_index < 0 or target_index >= len(annotations):
        raise IndexError(f"target_index di luar label: {event_row['target_key']}")
    annotation = annotations[target_index]
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Gagal baca gambar: {image_path}")
    return image_path, image, annotation


def _context_crop(image: np.ndarray, annotation, scale: float = PADDING_SCALE) -> np.ndarray:
    h, w = image.shape[:2]
    cx = float(annotation.x_center) * w
    cy = float(annotation.y_center) * h
    bw = max(2.0, float(annotation.width) * w)
    bh = max(2.0, float(annotation.height) * h)
    side_w = bw * scale
    side_h = bh * scale
    x1 = max(0, int(math.floor(cx - side_w / 2)))
    y1 = max(0, int(math.floor(cy - side_h / 2)))
    x2 = min(w, int(math.ceil(cx + side_w / 2)))
    y2 = min(h, int(math.ceil(cy + side_h / 2)))
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = image.copy()
    return crop


def _fit_tile(crop: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = np.full((TILE_SIZE + 72, TILE_SIZE, 3), 245, dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = min(TILE_SIZE / max(w, 1), TILE_SIZE / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x = (TILE_SIZE - nw) // 2
    y = (TILE_SIZE - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    for i, line in enumerate(lines[:3]):
        text = str(line)
        if len(text) > 34:
            text = text[:31] + "..."
        cv2.putText(canvas, text, (5, TILE_SIZE + 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def _make_sheet(data_root: Path, cpe_events: dict, cir_events: dict, keys: list[str], output: Path, title: str) -> None:
    tiles = []
    for key in keys:
        cpe = cpe_events[key]
        cir = cir_events[key]
        _, image, ann = _read_target(data_root, cpe)
        crop = _context_crop(image, ann)
        lines = [
            f"GT: {cpe['gt_class_name']}",
            f"CPE0: {cpe.get('pred_class_name') or 'MISS'}",
            f"CIR0: {cir.get('pred_class_name') or 'MISS'}",
        ]
        tiles.append(_fit_tile(crop, lines))
    if not tiles:
        return
    cols = min(4, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    th, tw = tiles[0].shape[:2]
    header = 46
    sheet = np.full((header + rows * th, cols * tw, 3), 255, dtype=np.uint8)
    cv2.putText(sheet, title[:95], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[header + r*th:header+(r+1)*th, c*tw:(c+1)*tw] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def run(cpe0_event, cir0_event, consensus_json, data_root, output_root) -> dict:
    cpe = _load_event(cpe0_event, "CPE0")
    cir = _load_event(cir0_event, "CIR0")
    if set(cpe["events"]) != set(cir["events"]):
        raise RuntimeError("Target universe CPE0/CIR0 berbeda")
    data_root = Path(data_root).expanduser().resolve()
    layout = discover_layout(data_root)
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit menolak data root yang mengekspos test")
    pairs, consensus_support = _load_frozen_pairs(consensus_json)
    output_root = Path(output_root).expanduser().resolve()
    sheets = output_root / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    family_rows = []
    all_category_counts = Counter()
    for rank, family in enumerate(pairs, 1):
        left, right = family.split(" <-> ", 1)
        categories = defaultdict(list)
        gt_counts = Counter()
        for key, a in cpe["events"].items():
            b = cir["events"][key]
            if a.get("gt_class_name") not in (left, right):
                continue
            gt_counts[str(a.get("gt_class_name"))] += 1
            ea = _is_pair_error(a, family)
            eb = _is_pair_error(b, family)
            if ea and eb:
                categories["shared_pair_error"].append(key)
            elif ea and not eb:
                categories["cpe0_pair_only"].append(key)
            elif eb and not ea:
                categories["cir0_pair_only"].append(key)
            elif a.get("correct") and b.get("correct"):
                categories["both_correct"].append(key)
            elif a.get("matched") and b.get("matched"):
                categories["other_matched_state"].append(key)
            else:
                categories["contains_unmatched"].append(key)

        category_counts = {name: len(values) for name, values in categories.items()}
        all_category_counts.update(category_counts)
        pair_errors_cpe = category_counts.get("shared_pair_error", 0) + category_counts.get("cpe0_pair_only", 0)
        pair_errors_cir = category_counts.get("shared_pair_error", 0) + category_counts.get("cir0_pair_only", 0)
        denom = max(1, sum(gt_counts.values()))
        row = {
            "rank": rank,
            "family": family,
            "consensus_total_support_seed42_six_variants": consensus_support.get(family, 0),
            "gt_instances_in_pair": int(sum(gt_counts.values())),
            "gt_count_by_class": dict(gt_counts),
            "cpe0_pair_errors": pair_errors_cpe,
            "cir0_pair_errors": pair_errors_cir,
            "delta_cir0_minus_cpe0": pair_errors_cir - pair_errors_cpe,
            "cpe0_pair_error_rate_over_pair_gt": pair_errors_cpe / denom,
            "cir0_pair_error_rate_over_pair_gt": pair_errors_cir / denom,
            "categories": category_counts,
            "contact_sheets": {},
        }
        for category in ("shared_pair_error", "cpe0_pair_only", "cir0_pair_only", "both_correct"):
            keys = sorted(categories.get(category, []))[:MAX_PER_CATEGORY]
            if not keys:
                continue
            path = sheets / f"{rank:02d}_{_safe_name(family)}__{category}.jpg"
            _make_sheet(data_root, cpe["events"], cir["events"], keys, path, f"{family} | {category} | n={len(categories[category])}")
            row["contact_sheets"][category] = str(path)
        family_rows.append(row)

    family_rows.sort(key=lambda r: (r["consensus_total_support_seed42_six_variants"], r["cpe0_pair_errors"] + r["cir0_pair_errors"]), reverse=True)
    summary = {
        "protocol": PROTOCOL,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "frozen_pair_source": str(Path(consensus_json).expanduser().resolve()),
        "n_frozen_undirected_hard_pairs": len(pairs),
        "purpose": "visual and annotation identifiability diagnostic; not model selection and not evidence of label error without human review",
        "category_totals_across_pairs": dict(all_category_counts),
        "families": family_rows,
        "interpretation_guardrail": (
            "Contact sheets can reveal visually overlapping phenotypes, inconsistent-looking labels, or context dependence, "
            "but this automated audit cannot itself adjudicate ground-truth correctness. Human review is required."
        ),
        "screening_decision_remains": "STOP_CIRCLE_CPE",
    }
    destination = output_root / "hard_pair_identifiability_audit.json"
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", destination)
    print("FROZEN PAIRS:", len(pairs))
    print("CATEGORY TOTALS:", summary["category_totals_across_pairs"])
    print("SCREENING DECISION REMAINS:", summary["screening_decision_remains"])
    print("TOP PAIRS:")
    for row in family_rows[:10]:
        print(row["family"], "CPE0=", row["cpe0_pair_errors"], "CIR0=", row["cir0_pair_errors"], "delta=", row["delta_cir0_minus_cpe0"], "sheets=", len(row["contact_sheets"]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpe0-event", required=True)
    parser.add_argument("--cir0-event", required=True)
    parser.add_argument("--consensus-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(args.cpe0_event, args.cir0_event, args.consensus_json, args.data_root, args.output_root)


if __name__ == "__main__":
    main()
