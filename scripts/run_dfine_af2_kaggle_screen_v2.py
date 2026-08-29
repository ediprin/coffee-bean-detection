from __future__ import annotations

"""Fail-closed launcher for the AF2 x D-FINE-N seed-42 screen.

This v2 wrapper fixes one infrastructure bug in the original YOLO->COCO adapter:
D-FINE custom datasets with remap_mscoco_category=False consume category_id directly,
so a 21-class detector requires target labels 0..20, not 1..21.

No scientific setting is changed. AF2, data split, initialization, schedule, seed,
metrics, promotion gates, and locked-test policy remain identical.
"""

import importlib
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_dfine_af2_transfer as prep  # noqa: E402
import preflight_dfine_af2_transfer as preflight  # noqa: E402


_ORIGINAL_CONVERT_SPLIT = prep._convert_split
_ORIGINAL_PREFLIGHT = preflight.run_preflight
NUM_CLASSES = 21
EXPECTED_IDS = list(range(NUM_CLASSES))


def _convert_split_zero_based(images_dir, labels_dir, names, output_json):
    """Run the frozen converter, then correct only COCO category indexing.

    The original converter wrote category IDs as 1..21. With D-FINE's
    remap_mscoco_category=False those values become training labels directly,
    making label 21 invalid for a 21-logit classifier. This wrapper converts
    both category declarations and annotation category_id values to 0..20.
    """

    result = _ORIGINAL_CONVERT_SPLIT(images_dir, labels_dir, names, output_json)
    path = Path(output_json).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))

    category_ids_before = sorted(int(item["id"]) for item in payload["categories"])
    if category_ids_before != list(range(1, NUM_CLASSES + 1)):
        raise RuntimeError(
            "Unexpected pre-hotfix COCO category IDs: "
            f"{category_ids_before}; expected 1..{NUM_CLASSES}"
        )

    for category in payload["categories"]:
        category["id"] = int(category["id"]) - 1
    for annotation in payload["annotations"]:
        annotation["category_id"] = int(annotation["category_id"]) - 1

    category_ids_after = sorted(int(item["id"]) for item in payload["categories"])
    annotation_ids = [int(item["category_id"]) for item in payload["annotations"]]
    if category_ids_after != EXPECTED_IDS:
        raise RuntimeError(f"Zero-based category contract failed: {category_ids_after}")
    if not annotation_ids or min(annotation_ids) < 0 or max(annotation_ids) >= NUM_CLASSES:
        raise RuntimeError(
            "Annotation category IDs outside detector range after hotfix: "
            f"min={min(annotation_ids) if annotation_ids else None}, "
            f"max={max(annotation_ids) if annotation_ids else None}"
        )

    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    result["sha256"] = prep.sha256(path)
    result["category_ids"] = category_ids_after
    result["annotation_category_id_min"] = min(annotation_ids)
    result["annotation_category_id_max"] = max(annotation_ids)
    result["category_indexing"] = "zero_based_for_dfine_remap_false"
    return result


def _validate_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    category_ids = sorted(int(item["id"]) for item in payload.get("categories", []))
    annotation_ids = [int(item["category_id"]) for item in payload.get("annotations", [])]
    exact_categories = category_ids == EXPECTED_IDS
    annotations_in_range = bool(annotation_ids) and min(annotation_ids) >= 0 and max(annotation_ids) < NUM_CLASSES
    return {
        "category_ids": category_ids,
        "annotation_category_id_min": min(annotation_ids) if annotation_ids else None,
        "annotation_category_id_max": max(annotation_ids) if annotation_ids else None,
        "category_ids_exact_zero_based": exact_categories,
        "annotation_category_ids_in_range": annotations_in_range,
        "pass": exact_categories and annotations_in_range,
    }


def _strict_preflight(**kwargs):
    prep_report = json.loads(
        Path(kwargs["preparation_report"]).expanduser().resolve().read_text(encoding="utf-8")
    )
    split_contract = {}
    for split_name in ("train", "val"):
        manifest = Path(prep_report["dataset"]["splits"][split_name]["path"])
        split_contract[split_name] = _validate_manifest(manifest)
        if not split_contract[split_name]["pass"]:
            raise RuntimeError(
                f"D-FINE zero-based label contract failed for {split_name}: "
                f"{split_contract[split_name]}"
            )

    result = _ORIGINAL_PREFLIGHT(**kwargs)
    result["label_contract"] = split_contract
    result["gates"]["train_category_ids_exact_zero_based"] = split_contract["train"][
        "category_ids_exact_zero_based"
    ]
    result["gates"]["train_annotation_category_ids_in_range"] = split_contract["train"][
        "annotation_category_ids_in_range"
    ]
    result["gates"]["val_category_ids_exact_zero_based"] = split_contract["val"][
        "category_ids_exact_zero_based"
    ]
    result["gates"]["val_annotation_category_ids_in_range"] = split_contract["val"][
        "annotation_category_ids_in_range"
    ]
    result["decision"] = "PASS" if all(result["gates"].values()) else "FAIL"
    result["training_authorized"] = result["decision"] == "PASS"
    Path(kwargs["output"]).expanduser().resolve().write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    prep._convert_split = _convert_split_zero_based
    preflight.run_preflight = _strict_preflight

    # Import only after monkeypatching so the original runner binds the strict
    # prepare/preflight functions for this process.
    if "run_dfine_af2_kaggle_screen" in sys.modules:
        del sys.modules["run_dfine_af2_kaggle_screen"]
    runner = importlib.import_module("run_dfine_af2_kaggle_screen")
    runner.main()


if __name__ == "__main__":
    main()
