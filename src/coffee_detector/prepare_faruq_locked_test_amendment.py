"""Freeze the support-qualified v2 amendment before any test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .dataset import write_json


V2_MINIMUM_IMAGES = 100
V2_MINIMUM_INSTANCES_PER_CLASS = 5
V2_MINIMUM_PARENTS_PER_CLASS = 4
SAFE_V1_GATES = (
    "zero_development_parent_overlap",
    "zero_development_hash_overlap",
    "one_image_per_test_parent",
    "minimum_independent_images",
    "all_21_classes_present",
    "zero_quarantined_selected_images",
)
SUPPORT_V1_GATES = (
    "minimum_instances_per_class",
    "minimum_parents_per_class",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_faruq_locked_test_amendment(
    eligibility_summary: str | Path,
    output: str | Path,
) -> dict:
    eligibility_path = Path(eligibility_summary).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not eligibility_path.is_file():
        raise FileNotFoundError(eligibility_path)
    eligibility_hash = _sha256(eligibility_path)
    if output.is_file():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if (
            cached.get("status") == "complete"
            and cached.get("source_v1_eligibility_sha256") == eligibility_hash
        ):
            return cached
        raise RuntimeError("Cache amendemen tidak cocok dengan audit eligibility v1")
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    gates = eligibility.get("gates", {})
    if (
        eligibility.get("format")
        != "coffee_detector.faruq_locked_test_eligibility.v1"
        or eligibility.get("decision") != "FAIL"
        or eligibility.get("inference_executed") is not False
        or eligibility.get("training_executed") is not False
    ):
        raise RuntimeError("Amendemen v2 hanya menerima audit v1 FAIL pra-inference")
    failed = {name for name, passed in gates.items() if not passed}
    if not failed or not failed <= set(SUPPORT_V1_GATES):
        raise RuntimeError(f"FAIL v1 bukan hanya masalah support: {sorted(failed)}")

    instances = {name: int(value) for name, value in eligibility["instances_by_class"].items()}
    parents = {name: int(value) for name, value in eligibility["parents_by_class"].items()}
    v2_gates = {
        **{name: bool(gates.get(name)) for name in SAFE_V1_GATES},
        "minimum_100_independent_images": int(eligibility["materialized_images"])
        >= V2_MINIMUM_IMAGES,
        "minimum_5_instances_per_class": min(instances.values(), default=0)
        >= V2_MINIMUM_INSTANCES_PER_CLASS,
        "minimum_4_parents_per_class": min(parents.values(), default=0)
        >= V2_MINIMUM_PARENTS_PER_CLASS,
    }
    decision = "PASS" if all(v2_gates.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.faruq_locked_test_amendment.v2",
        "status": "complete",
        "decision": decision,
        "source_v1_eligibility": str(eligibility_path),
        "source_v1_eligibility_sha256": eligibility_hash,
        "source_v1_decision": "FAIL",
        "amendment_basis": "support_only_before_model_inference",
        "test_images_accessed": True,
        "model_inference_executed": False,
        "training_executed": False,
        "materialized_images": int(eligibility["materialized_images"]),
        "minimum_instances_observed": min(instances.values(), default=0),
        "minimum_parents_observed": min(parents.values(), default=0),
        "primary_endpoint": "paired_three_seed_macro_map50_95_delta",
        "uncertainty": "paired_parent_bootstrap",
        "secondary_descriptive_only": [
            "bottom3_class_map50_95",
            "worst_class_map50_95",
        ],
        "thresholds": {
            "minimum_independent_images": V2_MINIMUM_IMAGES,
            "minimum_instances_per_class": V2_MINIMUM_INSTANCES_PER_CLASS,
            "minimum_parents_per_class": V2_MINIMUM_PARENTS_PER_CLASS,
        },
        "gates": v2_gates,
        "next_action": (
            "AUTHORIZE_V2_FROZEN_ACMC_TEST_INFERENCE"
            if decision == "PASS"
            else "STOP_TEST_INFERENCE_USE_GROUPED_CV_OR_EXTERNAL_TEST"
        ),
        "further_tuning_authorized": False,
    }
    write_json(payload, output)
    payload["summary"] = str(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Faruq locked-test support amendment v2")
    parser.add_argument("--eligibility-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_faruq_locked_test_amendment(args.eligibility_summary, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
