"""IGEM-only static authorization for the AF2FS frozen-parent confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audit import run_af2_parent_residual_static_audit


def run_af2_igem_parent_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 64,
) -> dict[str, Any]:
    """Run the established audit machinery but authorize only the IGEM pair.

    The legacy audit also materializes SAF records. They are deliberately not
    part of this confirmation decision: a SAF-specific gate cannot block or
    authorize the IGEM hypothesis.
    """

    destination = Path(output).expanduser().resolve()
    raw_path = destination.with_name(destination.stem + ".raw_all_families.json")
    raw = run_af2_parent_residual_static_audit(
        af2_checkpoint,
        raw_path,
        device=device,
        image_size=image_size,
    )

    required_gate_names = [
        "source_is_af2",
        "igem_same_model_yaml",
        "igem_same_af2_config",
        "igem_same_training_schedule",
        "igem_same_parameter_count",
        "igem_same_trainable_count",
        "igem_same_state_schema",
        "igem_only_conditioning_differs",
        "igem_control_receives_zero_information",
        "igem_candidate_receives_features",
        "AF2IGEM0_initial_identity",
        "AF2IGEM0_boxes_preserved",
        "AF2IGEM0_finite_gradients",
        "AF2IGEM0_frozen_parent",
        "AF2IGEM0_zero_information_identity",
        "AF2IGEM1_initial_identity",
        "AF2IGEM1_boxes_preserved",
        "AF2IGEM1_finite_gradients",
        "AF2IGEM1_frozen_parent",
        "AF2IGEM1_active_scores",
    ]
    missing = [name for name in required_gate_names if name not in raw["gates"]]
    if missing:
        raise RuntimeError(f"Audit IGEM kehilangan gate: {missing}")

    gates = {name: bool(raw["gates"][name]) for name in required_gate_names}
    gates["test_accessed"] = bool(raw["gates"].get("test_accessed", False))
    decision = "PASS" if all(gates[name] for name in required_gate_names) and not gates["test_accessed"] else "FAIL"

    result = {
        "format": "coffee_detector.af2_parent_residual.igem_static_audit.v1",
        "decision": decision,
        "checkpoint": raw["checkpoint"],
        "checkpoint_sha256": raw["checkpoint_sha256"],
        "records": {
            "AF2IGEM0": raw["records"]["AF2IGEM0"],
            "AF2IGEM1": raw["records"]["AF2IGEM1"],
        },
        "gates": gates,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
        "raw_audit": str(raw_path),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
