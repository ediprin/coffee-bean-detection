from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def run_faruq_v3_af2_ffa_decision(
    output_root: str | Path, *, seed: int = 42
) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    results = {
        arm: _read(output_root / "val_reports" / f"{arm}_seed{seed}_result.json")
        for arm in ("AF2FFA0", "AF2FFA1")
    }
    if any(item.get("test_images_accessed") is not False for item in results.values()):
        raise RuntimeError("Salah satu arm mengakses test")
    values = {arm: _metrics(item) for arm, item in results.items()}
    deltas = {
        metric: values["AF2FFA1"][metric] - values["AF2FFA0"][metric]
        for metric in METRICS
    }
    criteria = {
        "macro_gain_at_least_0_5_point": deltas["macro_map50_95"] >= 0.005,
        "bottom3_not_lower": deltas["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point": deltas["worst_class_map50_95"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.af2_ffa.seed42_decision.v1",
        "seed": seed,
        "values": values,
        "candidate_minus_control": deltas,
        "criteria": criteria,
        "decision": decision,
        "next": (
            "AUTHORIZE_PAIRED_THREE_SEED_CONFIRMATION"
            if decision == "PASS"
            else "STOP_WITHOUT_TEST_OR_EXTRA_SEEDS"
        ),
        "test_opened": False,
    }
    destination = output_root / "val_reports" / f"af2_ffa_seed{seed}_decision.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2-FFA screening")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_faruq_v3_af2_ffa_decision(args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()
