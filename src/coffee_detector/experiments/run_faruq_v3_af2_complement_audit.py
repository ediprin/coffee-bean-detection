from __future__ import annotations

import argparse
import json

from coffee_detector.af2_complement.audit import run_af2_complement_static_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit AF2 complementary mechanisms")
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_af2_complement_static_audit(
        args.af2_checkpoint, args.output, device=args.device
    )
    print("PARAMETERS:", {
        arm: {"total": row["parameters"], "added": row["added_parameters"]}
        for arm, row in result["arms"].items()
    })
    print("GATES:", result["gates"])
    print("DECISION:", result["decision"])
    print("SAVED:", args.output)
    if result["decision"] != "PASS":
        raise AssertionError("STOP: static audit gagal; jangan training")
    print(json.dumps({"training_authorized": True, "test_authorized": False}))


if __name__ == "__main__":
    main()
