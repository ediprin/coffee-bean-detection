from __future__ import annotations

import argparse
import json

from coffee_detector.af2_scaffold import run_af2_scaffold_static_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit AF2MTS1")
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_af2_scaffold_static_audit(
        args.af2_checkpoint, args.output, device=args.device
    )
    print(json.dumps(result, indent=2), flush=True)
    if result["decision"] != "PASS":
        raise SystemExit("STOP: static audit gagal; training tidak diizinkan")


if __name__ == "__main__":
    main()
