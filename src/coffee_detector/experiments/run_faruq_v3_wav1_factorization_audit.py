from __future__ import annotations

import argparse
import json

from coffee_detector.wav1_factorization.audit import run_static_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit for WAV1 mechanism factorization")
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run_static_audit(args.d0_checkpoint, args.output)
    print(json.dumps(payload, indent=2), flush=True)
    if payload["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
