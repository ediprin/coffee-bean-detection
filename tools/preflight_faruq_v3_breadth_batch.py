from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


KNOWN_OPTIONS = {
    "--data-root",
    "--grouped-summary",
    "--control-summary",
    "--d0-checkpoint",
    "--d0ft-report",
    "--acmc1-report",
    "--output-root",
    "--seed",
    "--device",
    "--authorize-training",
    "--text-embeddings",
    "--ontology",
    "--ontology-path",
    "--ontology-config",
}


def load_controller(path: Path):
    spec = importlib.util.spec_from_file_location("breadth_controller", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight all frozen Faruq-v3 breadth runners")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    manifest_path = Path(args.manifest).resolve()
    controller_path = repo / "tools/run_faruq_v3_breadth_batch.py"
    controller = load_controller(controller_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [
        item for item in manifest["candidates"]
        if item.get("enabled") and item.get("group") == "detector"
    ]
    controller_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    env = os.environ.copy()
    failures = []
    rows = []

    try:
        for item in selected:
            fetched = subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", item["sha"]],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if fetched.returncode != 0:
                failures.append((item["id"], "git-fetch", fetched.stdout))
                continue
            checked = subprocess.run(
                ["git", "checkout", "--detach", item["sha"]],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if checked.returncode != 0:
                failures.append((item["id"], "git-checkout", checked.stdout))
                continue
            candidate_env = env.copy()
            candidate_env["PYTHONPATH"] = str(repo / "src") + os.pathsep + candidate_env.get("PYTHONPATH", "")
            try:
                runner = controller.discover_runner(repo, item["runner_token"])
                module = controller.module_name(repo, runner)
                help_output = controller.help_text(repo, module, candidate_env)
                required = controller.find_required_options(help_output)
                unknown = sorted(option for option in required if option not in KNOWN_OPTIONS and option != "--help")
                row = {
                    "id": item["id"],
                    "family": item["family"],
                    "sha": item["sha"],
                    "runner": module,
                    "required_options": sorted(required),
                    "unknown_required_options": unknown,
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
                if unknown:
                    failures.append((item["id"], "unknown-required-options", unknown))
            except Exception as exc:
                failures.append((item["id"], "runner-preflight", repr(exc)))
    finally:
        subprocess.run(
            ["git", "checkout", "--detach", controller_head],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    report = {"protocol": manifest["protocol"], "count": len(rows), "rows": rows, "failures": failures}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if len(rows) != len(selected):
        raise SystemExit(f"Only {len(rows)}/{len(selected)} runners passed basic discovery")
    if failures:
        raise SystemExit(f"Preflight failures: {failures}")


if __name__ == "__main__":
    main()
