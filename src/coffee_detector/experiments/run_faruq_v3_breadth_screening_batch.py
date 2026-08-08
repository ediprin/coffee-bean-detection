from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "configs/breadth_screening/faruq_v3_breadth_screening_batch_v1.yaml"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    branch: str
    sha: str
    runner: str
    chunk: int
    expected_summary: str
    common_gate: bool = True
    dependency: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest() -> dict:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    if payload.get("protocol") != "faruq-v3-breadth-screening-batch-v1":
        raise RuntimeError("Manifest breadth screening salah atau belum dibekukan")
    return payload


def _candidate_rows(payload: dict) -> list[Candidate]:
    rows = []
    for item in payload.get("candidates", []):
        rows.append(
            Candidate(
                candidate_id=str(item["id"]),
                branch=str(item["branch"]),
                sha=str(item["sha"]),
                runner=str(item["runner"]),
                chunk=int(item["chunk"]),
                expected_summary=str(item["expected_summary"]),
                common_gate=bool(item.get("common_gate", True)),
                dependency=item.get("dependency"),
            )
        )
    return rows


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    if "results" in source and isinstance(source["results"], dict):
        for name in ("candidate", "ACMC1", "D0FT", "D0", "S1", "PW1", "FT3", "AF12"):
            if name in source["results"] and isinstance(source["results"][name], dict):
                source = source["results"][name]
                break
    return {name: float(source[name]) for name in METRICS}


def _decision(candidate: dict[str, float], control: dict[str, float]) -> dict:
    delta = {name: candidate[name] - control[name] for name in METRICS}
    safeguards = {
        "macro_drop_no_more_than_1_point": delta["macro_map50_95"] >= -0.010,
        "bottom3_drop_no_more_than_2_points": delta["bottom3_class_map50_95"] >= -0.020,
        "worst_drop_no_more_than_2_points": delta["worst_class_map50_95"] >= -0.020,
    }
    signal = (
        delta["macro_map50_95"] >= 0.002
        or delta["bottom3_class_map50_95"] >= 0.005
        or delta["worst_class_map50_95"] >= 0.005
    )
    keep = bool(signal and all(safeguards.values()))
    return {
        "decision": "RETAIN" if keep else "REJECT",
        "delta_vs_D0FT": delta,
        "discovery_signal": bool(signal),
        **safeguards,
    }


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=check, text=True, capture_output=True)


def _runner_module(token: str) -> str:
    return f"coffee_detector.experiments.run_faruq_v3_{token}_screening"


def _candidate_command(candidate: Candidate, *, project_root: Path, data_root: Path, output_root: Path, device: str) -> list[str]:
    shared = output_root / "shared"
    common = [
        sys.executable,
        "-m",
        _runner_module(candidate.runner),
        "--data-root",
        str(data_root),
        "--grouped-summary",
        str(project_root / "evidence/faruq-grouped-development-v1/faruq_grouped_summary.json"),
        "--d0-checkpoint",
        str(project_root / "experiments/faruq-v3-yolo26n-baseline-v1/D0_seed42/weights/best.pt"),
        "--d0ft-report",
        str(project_root / "experiments/faruq-v3-acmc-optimization-control-v1/val_reports/D0FT_seed42_val.json"),
        "--acmc1-report",
        str(project_root / "experiments/faruq-v3-acmc-optimization-control-v1/val_reports/acmc1_optimization_control_seed42.json"),
        "--output-root",
        str(output_root / "candidates" / candidate.candidate_id),
        "--seed",
        "42",
        "--device",
        device,
        "--authorize-training",
    ]
    if candidate.dependency:
        dependency_summary = output_root / "candidates" / candidate.dependency / candidate.expected_summary
        if not dependency_summary.is_file():
            dependency_summary = output_root / "candidates" / candidate.dependency / "val_reports" / "drnet_seed42_screening.json"
        common.extend(["--drnet-summary", str(dependency_summary)])
    return common


def _find_summary(candidate: Candidate, candidate_root: Path) -> Path | None:
    exact = candidate_root / candidate.expected_summary
    if exact.is_file():
        return exact
    matches = list(candidate_root.rglob(Path(candidate.expected_summary).name))
    return matches[0] if matches else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "status",
        "decision",
        "macro_map50_95",
        "bottom3_class_map50_95",
        "worst_class_map50_95",
        "delta_macro_vs_D0FT",
        "delta_bottom3_vs_D0FT",
        "delta_worst_vs_D0FT",
        "branch",
        "sha",
        "summary",
        "log",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_batch(project_root: Path, data_root: Path, output_root: Path, *, chunk: int, device: str) -> dict:
    manifest = _load_manifest()
    candidates = [row for row in _candidate_rows(manifest) if row.chunk == chunk]
    if not candidates:
        raise ValueError(f"Chunk {chunk} tidak memiliki kandidat")
    if (data_root / "test").exists():
        raise RuntimeError("TEST LOCK: development root tidak boleh memiliki test split")

    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "master_state.json"
    results_path = output_root / "master_results.json"
    csv_path = output_root / "master_results.csv"
    state = _load_json(state_path) if state_path.is_file() else {"protocol": manifest["protocol"], "candidates": {}}
    results = _load_json(results_path) if results_path.is_file() else {"protocol": manifest["protocol"], "results": {}}
    d0ft = _metrics(_load_json(project_root / "experiments/faruq-v3-acmc-optimization-control-v1/val_reports/D0FT_seed42_val.json"))
    original_sha = _git("rev-parse", "HEAD").stdout.strip()

    try:
        for candidate in candidates:
            record = state["candidates"].get(candidate.candidate_id, {})
            if record.get("status") == "completed":
                print(f"SKIP {candidate.candidate_id}: completed", flush=True)
                continue
            if candidate.dependency:
                dep = state["candidates"].get(candidate.dependency, {})
                if dep.get("status") != "completed":
                    print(f"SKIP {candidate.candidate_id}: dependency {candidate.dependency} belum completed", flush=True)
                    continue

            state["candidates"][candidate.candidate_id] = {
                "status": "running",
                "started_at": _now(),
                "branch": candidate.branch,
                "sha": candidate.sha,
            }
            _write_json(state_path, state)
            _git("checkout", "--detach", candidate.sha)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            command = _candidate_command(
                candidate,
                project_root=project_root,
                data_root=data_root,
                output_root=output_root,
                device=device,
            )
            print(f"RUN {candidate.candidate_id} @ {candidate.sha[:12]}", flush=True)
            log_path = logs_root / f"{candidate.candidate_id}.log"
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n===== {_now()} RUN {candidate.candidate_id} @ {candidate.sha} =====\n")
                log_handle.write("COMMAND: " + " ".join(command) + "\n")
                log_handle.flush()
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )

            if completed.returncode != 0:
                error = f"Runner exit code {completed.returncode}; lihat {log_path}"
                state["candidates"][candidate.candidate_id] = {
                    **state["candidates"][candidate.candidate_id],
                    "status": "failed",
                    "finished_at": _now(),
                    "error": error,
                    "log": str(log_path),
                }
                _write_json(state_path, state)
                print(f"FAILED {candidate.candidate_id}: {error}", flush=True)
                continue

            candidate_root = output_root / "candidates" / candidate.candidate_id
            summary_path = _find_summary(candidate, candidate_root)
            if summary_path is None:
                error = "Runner selesai tetapi summary tidak ditemukan"
                state["candidates"][candidate.candidate_id] = {
                    **state["candidates"][candidate.candidate_id],
                    "status": "failed",
                    "finished_at": _now(),
                    "error": error,
                    "log": str(log_path),
                }
                _write_json(state_path, state)
                print(f"FAILED {candidate.candidate_id}: {error}", flush=True)
                continue

            payload = _load_json(summary_path)
            candidate_metrics = _metrics(payload)
            gate = _decision(candidate_metrics, d0ft) if candidate.common_gate else {"decision": "SEPARATE_PROTOCOL"}
            result = {
                "candidate_id": candidate.candidate_id,
                "branch": candidate.branch,
                "sha": candidate.sha,
                "summary": str(summary_path),
                "log": str(log_path),
                "metrics": candidate_metrics,
                "gate": gate,
                "finished_at": _now(),
            }
            results["results"][candidate.candidate_id] = result
            state["candidates"][candidate.candidate_id] = {
                **state["candidates"][candidate.candidate_id],
                "status": "completed",
                "finished_at": _now(),
                "summary": str(summary_path),
                "log": str(log_path),
                "decision": gate["decision"],
            }
            _write_json(results_path, results)
            _write_json(state_path, state)

            flat = []
            for candidate_id, item in results["results"].items():
                metrics = item["metrics"]
                gate_item = item["gate"]
                delta = gate_item.get("delta_vs_D0FT", {})
                flat.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "completed",
                        "decision": gate_item.get("decision", ""),
                        "macro_map50_95": metrics["macro_map50_95"],
                        "bottom3_class_map50_95": metrics["bottom3_class_map50_95"],
                        "worst_class_map50_95": metrics["worst_class_map50_95"],
                        "delta_macro_vs_D0FT": delta.get("macro_map50_95", ""),
                        "delta_bottom3_vs_D0FT": delta.get("bottom3_class_map50_95", ""),
                        "delta_worst_vs_D0FT": delta.get("worst_class_map50_95", ""),
                        "branch": item["branch"],
                        "sha": item["sha"],
                        "summary": item["summary"],
                        "log": item.get("log", ""),
                        "error": "",
                    }
                )
            for candidate_id, item in state["candidates"].items():
                if item.get("status") == "failed" and candidate_id not in results["results"]:
                    flat.append(
                        {
                            "candidate_id": candidate_id,
                            "status": "failed",
                            "decision": "TECHNICAL_FAILURE",
                            "branch": item.get("branch", ""),
                            "sha": item.get("sha", ""),
                            "log": item.get("log", ""),
                            "error": item.get("error", ""),
                        }
                    )
            _write_csv(csv_path, flat)
            print(f"DONE {candidate.candidate_id}: {gate['decision']}", flush=True)
    finally:
        _git("checkout", "--detach", original_sha, check=False)

    return {"state": str(state_path), "results": str(results_path), "csv": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 breadth screening batch controller")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    payload = run_batch(
        Path(args.project_root).expanduser().resolve(),
        Path(args.data_root).expanduser().resolve(),
        Path(args.output_root).expanduser().resolve(),
        chunk=args.chunk,
        device=args.device,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
