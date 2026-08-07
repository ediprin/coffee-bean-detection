from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


METRIC_KEYS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
DEFAULT_MANIFEST = Path("configs/breadth_screening/faruq_v3_batch_v1.json")


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print("$", " ".join(shlex.quote(part) for part in cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )


def git(repo: Path, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return run(["git", *args], cwd=repo, capture=capture)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def discover_runner(repo: Path, token: str) -> Path:
    root = repo / "src/coffee_detector/experiments"
    candidates = sorted(root.glob("run_faruq_v3_*screening.py"))
    if not candidates:
        raise FileNotFoundError(f"Tidak ada runner screening di {root}")
    wanted = normalize_token(token)
    exactish = [p for p in candidates if wanted in normalize_token(p.stem)]
    if len(exactish) == 1:
        return exactish[0]
    if len(exactish) > 1:
        # Prefer the most specific/shortest candidate after token filtering.
        exactish.sort(key=lambda p: (len(p.name), p.name))
        return exactish[0]

    pieces = [piece for piece in wanted.split("_") if len(piece) >= 3]
    scored: list[tuple[int, int, Path]] = []
    for path in candidates:
        stem = normalize_token(path.stem)
        score = sum(piece in stem for piece in pieces)
        if score:
            scored.append((-score, len(path.name), path))
    if scored:
        scored.sort()
        best_score = scored[0][0]
        best = [entry for entry in scored if entry[0] == best_score]
        if len(best) == 1:
            return best[0][2]

    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(
        f"Runner ambigu untuk token={token!r}: {[p.name for p in candidates]}"
    )


def module_name(repo: Path, runner: Path) -> str:
    relative = runner.relative_to(repo / "src").with_suffix("")
    return ".".join(relative.parts)


def help_text(repo: Path, module: str, env: dict[str, str]) -> str:
    result = run([sys.executable, "-m", module, "--help"], cwd=repo, env=env, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"Runner --help gagal ({module}):\n{result.stdout}")
    return result.stdout or ""


def option_present(help_output: str, option: str) -> bool:
    return option in help_output


def find_required_options(help_output: str) -> set[str]:
    """Best-effort parse argparse usage: required options appear outside [] in usage."""
    usage_lines = []
    collecting = False
    for line in help_output.splitlines():
        if line.startswith("usage:"):
            collecting = True
        if collecting:
            usage_lines.append(line.strip())
            if line.strip().endswith("[-h]"):
                break
        elif usage_lines:
            break
    usage = " ".join(usage_lines)
    options = set(re.findall(r"(?<!\[)(--[a-zA-Z0-9_-]+)(?:\s|=)", usage))
    return options


def prepare_ftif_cache(repo: Path, shared_root: Path, env: dict[str, str]) -> Path:
    output = shared_root / "sni21_openai_clip_vit_b32_text_embeddings.pt"
    if output.is_file():
        return output
    prompt = repo / "configs/ftif/sni21_prompts.yaml"
    if not prompt.is_file():
        raise FileNotFoundError(prompt)
    script = (
        "from coffee_detector.ftif import generate_clip_text_embeddings; "
        f"generate_clip_text_embeddings(r'{prompt}', r'{output}', "
        "model_name='ViT-B-32', pretrained='openai', device='cuda')"
    )
    result = run([sys.executable, "-c", script], cwd=repo, env=env, capture=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Gagal membuat FTIF text cache. Pastikan open_clip_torch terpasang dan CLIP weights dapat diunduh.\n"
            + (result.stdout or "")
        )
    if not output.is_file():
        raise FileNotFoundError(output)
    return output


def build_runner_command(
    *,
    repo: Path,
    module: str,
    help_output: str,
    data_root: Path,
    grouped_summary: Path,
    control_summary: Path,
    d0_checkpoint: Path,
    d0ft_report: Path,
    output_root: Path,
    shared_root: Path,
    seed: int,
    device: str,
    pre_hook: str | None,
    env: dict[str, str],
) -> list[str]:
    providers: dict[str, str] = {
        "--data-root": str(data_root),
        "--grouped-summary": str(grouped_summary),
        "--control-summary": str(control_summary),
        "--d0-checkpoint": str(d0_checkpoint),
        "--d0ft-report": str(d0ft_report),
        "--acmc1-report": str(control_summary),
        "--output-root": str(output_root),
        "--seed": str(seed),
        "--device": str(device),
    }
    ontology = repo / "configs/sni21/structured_ontology_v1.yaml"
    for name in ("--ontology", "--ontology-path", "--ontology-config"):
        providers[name] = str(ontology)

    if pre_hook == "ftif_text":
        providers["--text-embeddings"] = str(prepare_ftif_cache(repo, shared_root, env))

    cmd = [sys.executable, "-u", "-m", module]
    for option, value in providers.items():
        if option_present(help_output, option):
            cmd += [option, value]
    if option_present(help_output, "--authorize-training"):
        cmd.append("--authorize-training")

    required = find_required_options(help_output)
    missing = sorted(
        option for option in required
        if option not in providers and option != "--authorize-training" and option != "--help"
    )
    if missing:
        raise RuntimeError(f"Runner membutuhkan argumen yang belum dipetakan: {missing}")
    return cmd


def is_metric_dict(value: Any) -> bool:
    return isinstance(value, dict) and all(key in value for key in METRIC_KEYS)


def metric_row(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"arm": label, **{key: float(metrics[key]) for key in METRIC_KEYS}}


def extract_candidate_rows(payload: dict[str, Any], default_label: str) -> list[dict[str, Any]]:
    candidate = payload.get("candidate")
    if is_metric_dict(candidate):
        return [metric_row(default_label, candidate)]
    if isinstance(candidate, dict):
        rows = [metric_row(str(name), value) for name, value in candidate.items() if is_metric_dict(value)]
        if rows:
            return rows

    results = payload.get("results")
    if isinstance(results, dict):
        rows = [
            metric_row(str(name), value)
            for name, value in results.items()
            if is_metric_dict(value) and str(name).upper() not in {"D0", "D0FT", "ACMC1"}
        ]
        if rows:
            return rows

    # Conservative fallback: recurse while explicitly excluding controls.
    rows: list[dict[str, Any]] = []
    def walk(value: Any, path: tuple[str, ...]) -> None:
        if is_metric_dict(value):
            lowered = {part.lower() for part in path}
            if not lowered.intersection({"controls", "control", "d0", "d0ft", "acmc1"}):
                rows.append(metric_row("/".join(path) or default_label, value))
            return
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, path + (str(key),))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + (str(index),))
    walk(payload, tuple())
    return rows


def locate_summary(output_root: Path) -> Path:
    candidates = sorted(output_root.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Tidak ada JSON report di {output_root}")
    preferred = [
        path for path in candidates
        if "screen" in path.name.lower() or "summary" in path.name.lower()
    ]
    for path in preferred + candidates:
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and (
            "candidate" in payload or "results" in payload or is_metric_dict(payload.get("metrics"))
        ):
            return path
    return candidates[0]


def canonical_decision(row: dict[str, Any], d0ft: dict[str, float], gate: dict[str, Any]) -> dict[str, Any]:
    delta = {key: float(row[key]) - float(d0ft[key]) for key in METRIC_KEYS}
    signal = gate["discovery_signal_any"]
    criteria = {
        "macro_retention": delta["macro_map50_95"] >= float(gate["macro_vs_d0ft_min"]),
        "bottom3_retention": delta["bottom3_class_map50_95"] >= float(gate["bottom3_vs_d0ft_min"]),
        "worst_retention": delta["worst_class_map50_95"] >= float(gate["worst_vs_d0ft_min"]),
        "discovery_signal": (
            delta["macro_map50_95"] >= float(signal["macro_vs_d0ft_min"])
            or delta["bottom3_class_map50_95"] >= float(signal["bottom3_vs_d0ft_min"])
            or delta["worst_class_map50_95"] >= float(signal["worst_vs_d0ft_min"])
        ),
    }
    return {
        "delta_vs_D0FT": delta,
        "criteria": criteria,
        "decision": "RETAIN" if all(criteria.values()) else "REJECT",
    }


def extract_control_metrics(control_summary: Path) -> dict[str, dict[str, float]]:
    payload = read_json(control_summary)
    results = payload.get("results", {})
    controls: dict[str, dict[str, float]] = {}
    for name in ("D0", "D0FT", "ACMC1"):
        value = results.get(name)
        if value is None:
            continue
        source = value.get("metrics", value) if isinstance(value, dict) else value
        if is_metric_dict(source):
            controls[name] = {key: float(source[key]) for key in METRIC_KEYS}
    if "D0FT" not in controls:
        raise RuntimeError("Control summary tidak memiliki D0FT metrics")
    return controls


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id", "family", "arm", "decision",
        "macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95",
        "delta_macro_vs_D0FT", "delta_bottom3_vs_D0FT", "delta_worst_vs_D0FT",
        "branch", "sha", "summary_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Faruq-v3 breadth screening orchestrator")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--repo", required=True, help="Working git clone that may be detached/checkouted")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--d0ft-report", required=True)
    parser.add_argument("--batch-root", required=True, help="Persistent Drive output root")
    parser.add_argument("--device", default="0")
    parser.add_argument("--only", nargs="*", default=[], help="Candidate IDs; empty means all enabled detector candidates")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    grouped_summary = Path(args.grouped_summary).expanduser().resolve()
    control_summary = Path(args.control_summary).expanduser().resolve()
    d0_checkpoint = Path(args.d0_checkpoint).expanduser().resolve()
    d0ft_report = Path(args.d0ft_report).expanduser().resolve()
    batch_root = Path(args.batch_root).expanduser().resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    manifest = read_json(manifest_path)

    if manifest.get("seed") != 42 or manifest.get("evaluation_split") != "val":
        raise RuntimeError("Manifest bukan frozen seed42/val batch")
    if (data_root / "test").exists():
        raise RuntimeError("STOP: development root mengekspos split test")
    for required in (grouped_summary, control_summary, d0_checkpoint, d0ft_report):
        if not required.is_file():
            raise FileNotFoundError(required)

    controls = extract_control_metrics(control_summary)
    gate = manifest["canonical_gate"]
    batch_root.mkdir(parents=True, exist_ok=True)
    shared_root = batch_root / "shared"
    shared_root.mkdir(parents=True, exist_ok=True)
    state_path = batch_root / "master_state.json"
    csv_path = batch_root / "master_results.csv"
    json_path = batch_root / "master_results.json"

    controller_head = git(repo, "rev-parse", "HEAD", capture=True).stdout.strip()
    if not controller_head:
        raise RuntimeError("Tidak dapat membaca controller HEAD")

    selected = [
        item for item in manifest["candidates"]
        if item.get("enabled") and item.get("group") == "detector"
    ]
    if args.only:
        wanted = set(args.only)
        selected = [item for item in selected if item["id"] in wanted]
        missing_ids = wanted - {item["id"] for item in selected}
        if missing_ids:
            raise ValueError(f"ID --only tidak ditemukan/disabled: {sorted(missing_ids)}")
    if args.skip:
        skipped = set(args.skip)
        selected = [item for item in selected if item["id"] not in skipped]
    if args.max_candidates is not None:
        selected = selected[: args.max_candidates]

    state = read_json(state_path) if state_path.is_file() else {
        "protocol": manifest["protocol"],
        "controller_head": controller_head,
        "seed": 42,
        "test_opened": False,
        "test_images_accessed": False,
        "candidates": {},
    }
    master_rows: list[dict[str, Any]] = []
    if json_path.is_file():
        previous = read_json(json_path)
        master_rows = list(previous.get("rows", []))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        for index, item in enumerate(selected, start=1):
            candidate_id = item["id"]
            prior = state["candidates"].get(candidate_id, {})
            if prior.get("status") == "completed" and not args.force_rerun:
                print(f"[{index}/{len(selected)}] {candidate_id}: already completed, skip", flush=True)
                continue

            print("\n" + "=" * 90)
            print(f"[{index}/{len(selected)}] {candidate_id} | {item['family']} | {item['sha']}")
            print("=" * 90, flush=True)
            state["candidates"][candidate_id] = {
                "status": "running",
                "branch": item["branch"],
                "sha": item["sha"],
                "started_at_unix": time.time(),
            }
            write_json(state_path, state)

            try:
                result = git(repo, "fetch", "--depth", "1", "origin", item["sha"], capture=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stdout or "git fetch gagal")
                result = git(repo, "checkout", "--detach", item["sha"], capture=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stdout or "git checkout gagal")
                actual = git(repo, "rev-parse", "HEAD", capture=True).stdout.strip()
                if actual != item["sha"]:
                    raise RuntimeError(f"SHA mismatch: {actual} != {item['sha']}")

                candidate_env = env.copy()
                candidate_env["PYTHONPATH"] = str(repo / "src") + os.pathsep + candidate_env.get("PYTHONPATH", "")
                runner = discover_runner(repo, item["runner_token"])
                module = module_name(repo, runner)
                help_output = help_text(repo, module, candidate_env)
                output_root = batch_root / "candidates" / candidate_id
                output_root.mkdir(parents=True, exist_ok=True)
                cmd = build_runner_command(
                    repo=repo,
                    module=module,
                    help_output=help_output,
                    data_root=data_root,
                    grouped_summary=grouped_summary,
                    control_summary=control_summary,
                    d0_checkpoint=d0_checkpoint,
                    d0ft_report=d0ft_report,
                    output_root=output_root,
                    shared_root=shared_root,
                    seed=42,
                    device=args.device,
                    pre_hook=item.get("pre_hook"),
                    env=candidate_env,
                )
                command_text = " ".join(shlex.quote(part) for part in cmd)
                if "test" in command_text.lower() and "pytest" not in command_text.lower():
                    raise RuntimeError("STOP: constructed command contains forbidden test token")
                result = run(cmd, cwd=repo, env=candidate_env, capture=False)
                if result.returncode != 0:
                    raise RuntimeError(f"Runner exit code {result.returncode}")

                summary = locate_summary(output_root)
                payload = read_json(summary)
                rows = extract_candidate_rows(payload, candidate_id)
                if not rows:
                    raise RuntimeError(f"Tidak dapat mengekstrak candidate metrics dari {summary}")

                master_rows = [row for row in master_rows if row.get("candidate_id") != candidate_id]
                standardized_rows = []
                for row in rows:
                    decision = canonical_decision(row, controls["D0FT"], gate)
                    standardized = {
                        "candidate_id": candidate_id,
                        "family": item["family"],
                        "arm": row["arm"],
                        **{key: row[key] for key in METRIC_KEYS},
                        "decision": decision["decision"],
                        "delta_macro_vs_D0FT": decision["delta_vs_D0FT"]["macro_map50_95"],
                        "delta_bottom3_vs_D0FT": decision["delta_vs_D0FT"]["bottom3_class_map50_95"],
                        "delta_worst_vs_D0FT": decision["delta_vs_D0FT"]["worst_class_map50_95"],
                        "criteria": decision["criteria"],
                        "branch": item["branch"],
                        "sha": item["sha"],
                        "runner": module,
                        "summary_path": str(summary),
                    }
                    standardized_rows.append(standardized)
                    master_rows.append(standardized)

                state["candidates"][candidate_id] = {
                    "status": "completed",
                    "branch": item["branch"],
                    "sha": item["sha"],
                    "runner": module,
                    "summary_path": str(summary),
                    "rows": standardized_rows,
                    "finished_at_unix": time.time(),
                }
                write_json(state_path, state)
                write_json(json_path, {
                    "protocol": manifest["protocol"],
                    "canonical_gate": gate,
                    "controls": controls,
                    "test_opened": False,
                    "test_images_accessed": False,
                    "rows": master_rows,
                })
                save_csv(csv_path, master_rows)
                print(f"{candidate_id}: completed -> {summary}", flush=True)
                for row in standardized_rows:
                    print(
                        f"  {row['arm']}: {row['decision']} | "
                        f"macro={row['macro_map50_95']:.4f} "
                        f"bottom3={row['bottom3_class_map50_95']:.4f} "
                        f"worst={row['worst_class_map50_95']:.4f}",
                        flush=True,
                    )
            except Exception as exc:
                state["candidates"][candidate_id] = {
                    **state["candidates"].get(candidate_id, {}),
                    "status": "failed",
                    "error": repr(exc),
                    "finished_at_unix": time.time(),
                }
                write_json(state_path, state)
                print(f"{candidate_id}: FAILED -> {exc!r}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    raise
    finally:
        # Restore the controller commit if possible so the notebook remains inspectable.
        git(repo, "checkout", "--detach", controller_head, capture=True)

    write_json(json_path, {
        "protocol": manifest["protocol"],
        "canonical_gate": gate,
        "controls": controls,
        "test_opened": False,
        "test_images_accessed": False,
        "rows": master_rows,
    })
    save_csv(csv_path, master_rows)
    retained = [row for row in master_rows if row.get("decision") == "RETAIN"]
    retained.sort(
        key=lambda row: (
            row["delta_macro_vs_D0FT"],
            row["delta_bottom3_vs_D0FT"],
            row["delta_worst_vs_D0FT"],
        ),
        reverse=True,
    )
    print("\nBATCH COMPLETE")
    print(f"rows={len(master_rows)} retained={len(retained)}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    for row in retained[:10]:
        print(
            f"RETAIN {row['candidate_id']}/{row['arm']} "
            f"dMacro={row['delta_macro_vs_D0FT']:+.4f} "
            f"dBottom3={row['delta_bottom3_vs_D0FT']:+.4f} "
            f"dWorst={row['delta_worst_vs_D0FT']:+.4f}"
        )


if __name__ == "__main__":
    main()
