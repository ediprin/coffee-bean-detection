from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path


_BASE_PATH = Path(__file__).resolve().with_name("run_faruq_v3_breadth_batch.py")
_SPEC = importlib.util.spec_from_file_location("faruq_v3_breadth_batch_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Tidak dapat memuat controller dasar: {_BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

_ORIGINAL_RUN = base.run
_ORIGINAL_BUILD_RUNNER_COMMAND = base.build_runner_command


def _candidate_log_path(cmd: list[str]) -> Path | None:
    """Resolve a persistent candidate log only for actual screening runner calls."""
    if "--output-root" not in cmd or "--authorize-training" not in cmd:
        return None
    try:
        output_root = Path(cmd[cmd.index("--output-root") + 1]).expanduser().resolve()
    except (IndexError, ValueError):
        return None
    return output_root / "runner.log"


def run_with_tee(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Preserve controller semantics while teeing candidate stdout/stderr to Drive."""
    log_path = None if capture else _candidate_log_path(cmd)
    if log_path is None:
        return _ORIGINAL_RUN(cmd, cwd=cwd, env=env, capture=capture)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(shlex.quote(part) for part in cmd), flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
        returncode = process.wait()

    return subprocess.CompletedProcess(cmd, returncode, stdout=None, stderr=None)


def build_runner_command_with_dependencies(**kwargs):
    """Wire branch-local predecessor reports without weakening required-arg checks."""
    help_output = kwargs["help_output"]
    needs_drnet = "--drnet-summary" in help_output
    if needs_drnet:
        # Hide this single dependency from the base required-option parser; we
        # resolve and append it explicitly below from the sibling DRNET output.
        patched = help_output.replace(
            "--drnet-summary DRNET_SUMMARY",
            "[--drnet-summary DRNET_SUMMARY]",
        )
        kwargs = dict(kwargs)
        kwargs["help_output"] = patched
    cmd = _ORIGINAL_BUILD_RUNNER_COMMAND(**kwargs)
    if needs_drnet:
        output_root = Path(kwargs["output_root"]).expanduser().resolve()
        reports = output_root.parent / "DRNET" / "val_reports"
        candidates = sorted(
            reports.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if reports.is_dir() else []
        candidates = [p for p in candidates if "audit" not in p.name.lower()]
        if not candidates:
            raise FileNotFoundError(
                f"DRIV membutuhkan summary DRNET yang valid, tidak ditemukan di {reports}"
            )
        cmd += ["--drnet-summary", str(candidates[0])]
    return cmd


base.run = run_with_tee
base.build_runner_command = build_runner_command_with_dependencies


if __name__ == "__main__":
    base.main()
