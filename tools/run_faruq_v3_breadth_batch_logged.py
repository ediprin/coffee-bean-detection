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
    """Preserve controller semantics while teeing candidate stdout/stderr to Drive.

    Helper/git/preflight calls keep the original capture behavior. Actual candidate
    runner calls stream line-by-line to both the Colab console and a persistent
    runner.log under that candidate's output root, so a first-batch failure keeps
    the full traceback instead of only surfacing `Runner exit code 1`.
    """
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


base.run = run_with_tee


if __name__ == "__main__":
    base.main()
