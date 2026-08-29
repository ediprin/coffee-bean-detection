from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
importlib.invalidate_caches()

from coffee_detector.experiments.prepare_faruq_v3_kaggle import (  # noqa: E402
    prepare_faruq_v3_kaggle_input,
)
from analyze_dfine_af2_transfer import compare, summarize  # noqa: E402
from apply_dfine_single_gpu_profile import apply as apply_single_gpu_profile  # noqa: E402
from preflight_dfine_af2_transfer import run_preflight  # noqa: E402
from prepare_dfine_af2_transfer import prepare  # noqa: E402


DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
DFINE_PRETRAIN_URL = (
    "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"
)
DFINE_PRETRAIN_BYTES = 15_489_558


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def log_last_epoch(run_dir: Path) -> int:
    path = run_dir / "log.txt"
    if not path.is_file():
        return -1
    last = -1
    for line in path.read_text(errors="replace").splitlines():
        try:
            last = max(last, int(json.loads(line).get("epoch", -1)))
        except Exception:
            continue
    return last


def snapshot_state(work: Path, out: Path, state_zip: Path) -> Path:
    if state_zip.exists():
        state_zip.unlink()
    archive = Path(
        shutil.make_archive(
            str(state_zip.with_suffix("")),
            "zip",
            root_dir=work,
            base_dir=out.name,
        )
    )
    print("STATE SNAPSHOT:", archive, archive.stat().st_size, "bytes", flush=True)
    return archive


def train_arm(
    *,
    label: str,
    config: Path,
    dfine: Path,
    pretrained: Path,
    out: Path,
    work: Path,
    state_zip: Path,
) -> None:
    run_dir = out / f"{label}_seed42"
    last_epoch = log_last_epoch(run_dir)
    if last_epoch >= 219:
        print(label, "already complete at epoch", last_epoch, flush=True)
        return

    last_ckpt = run_dir / "last.pth"
    command = [
        sys.executable,
        "-u",
        "train.py",
        "-c",
        str(config),
        "--use-amp",
        "--seed=42",
    ]
    if last_ckpt.is_file():
        command += ["-r", str(last_ckpt)]
        print(label, "RESUME", last_ckpt, "logged epoch", last_epoch, flush=True)
    else:
        command += ["-t", str(pretrained)]
        print(label, "START from official D-FINE-N COCO", flush=True)

    log_path = out / f"{label}_console.log"
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=dfine,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-180:]))
        snapshot_state(work, out, state_zip)
        raise RuntimeError(f"{label} training failed rc={process.returncode}")

    completed = log_last_epoch(run_dir)
    if completed < 219:
        snapshot_state(work, out, state_zip)
        raise RuntimeError(f"{label} ended before frozen epoch 219: {completed}")
    print(label, "TRAINING COMPLETE", completed, flush=True)
    snapshot_state(work, out, state_zip)


def select_official_best(run_dir: Path) -> Path:
    stage2 = run_dir / "best_stg2.pth"
    stage1 = run_dir / "best_stg1.pth"
    if stage2.is_file():
        return stage2
    if stage1.is_file():
        return stage1
    raise FileNotFoundError(f"No official D-FINE best checkpoint under {run_dir}")


def evaluate_arm(*, label: str, config: Path, dfine: Path, out: Path) -> tuple[Path, Path]:
    run_dir = out / f"{label}_seed42"
    best = select_official_best(run_dir)
    eval_path = run_dir / "eval.pth"
    command = [
        sys.executable,
        "-u",
        "train.py",
        "-c",
        str(config),
        "--test-only",
        "-r",
        str(best),
        "--seed=42",
    ]
    log_path = out / f"{label}_eval_console.log"
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=dfine,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if process.returncode:
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-180:]))
        raise RuntimeError(f"{label} validation failed rc={process.returncode}")
    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)
    print(label, "BEST", best.name, "EVAL", eval_path, flush=True)
    return eval_path, best


def main() -> None:
    input_root = Path("/kaggle/input")
    work = Path("/kaggle/working")
    if not input_root.is_dir() or not work.is_dir():
        raise RuntimeError("Kaggle-only runner")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Aktifkan Kaggle GPU")

    dfine = work / "D-FINE"
    if not (dfine / ".git").is_dir():
        raise FileNotFoundError(
            "Pinned D-FINE checkout tidak ditemukan di /kaggle/working/D-FINE"
        )
    if git_head(dfine) != DFINE_COMMIT:
        raise RuntimeError(f"D-FINE commit mismatch: {git_head(dfine)}")

    data, core = prepare_faruq_v3_kaggle_input(input_root, work)
    if core.get("decision") != "PASS" or core.get("test_images_accessed") is not False:
        raise RuntimeError("Faruq-v3 development input contract gagal")
    if (data / "test").exists():
        raise RuntimeError("TEST TEREXPOSE — STOP")

    out = work / "af2-dfine-n-transfer-seed42-v1"
    state_zip = work / "af2-dfine-n-transfer-seed42-state.zip"

    # Optional checkpointed run state from a previous Kaggle session.
    resume_states = sorted(input_root.rglob(state_zip.name))
    if len(resume_states) > 1:
        raise RuntimeError(f"Resume state ambigu: {resume_states}")
    if len(resume_states) == 1 and not out.exists():
        print("RESTORE STATE:", resume_states[0], flush=True)
        with zipfile.ZipFile(resume_states[0], "r") as archive:
            archive.extractall(work)
    out.mkdir(parents=True, exist_ok=True)

    coco_dir = work / "faruq-v3-dfine-coco"
    prep_path = out / "dfine_transfer_preparation.json"
    prep = prepare(
        dfine_root=dfine,
        data_yaml=data / "data.yaml",
        coco_output=coco_dir,
        run_output=out,
        report_path=prep_path,
    )
    native_config = Path(prep["configs"]["DFN0"])
    af2_config = Path(prep["configs"]["DFN_AF2"])

    # Hardware profile is frozen prospectively and applied identically.
    apply_single_gpu_profile(native_config)
    apply_single_gpu_profile(af2_config)

    pretrained = work / "dfine_n_coco.pth"
    if not pretrained.is_file():
        print("DOWNLOAD OFFICIAL D-FINE-N COCO CHECKPOINT...", flush=True)
        urllib.request.urlretrieve(DFINE_PRETRAIN_URL, pretrained)
    if pretrained.stat().st_size != DFINE_PRETRAIN_BYTES:
        raise RuntimeError(
            f"Unexpected D-FINE-N checkpoint bytes: {pretrained.stat().st_size}"
        )
    pretrained_sha = sha256(pretrained)
    print("PRETRAIN SHA256:", pretrained_sha, flush=True)

    static_path = out / "dfine_af2_static_preflight.json"
    static = run_preflight(
        dfine_root=dfine,
        native_config=native_config,
        af2_config=af2_config,
        pretrained_checkpoint=pretrained,
        preparation_report=prep_path,
        output=static_path,
        seed=42,
    )
    failed = [key for key, value in static["gates"].items() if not value]
    if static["decision"] != "PASS" or failed:
        raise RuntimeError(f"STATIC PREFLIGHT FAIL: {failed}")
    if static["pretrained_checkpoint_sha256"] != pretrained_sha:
        raise RuntimeError("Checkpoint SHA changed during preflight")
    if (data / "test").exists():
        raise RuntimeError("TEST TEREXPOSE — STOP")

    print("STATIC PREFLIGHT PASS", flush=True)
    print(
        "PARAMS native/candidate:",
        static["native_parameter_count"],
        static["candidate_parameter_count"],
        flush=True,
    )
    print("COMMON INIT SHA:", static["common_initialized_detector_state_sha256"], flush=True)
    print("AF2 PARAMS:", static["af2_learned_parameter_count"], flush=True)

    train_arm(
        label="DFN0",
        config=native_config,
        dfine=dfine,
        pretrained=pretrained,
        out=out,
        work=work,
        state_zip=state_zip,
    )
    train_arm(
        label="DFN_AF2",
        config=af2_config,
        dfine=dfine,
        pretrained=pretrained,
        out=out,
        work=work,
        state_zip=state_zip,
    )

    control_eval, control_best = evaluate_arm(
        label="DFN0", config=native_config, dfine=dfine, out=out
    )
    candidate_eval, candidate_best = evaluate_arm(
        label="DFN_AF2", config=af2_config, dfine=dfine, out=out
    )

    val_annotations = Path(prep["dataset"]["splits"]["val"]["path"])
    control = summarize(control_eval, val_annotations)
    candidate = summarize(candidate_eval, val_annotations)
    summary = compare(control, candidate)
    summary["provenance"] = {
        "dfine_commit": DFINE_COMMIT,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "native_initialized_detector_sha256": static[
            "common_initialized_detector_state_sha256"
        ],
        "candidate_initialized_detector_sha256": static[
            "candidate_initialized_detector_state_sha256"
        ],
        "native_best_checkpoint": str(control_best),
        "candidate_best_checkpoint": str(candidate_best),
        "single_gpu_profile": "batch16-linear-scaled-v1",
        "dataset_archive_sha256": core["archive_sha256"],
    }
    summary_path = out / "af2_dfine_seed42_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    snapshot_state(work, out, state_zip)

    c = summary["control"]
    a = summary["candidate"]
    d = summary["deltas_af2_minus_native"]
    print("\n=== AF2 × D-FINE-N SEED42 ===")
    print(
        f"Macro     DFN0={c['macro_class_ap50_95']:.4f} "
        f"DFN_AF2={a['macro_class_ap50_95']:.4f} delta={d['macro']:+.4f}"
    )
    print(
        f"Bottom-3  DFN0={c['bottom3_class_ap50_95']:.4f} "
        f"DFN_AF2={a['bottom3_class_ap50_95']:.4f} delta={d['bottom3']:+.4f}"
    )
    print(
        f"Worst     DFN0={c['worst_class_ap50_95']:.4f} "
        f"DFN_AF2={a['worst_class_ap50_95']:.4f} delta={d['worst']:+.4f}"
    )
    print(
        f"COCO AP   DFN0={c['global_coco_ap_from_precision']:.4f} "
        f"DFN_AF2={a['global_coco_ap_from_precision']:.4f} "
        f"delta={d['global_coco_ap']:+.4f}"
    )
    print("SCREEN:", json.dumps(summary["screen"], indent=2))
    print("PRETRAIN SHA256:", pretrained_sha)
    print("TEST:", summary["test_images_accessed"])
    print("SUMMARY:", summary_path)
    print("STATE ZIP:", state_zip)
    print("Kirim output akhir ini. Jangan buka test.")


if __name__ == "__main__":
    main()
