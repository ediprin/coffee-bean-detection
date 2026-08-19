from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from coffee_detector.stb.model import STBDetectHead
from coffee_detector.stb_guided.audit import sha256


TEACHER_NAME = "STB1_seed42_teacher_best.pt"
MANIFEST_NAME = "stb_guided_teacher_manifest.json"
MANIFEST_FORMAT = "coffee_detector.stb_guided.teacher_addon.v1"
PROJECT_TEACHER = (
    "experiments/faruq-v3-breadth-screening-batch-v1/"
    "candidates/STB1/STB1_seed42/weights/best.pt"
)
EXPECTED_PARAMETERS = 4_589_201


def _validate_teacher(path: str | Path) -> dict:
    from ultralytics import YOLO

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
        raise FileNotFoundError(checkpoint)
    model = YOLO(str(checkpoint)).model.cpu().eval()
    head = model.model[-1]
    if not isinstance(head, STBDetectHead):
        raise RuntimeError(f"Teacher addon bukan STB1: {type(head).__name__}")
    nc = int(getattr(head, "nc", -1))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if nc != 21:
        raise RuntimeError(f"Teacher bukan SNI-21: nc={nc}")
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Parameter STB1 berubah: {parameters} != {EXPECTED_PARAMETERS}"
        )
    return {
        "loadable_by_ultralytics": True,
        "head": type(head).__name__,
        "nc": nc,
        "parameters": parameters,
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256(checkpoint),
    }


def build_stb_guided_teacher_addon(
    project_root: str | Path,
    output_root: str | Path,
) -> dict:
    project_root = Path(project_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source = project_root / PROJECT_TEACHER
    if not source.is_file():
        raise FileNotFoundError(f"STB1 seed42 tidak ditemukan: {source}")
    target = output_root / TEACHER_NAME
    shutil.copy2(source, target)
    if sha256(source) != sha256(target) or source.stat().st_size != target.stat().st_size:
        raise RuntimeError("Copy STB1 teacher mengubah bytes/SHA")
    proof = _validate_teacher(target)
    manifest = {
        "format": MANIFEST_FORMAT,
        "teacher": {
            "name": TEACHER_NAME,
            "source": PROJECT_TEACHER,
            **proof,
        },
        "test_images_included": False,
    }
    manifest_path = output_root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def prepare_stb_guided_teacher_addon(input_root: str | Path) -> tuple[Path, dict]:
    input_root = Path(input_root).expanduser().resolve()
    manifests = sorted(input_root.rglob(MANIFEST_NAME))
    if len(manifests) != 1:
        raise FileNotFoundError(f"Harus ada satu {MANIFEST_NAME}; ditemukan {manifests}")
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    if payload.get("format") != MANIFEST_FORMAT:
        raise RuntimeError("Format teacher addon tidak sesuai")
    if payload.get("test_images_included") is not False:
        raise RuntimeError("Teacher addon tidak boleh memuat test images")
    contract = payload.get("teacher", {})
    if contract.get("name") != TEACHER_NAME:
        raise RuntimeError("Nama teacher addon berubah")
    matches = sorted(input_root.rglob(TEACHER_NAME))
    if len(matches) != 1:
        raise FileNotFoundError(f"Harus ada satu {TEACHER_NAME}; ditemukan {matches}")
    teacher = matches[0]
    if teacher.stat().st_size != int(contract["bytes"]) or sha256(teacher) != contract["sha256"]:
        raise RuntimeError("SHA/ukuran STB1 teacher addon gagal")
    proof = _validate_teacher(teacher)
    if proof["sha256"] != contract["sha256"]:
        raise RuntimeError("Load-test teacher tidak sesuai manifest")
    result = {
        "format": "coffee_detector.stb_guided.teacher_input.v1",
        "teacher": str(teacher),
        "teacher_sha256": proof["sha256"],
        "parameters": proof["parameters"],
        "test_images_accessed": False,
        "decision": "PASS",
    }
    return teacher, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/validate STB1 seed42 Kaggle teacher addon")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--project-root", required=True)
    build.add_argument("--output-root", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--input-root", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_stb_guided_teacher_addon(args.project_root, args.output_root)
    else:
        _, result = prepare_stb_guided_teacher_addon(args.input_root)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
