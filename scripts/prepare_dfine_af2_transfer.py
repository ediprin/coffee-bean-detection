from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image as PILImage

from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer


DFINE_REPOSITORY = "Peterande/D-FINE"
DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
DFINE_MODEL_PATH = Path("src/zoo/dfine/dfine.py")
DFINE_NATIVE_CUSTOM_CONFIG = Path("configs/dfine/custom/dfine_hgnetv2_n_custom.yml")
DFINE_PRETRAIN_URL = (
    "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"
)
PATCH_MARKER = "# coffee-detector frozen AF2 bridge"

AF2_MAPPING: dict[str, Any] = {
    "mode": "af2",
    "patch_size": 32,
    "overlap": 0.50,
    "radius_ratio": 0.05,
    "gamma": 0.10,
    "angular_bins": 360,
    "chunk_size": 128,
    "eps": 1.0e-8,
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def verify_dfine_checkout(dfine_root: str | Path) -> dict[str, Any]:
    root = Path(dfine_root).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Bukan checkout Git D-FINE: {root}")
    head = _git(root, "rev-parse", "HEAD")
    if head != DFINE_COMMIT:
        raise RuntimeError(f"D-FINE harus commit {DFINE_COMMIT}; diterima {head}")
    model_path = root / DFINE_MODEL_PATH
    base_config = root / DFINE_NATIVE_CUSTOM_CONFIG
    if not model_path.is_file() or not base_config.is_file():
        raise RuntimeError("Checkout D-FINE tidak memiliki file model/config yang dibekukan")
    return {
        "repository": DFINE_REPOSITORY,
        "root": str(root),
        "commit": head,
        "model_path": str(model_path),
        "native_custom_config": str(base_config),
    }


def patch_dfine_for_af2(dfine_root: str | Path) -> Path:
    root = Path(dfine_root).expanduser().resolve()
    verify_dfine_checkout(root)
    path = root / DFINE_MODEL_PATH
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return path

    import_needle = "from ...core import register\n"
    signature_needle = "        decoder: nn.Module,\n    ):\n"
    state_needle = "        self.encoder = encoder\n"
    forward_needle = "    def forward(self, x, targets=None):\n        x = self.backbone(x)\n"
    for needle in (import_needle, signature_needle, state_needle, forward_needle):
        if needle not in text:
            raise RuntimeError(
                "Upstream D-FINE source tidak cocok dengan patch yang dibekukan; "
                f"needle tidak ditemukan: {needle!r}"
            )

    text = text.replace(
        import_needle,
        import_needle
        + "from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer\n",
        1,
    )
    text = text.replace(
        signature_needle,
        "        decoder: nn.Module,\n        af2=None,\n    ):\n",
        1,
    )
    text = text.replace(
        state_needle,
        state_needle
        + f"        {PATCH_MARKER}\n"
        + "        self.input_frontend = (\n"
        + "            None\n"
        + "            if af2 is None\n"
        + "            else AFABInputEnhancer(AFABConfig.from_mapping(af2))\n"
        + "        )\n",
        1,
    )
    text = text.replace(
        forward_needle,
        "    def forward(self, x, targets=None):\n"
        "        if self.input_frontend is not None:\n"
        "            x = self.input_frontend(x)\n"
        "        x = self.backbone(x)\n",
        1,
    )
    path.write_text(text, encoding="utf-8")
    return path


def _load_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, list):
        return [str(item) for item in names]
    if isinstance(names, dict):
        ordered = sorted((int(key), str(value)) for key, value in names.items())
        if [idx for idx, _ in ordered] != list(range(len(ordered))):
            raise RuntimeError("Class IDs pada data.yaml harus kontigu mulai dari 0")
        return [value for _, value in ordered]
    raise RuntimeError("data.yaml tidak memiliki field names berbentuk list/dict")


def _resolve_ultralytics_split(data_yaml: Path, data: dict[str, Any], split: str) -> Path:
    base_raw = data.get("path", ".")
    base = Path(base_raw).expanduser()
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    entry = data.get(split)
    if not isinstance(entry, str):
        raise RuntimeError(f"Split {split!r} harus berupa path direktori pada screen ini")
    value = Path(entry).expanduser()
    if not value.is_absolute():
        value = (base / value).resolve()
    if not value.is_dir():
        raise FileNotFoundError(value)
    return value


def _labels_from_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    positions = [i for i, part in enumerate(parts) if part == "images"]
    if not positions:
        raise RuntimeError(
            f"Tidak dapat menurunkan labels dir dari {images_dir}; path harus memuat komponen 'images'"
        )
    parts[positions[-1]] = "labels"
    labels = Path(*parts)
    if not labels.is_dir():
        raise FileNotFoundError(labels)
    return labels


def _convert_split(images_dir: Path, labels_dir: Path, names: list[str], output_json: Path) -> dict[str, Any]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )
    if not image_paths:
        raise RuntimeError(f"Tidak ada citra pada {images_dir}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    class_counts = [0 for _ in names]

    for image_id, image_path in enumerate(image_paths, start=1):
        relative = image_path.relative_to(images_dir)
        with PILImage.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": relative.as_posix(),
                "width": int(width),
                "height": int(height),
            }
        )

        label_path = (labels_dir / relative).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Label tidak ditemukan untuk {image_path}: {label_path}")
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            raw = raw.strip()
            if not raw:
                continue
            fields = raw.split()
            if len(fields) != 5:
                raise RuntimeError(f"YOLO label bukan 5 kolom: {label_path}:{line_number}")
            class_id = int(fields[0])
            if not 0 <= class_id < len(names):
                raise RuntimeError(f"Class ID di luar names: {label_path}:{line_number} -> {class_id}")
            cx, cy, bw, bh = map(float, fields[1:])
            x0 = max(0.0, (cx - bw / 2.0) * width)
            y0 = max(0.0, (cy - bh / 2.0) * height)
            x1 = min(float(width), (cx + bw / 2.0) * width)
            y1 = min(float(height), (cy + bh / 2.0) * height)
            box_w = x1 - x0
            box_h = y1 - y0
            if box_w <= 0.0 or box_h <= 0.0:
                raise RuntimeError(f"Box degenerat: {label_path}:{line_number}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x0, y0, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            class_counts[class_id] += 1

    categories = [
        {"id": class_id + 1, "name": name, "supercategory": "coffee"}
        for class_id, name in enumerate(names)
    ]
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {
        "images": len(images),
        "annotations": len(annotations),
        "class_counts": class_counts,
        "sha256": sha256(output_json),
        "path": str(output_json.resolve()),
    }


def convert_grouped_yolo_to_coco(data_yaml: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(data_yaml).expanduser().resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    names = _load_names(data)
    if len(names) != 21:
        raise RuntimeError(f"Transfer screen dibekukan untuk 21 kelas; data.yaml memiliki {len(names)}")

    output = Path(output_dir).expanduser().resolve()
    result: dict[str, Any] = {"names": names, "splits": {}}
    for split in ("train", "val"):
        images_dir = _resolve_ultralytics_split(source, data, split)
        labels_dir = _labels_from_images(images_dir)
        split_result = _convert_split(
            images_dir,
            labels_dir,
            names,
            output / f"instances_{split}.json",
        )
        split_result["images_dir"] = str(images_dir)
        split_result["labels_dir"] = str(labels_dir)
        result["splits"][split] = split_result

    expected = {"train": (1665, 2986), "val": (294, 526)}
    for split, (expected_images, expected_annotations) in expected.items():
        actual = result["splits"][split]
        if actual["images"] != expected_images or actual["annotations"] != expected_annotations:
            raise RuntimeError(
                f"Faruq-v3 grouped contract berubah pada {split}: "
                f"{actual['images']} images/{actual['annotations']} annotations, "
                f"expected {expected_images}/{expected_annotations}"
            )
    return result


def write_dfine_pair_configs(
    dfine_root: str | Path,
    conversion_summary: dict[str, Any],
    output_root: str | Path,
) -> dict[str, str]:
    root = Path(dfine_root).expanduser().resolve()
    verify_dfine_checkout(root)
    output_root = Path(output_root).expanduser().resolve()
    config_dir = root / "configs/dfine/custom"

    train = conversion_summary["splits"]["train"]
    val = conversion_summary["splits"]["val"]
    common: dict[str, Any] = {
        "__include__": ["./dfine_hgnetv2_n_custom.yml"],
        "num_classes": 21,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "dataset": {
                "img_folder": train["images_dir"],
                "ann_file": train["path"],
            }
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": val["images_dir"],
                "ann_file": val["path"],
            }
        },
    }

    native = copy.deepcopy(common)
    native["output_dir"] = str(output_root / "DFN0_seed42")
    candidate = copy.deepcopy(common)
    candidate["output_dir"] = str(output_root / "DFN_AF2_seed42")
    candidate["DFINE"] = {"af2": copy.deepcopy(AF2_MAPPING)}

    native_path = config_dir / "dfine_hgnetv2_n_faruq_v3_seed42.yml"
    candidate_path = config_dir / "dfine_hgnetv2_n_faruq_v3_af2_seed42.yml"
    native_path.write_text(yaml.safe_dump(native, sort_keys=False), encoding="utf-8")
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    return {"DFN0": str(native_path), "DFN_AF2": str(candidate_path)}


def af2_static_probe() -> dict[str, Any]:
    config = AFABConfig.from_mapping(AF2_MAPPING)
    frontend = AFABInputEnhancer(config)
    params = sum(parameter.numel() for parameter in frontend.parameters())
    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64, dtype=torch.float32).reshape(1, 3, 64, 64)
    with torch.inference_mode():
        enhanced = frontend(probe)
    return {
        "parameter_count": int(params),
        "shape_preserved": tuple(enhanced.shape) == tuple(probe.shape),
        "finite": bool(torch.isfinite(enhanced).all().item()),
        "max_abs_change": float((enhanced - probe).abs().max().item()),
    }


def prepare(
    dfine_root: str | Path,
    data_yaml: str | Path,
    coco_output: str | Path,
    run_output: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    checkout = verify_dfine_checkout(dfine_root)
    patched_model = patch_dfine_for_af2(dfine_root)
    conversion = convert_grouped_yolo_to_coco(data_yaml, coco_output)
    configs = write_dfine_pair_configs(dfine_root, conversion, run_output)
    probe = af2_static_probe()
    if probe["parameter_count"] != 0 or not probe["shape_preserved"] or not probe["finite"]:
        raise RuntimeError(f"AF2 static probe gagal: {probe}")
    if probe["max_abs_change"] <= 0.0:
        raise RuntimeError("AF2 static probe tidak mengubah input")

    report = {
        "format": "coffee_detector.af2_dfine_transfer.preparation.v1",
        "protocol": "faruq-v3-af2-dfine-n-transfer-seed42-v1",
        "seed": 42,
        "test_images_accessed": False,
        "dfine": checkout,
        "patched_model": str(patched_model),
        "patched_model_sha256": sha256(patched_model),
        "dfine_pretrain_url": DFINE_PRETRAIN_URL,
        "dfine_pretrain_sha256": None,
        "af2": copy.deepcopy(AF2_MAPPING),
        "af2_probe": probe,
        "dataset": conversion,
        "configs": configs,
        "training_commands": {
            "DFN0": (
                f"python train.py -c {configs['DFN0']} --use-amp --seed=42 "
                "-t /path/to/dfine_n_coco.pth"
            ),
            "DFN_AF2": (
                f"python train.py -c {configs['DFN_AF2']} --use-amp --seed=42 "
                "-t /path/to/dfine_n_coco.pth"
            ),
        },
    }
    destination = Path(report_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare frozen AF2 × D-FINE-N Faruq-v3 transfer screen")
    parser.add_argument("--dfine-root", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--coco-output", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = prepare(
        dfine_root=args.dfine_root,
        data_yaml=args.data_yaml,
        coco_output=args.coco_output,
        run_output=args.run_output,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
