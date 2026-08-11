from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

from .audit_dataset import audit_dataset
from .dataset import IMAGE_SUFFIXES, parse_label, write_json
from .prepare_sni_fullscene import SNI21_CLASSES


SOURCES = ("adrian_detection", "faruq_segmentation")
DEVELOPMENT_SPLITS = ("train", "val")


def _source_from_name(name: str) -> str:
    matches = [source for source in SOURCES if name.startswith(f"{source}__")]
    if len(matches) != 1:
        raise ValueError(
            "Nama gambar A0 tidak menyimpan tepat satu source dataset: " + name
        )
    return matches[0]


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(source, target)


def _count_boxes(label: Path, valid_ids: set[int]) -> Counter[int]:
    return Counter(box.class_id for box in parse_label(label, valid_ids))


def separate_sni21_sources(
    combined_root: str | Path,
    output_root: str | Path,
    *,
    link_mode: str = "auto",
) -> dict:
    """Split combined A0 train/validation into independent source domains.

    Test is deliberately neither discovered nor materialized. The function is a
    development-data operation used to disentangle annotation/domain effects;
    it never trains a model.
    """

    if link_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("link_mode harus auto, hardlink, atau copy")
    combined_root = Path(combined_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not combined_root.is_dir():
        raise FileNotFoundError(f"A0 gabungan tidak ditemukan: {combined_root}")
    if output_root.exists() and any(output_root.iterdir()):
        summary_path = output_root / "source_separation_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("status") == "complete"
                and Path(summary.get("combined_root", "")).resolve() == combined_root
            ):
                print(f"REUSE SOURCE SEPARATION: {output_root}", flush=True)
                return summary
        raise FileExistsError(f"Output pemisahan tidak kosong/complete: {output_root}")

    valid_ids = set(range(len(SNI21_CLASSES)))
    counts: dict[str, dict[str, Counter]] = {
        source: {
            "images": Counter(),
            "boxes": Counter(),
            "classes": Counter(),
        }
        for source in SOURCES
    }
    manifest: list[dict] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for split in DEVELOPMENT_SPLITS:
        image_root = combined_root / split / "images"
        label_root = combined_root / split / "labels"
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(
                f"A0 development belum lengkap: {image_root} / {label_root}"
            )
        image_paths = sorted(
            path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for index, image_path in enumerate(image_paths, 1):
            source = _source_from_name(image_path.name)
            relative = image_path.relative_to(image_root)
            label_path = (label_root / relative).with_suffix(".txt")
            if not label_path.is_file():
                raise FileNotFoundError(f"Label A0 tidak ditemukan: {label_path}")

            target_root = output_root / source
            image_target = target_root / split / "images" / relative
            label_target = target_root / split / "labels" / relative.with_suffix(".txt")
            _link_or_copy(image_path, image_target, link_mode)
            _link_or_copy(label_path, label_target, link_mode)
            class_counts = _count_boxes(label_path, valid_ids)
            counts[source]["images"][split] += 1
            counts[source]["boxes"][split] += sum(class_counts.values())
            counts[source]["classes"].update(class_counts)
            manifest.append(
                {
                    "source_dataset": source,
                    "split": split,
                    "input_image": str(image_path),
                    "input_label": str(label_path),
                    "output_image": str(image_target),
                    "output_label": str(label_target),
                    "boxes": sum(class_counts.values()),
                }
            )
            if index % 1000 == 0 or index == len(image_paths):
                print(
                    f"SEPARATE {split}: {index}/{len(image_paths)} gambar",
                    flush=True,
                )

    names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    audits = {}
    source_rows = []
    for source in SOURCES:
        source_root = output_root / source
        payload = {
            "path": str(source_root),
            "train": "train/images",
            "val": "val/images",
            "names": names,
        }
        (source_root / "data.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        audit_path = source_root / "audit.json"
        audit = audit_dataset(source_root, audit_path, near_threshold=-1)
        audits[source] = str(audit_path)
        missing_train = [
            SNI21_CLASSES[class_id]
            for class_id in range(len(SNI21_CLASSES))
            if counts[source]["classes"][class_id] == 0
        ]
        source_rows.append(
            {
                "source_dataset": source,
                "train_images": counts[source]["images"]["train"],
                "val_images": counts[source]["images"]["val"],
                "train_boxes": counts[source]["boxes"]["train"],
                "val_boxes": counts[source]["boxes"]["val"],
                "missing_classes_development": missing_train,
                "audit_safe": bool(audit["safe_for_training"]),
                "dataset_root": str(source_root),
            }
        )

    summary = {
        "format": "coffee_detector.sni21_source_separation.v1",
        "status": "complete",
        "combined_root": str(combined_root),
        "output_root": str(output_root),
        "sources": list(SOURCES),
        "splits_materialized": list(DEVELOPMENT_SPLITS),
        "test_locked": True,
        "test_images_accessed": False,
        "training_executed": False,
        "class_id_policy": "SNI21 canonical IDs retained without remapping",
        "link_mode": link_mode,
        "rows": source_rows,
        "audits": audits,
        "claim_note": (
            "Adrian detection dan Faruq segmentation diperlakukan sebagai dua "
            "domain eksperimen independen; hasil A0 gabungan bukan kontrol kausal "
            "untuk efek arsitektur."
        ),
    }
    write_json(manifest, output_root / "source_separation_manifest.json")
    write_json(summary, output_root / "source_separation_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pisahkan A0 SNI-21 gabungan menjadi development dataset Adrian dan "
            "Faruq tanpa membuka test atau training."
        )
    )
    parser.add_argument("--combined-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--link-mode", choices=("auto", "hardlink", "copy"), default="auto"
    )
    args = parser.parse_args()
    result = separate_sni21_sources(
        args.combined_root, args.output_root, link_mode=args.link_mode
    )
    print("\n=== SNI-21 SOURCE SEPARATION ===")
    for row in result["rows"]:
        print(
            f"{row['source_dataset']}: "
            f"train={row['train_images']} img/{row['train_boxes']} box | "
            f"val={row['val_images']} img/{row['val_boxes']} box | "
            f"audit={'PASS' if row['audit_safe'] else 'FAIL'}"
        )
    print(f"TEST LOCKED: {result['test_locked']}")
    print(f"TRAINING: {result['training_executed']}")
    print(f"SAVED: {Path(result['output_root']) / 'source_separation_summary.json'}")


if __name__ == "__main__":
    main()
