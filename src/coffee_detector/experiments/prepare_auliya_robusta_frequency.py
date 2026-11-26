"""Build the group-disjoint train/validation dataset for the SNI frequency study."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import yaml

from coffee_detector.analysis.auliya_robusta_preflight import (
    CLASS_COUNT,
    METADATA_SHA256,
    _archive_members,
    _audit_archive,
    _label_name,
    _load_folds,
    _load_manifest,
    _read_text,
    _sha256,
)


ARCHIVE_SHA256 = "e39cb0343bc19de5e3a5307985f43575a537977a96fde8d745c7e4524ba9bf9f"
PROTOCOL = "auliya-robusta-sni-frequency-v1"
GROUPS = {"train": "B", "val": "C"}
IMAGE_COUNTS = {"train": 375, "val": 420}


def prepare(metadata_zip: Path, images_zip: Path, output: Path) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use an unused development output directory: {output}")
    if _sha256(metadata_zip) != METADATA_SHA256 or _sha256(images_zip) != ARCHIVE_SHA256:
        raise ValueError("Expected verified Zenodo metadata and Roboflow v2 YOLOv8 export")
    with ZipFile(metadata_zip) as metadata, ZipFile(images_zip) as images:
        manifest = _load_manifest(metadata)
        official = _load_folds(metadata, manifest)["A"]
        roles = {
            role: sorted(path for path, row in manifest.items() if row["group"] == group)
            for role, group in GROUPS.items()
        }
        if {role: len(paths) for role, paths in roles.items()} != IMAGE_COUNTS:
            raise ValueError("Unexpected counts in the frozen B/C development split")
        if set(roles["train"] + roles["val"]) != set(official["train"] + official["val"]):
            raise ValueError("Development data must exactly cover the official non-A pool")
        polygon_counts = _audit_archive(images, manifest, roles)
        if any(not count for values in polygon_counts.values() for count in values.values()):
            raise ValueError("Both development groups must cover all 21 classes")
        names = yaml.safe_load(_read_text(metadata, "splits/data_v4_ujiA.yaml"))["names"]
        if len(names) != CLASS_COUNT or yaml.safe_load(_read_text(images, "data.yaml"))["names"] != names:
            raise ValueError("Class IDs differ between the image archive and the official metadata")
        entries = _archive_members(images)
        contract = {
            "protocol": PROTOCOL,
            "archive_sha256": ARCHIVE_SHA256,
            "metadata_sha256": METADATA_SHA256,
            "groups": {**GROUPS, "held_out": "A"},
            "images": IMAGE_COUNTS,
            "held_out_images": len(official["test"]),
            "polygon_instances_by_class": polygon_counts,
            "names": names,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent, prefix=".auliya-") as temp:
            staged = Path(temp) / "development"
            staged.mkdir()
            for role, paths in roles.items():
                for path in paths:
                    source_split, _, filename = path.split("/")
                    # Keep the original folder in the destination name to retain provenance.
                    image_target = staged / role / "images" / f"{source_split}__{filename}"
                    label_target = (
                        staged / role / "labels" / f"{source_split}__{Path(filename).stem}.txt"
                    )
                    for key, target in ((path, image_target), (_label_name(path), label_target)):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with images.open(entries[key]) as source, target.open("xb") as destination:
                            shutil.copyfileobj(source, destination)
            (staged / "data.yaml").write_text(yaml.safe_dump({
                "train": "train/images", "val": "val/images", "names": names,
            }, sort_keys=False), encoding="utf-8")
            (staged / "development_contract.json").write_text(
                json.dumps(contract, indent=2) + "\n", encoding="utf-8"
            )
            staged.rename(output)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.metadata_zip, args.images_zip, args.output), indent=2))


if __name__ == "__main__":
    main()
