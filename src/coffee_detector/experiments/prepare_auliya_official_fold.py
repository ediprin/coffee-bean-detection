"""Materialize only the official fold-A development images from the verified pilot bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import yaml

from coffee_detector.analysis.auliya_robusta_preflight import (
    METADATA_SHA256, _label_name, _load_folds, _load_manifest, _read_text, _sha256,
)

PROTOCOL = "auliya-official-fold-a-frequency-v1"
PILOT_SHA256 = "b8795e189b85a13201e54a562cbe0f42fa7e0f2bfc0eb4b6d04f85d2b4cf57be"
COUNTS = {"train": 676, "val": 119}
GROUP_A_COUNT = 1351
SPLIT_SHA256 = {
    "train": "0f35195ef060cf072bee2d9853f936b54a509ac6ea73c0b69b1896b24a32f89e",
    "val": "15d2102496acb72f889f82fb385175c2ac724b05c3483e3918a8b6625d8a954f",
}


def _archive_paths(image: str, group: str) -> tuple[str, str]:
    old_split, _, basename = image.split("/")
    role = {"B": "train", "C": "val"}[group]
    base = f"development/{role}"
    return f"{base}/images/{old_split}__{basename}", f"{base}/labels/{old_split}__{Path(basename).stem}.txt"


def _dataset(names: list[str]) -> dict:
    return {"train": "train.txt", "val": "val.txt", "nc": 21, "names": names}


def prepare_official_fold(metadata: Path, pilot: Path, output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh output for official fold A: {output}")
    if _sha256(metadata) != METADATA_SHA256 or _sha256(pilot) != PILOT_SHA256:
        raise ValueError("Expected the verified Zenodo metadata ZIP and pilot Colab bundle")

    with ZipFile(metadata) as src, ZipFile(pilot) as images:
        manifest = _load_manifest(src)
        official = _load_folds(src, manifest)["A"]
        if {r: len(official[r]) for r in COUNTS} != COUNTS or len(official["test"]) != GROUP_A_COUNT:
            raise ValueError("Official split does not have the frozen fold-A counts")
        names = yaml.safe_load(_read_text(src, "splits/data_v4_ujiA.yaml"))["names"]
        bundled = json.loads(images.read("development/development_contract.json"))
        if len(names) != 21 or names != bundled["names"] or bundled["groups"] != {
            "train": "B", "val": "C", "held_out": "A",
        }:
            raise ValueError("Pilot bundle does not cover the official B/C development pool")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".auliya-official-", dir=output.parent) as temp:
            stage = Path(temp) / "fold_a"
            stage.mkdir()
            for role in COUNTS:
                paths = []
                for image in official[role]:
                    if not all(part not in {"..", "."} for part in PurePosixPath(image).parts):
                        raise ValueError(f"Invalid official path: {image}")
                    sources = _archive_paths(image, manifest[image]["group"])
                    # Keep the original path order and name so dataloader sorting is unchanged.
                    relative = f"source/{image}"
                    for source, dest in zip(sources, (relative, f"source/{_label_name(image)}"), strict=True):
                        target = stage / dest
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with images.open(source) as stream, target.open("xb") as out:
                            shutil.copyfileobj(stream, out)
                    paths.append(f"./{relative}")
                (stage / f"{role}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")
            (stage / "data.yaml").write_text(yaml.safe_dump(_dataset(names), sort_keys=False), encoding="utf-8")
            contract = {
                "protocol": PROTOCOL, "held_out_group": "A", "held_out_count": GROUP_A_COUNT,
                "held_out_contents_read": False, "images": COUNTS, "names": names,
                "metadata_sha256": METADATA_SHA256, "pilot_sha256": PILOT_SHA256,
                "split_sha256": {r: _sha256(stage / f"{r}.txt") for r in COUNTS},
            }
            (stage / "official_fold_contract.json").write_text(
                json.dumps(contract, indent=2) + "\n", encoding="utf-8", newline="\n",
            )
            validate_official_fold(stage)
            stage.rename(output)
    return contract


def validate_official_fold(root: Path) -> Path:
    """Reject a changed list, leaked test input, or a noncanonical dataset YAML."""
    contract = json.loads((root / "official_fold_contract.json").read_text(encoding="utf-8"))
    names = contract["names"]
    expected = {
        "protocol": PROTOCOL, "held_out_group": "A", "held_out_count": GROUP_A_COUNT,
        "held_out_contents_read": False, "images": COUNTS, "names": names,
        "metadata_sha256": METADATA_SHA256, "pilot_sha256": PILOT_SHA256,
        "split_sha256": SPLIT_SHA256,
    }
    data = root / "data.yaml"
    if (contract != expected or len(names) != 21 or len(set(names)) != 21
            or yaml.safe_load(data.read_text(encoding="utf-8")) != _dataset(names)
            or (root / "test").exists() or (root / "test.txt").exists()):
        raise ValueError("Invalid train/val-only fold-A contract")
    images = set()
    labels = set()
    for role, amount in COUNTS.items():
        listing = root / f"{role}.txt"
        paths = listing.read_text(encoding="utf-8").splitlines()
        if _sha256(listing) != SPLIT_SHA256[role] or len(paths) != amount:
            raise ValueError(f"Official fold-A {role} list changed")
        for path in paths:
            original = path.removeprefix("./source/")
            parts = PurePosixPath(original).parts
            if (len(parts) != 3 or parts[0] not in {"train", "valid", "test"}
                    or parts[1] != "images" or PurePosixPath(original).suffix != ".jpg"):
                raise ValueError(f"Invalid fold-A image path: {path}")
            images.add(root / "source" / original)
            labels.add(root / "source" / _label_name(original))
    if (len(images) != sum(COUNTS.values()) or set((root / "source").rglob("*.jpg")) != images
            or set((root / "source").rglob("*.txt")) != labels):
        raise ValueError("Development image and label inventory changed")
    for path in images | labels:
        if path.resolve() != path.absolute() or not path.is_file():
            raise ValueError(f"External or missing fold-A file: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pilot-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_official_fold(args.metadata, args.pilot_bundle, args.output), indent=2))


if __name__ == "__main__":
    main()
