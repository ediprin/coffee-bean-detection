"""Recover J25 source siblings and freeze a class-complete source-level split."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from PIL import Image

from coffee_detector.analysis.coffee_standard_j25_thesis_provenance import AUTHOR_ARCHIVE_SHA256
from coffee_detector.data.prepare_coffee_standard_primary import (
    J25_CLASSES,
    grouped_three_way_assignment,
)


FORMAT = "coffee_detector.coffee_standard_j25_source_split.v1"
TRAIN_SIBLINGS_FORMAT = "coffee_detector.coffee_standard_j25_source_split.train_siblings.v2"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent(stem: str) -> str:
    return re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.I)


def _vector(image_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as image:
        value = np.asarray(
            image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32
        ).reshape(-1)
    value = (value - value.mean()) / (value.std() + 1e-6)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def _representative(archive: zipfile.ZipFile, members: list[dict]) -> tuple[dict, float]:
    if len(members) == 1:
        return members[0], 1.0
    vectors = np.stack([_vector(archive.read(row["image_member"])) for row in members])
    similarity = vectors @ vectors.T
    scores = (1.0 - similarity).sum(axis=1)
    chosen = min(range(len(members)), key=lambda index: (float(scores[index]), members[index]["image_member"]))
    minimum_pair_similarity = float(
        min(similarity[left, right] for left in range(len(members)) for right in range(left + 1, len(members)))
    )
    return members[chosen], minimum_pair_similarity


def prepare_j25_source_split(
    archive_path: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    expected_sha256: str = AUTHOR_ARCHIVE_SHA256,
    retain_train_siblings: bool = False,
) -> dict:
    archive_path = Path(archive_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    archive_sha = _sha256_file(archive_path)
    if archive_sha != expected_sha256.lower():
        raise RuntimeError(f"SHA arsip penulis berubah: {archive_sha}")
    summary_name = (
        "coffee_standard_j25_train_siblings_summary.json"
        if retain_train_siblings
        else "coffee_standard_j25_source_split_summary.json"
    )
    manifest_name = (
        "coffee_standard_j25_train_siblings_manifest.json"
        if retain_train_siblings
        else "coffee_standard_j25_source_split_manifest.json"
    )
    selected_format = TRAIN_SIBLINGS_FORMAT if retain_train_siblings else FORMAT
    summary_path = output_root / summary_name
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            cached.get("source_archive_sha256") == archive_sha
            and cached.get("seed") == seed
            and cached.get("format") == selected_format
        ):
            cached["summary"] = str(summary_path)
            return cached
        raise RuntimeError("Output lama berbeda kontrak")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        yaml_texts = {
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if PurePosixPath(name).name == "data.yaml"
        }
        if len(yaml_texts) != 1:
            raise RuntimeError("data.yaml ambigu")
        source_yaml = yaml.safe_load(yaml_texts.pop())
        if tuple(source_yaml.get("names", [])) != J25_CLASSES:
            raise RuntimeError("Ontologi bukan J25")
        for source_split in ("train", "valid", "test"):
            prefix = f"{source_split}/images/"
            for image_member in archive.namelist():
                if not image_member.startswith(prefix) or not image_member.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                stem = PurePosixPath(image_member).stem
                label_member = f"{source_split}/labels/{stem}.txt"
                label_bytes = archive.read(label_member)
                identity_material = _parent(stem).encode() + b"\0" + label_bytes
                identity = _sha256_bytes(identity_material)
                groups[identity].append(
                    {
                        "source_split": source_split,
                        "image_member": image_member,
                        "label_member": label_member,
                        "parent_basename": _parent(stem),
                        "label_sha256": _sha256_bytes(label_bytes),
                    }
                )

        components = []
        representatives = {}
        minimum_similarities = []
        for identity, members in sorted(groups.items()):
            source_splits = {row["source_split"] for row in members}
            if len(source_splits) != 1:
                raise RuntimeError(f"Recovered identity crosses supplied splits: {identity}")
            label_payloads = {archive.read(row["label_member"]) for row in members}
            if len(label_payloads) != 1:
                raise RuntimeError(f"Sibling labels differ: {identity}")
            label_bytes = label_payloads.pop()
            counts = Counter(
                int(line.split()[0]) for line in label_bytes.decode("utf-8").splitlines() if line.strip()
            )
            representative, minimum_similarity = _representative(archive, members)
            if len(members) > 1:
                minimum_similarities.append(minimum_similarity)
            representatives[identity] = representative
            components.append(
                {
                    "component_id": identity,
                    "images": 1,
                    "source_derivatives": len(members),
                    "source_split": next(iter(source_splits)),
                    "class_counts": [counts[index] for index in range(25)],
                    "representative": representative["image_member"],
                    "minimum_sibling_similarity": minimum_similarity,
                }
            )

        assignment, optimizer = grouped_three_way_assignment(components, seed=seed, restarts=128)
        manifest = []
        split_class_counts = {split: Counter() for split in ("train", "val", "test")}
        split_identity_counts = Counter()
        split_image_counts = Counter()
        for component in components:
            identity = component["component_id"]
            split = assignment[identity]
            row = representatives[identity]
            extraction_rows = (
                sorted(groups[identity], key=lambda item: item["image_member"])
                if retain_train_siblings and split == "train"
                else [row]
            )
            manifest.append(
                {
                    "component_id": identity,
                    "output_split": split,
                    "source_split": component["source_split"],
                    "source_derivatives": component["source_derivatives"],
                    "representative_image_member": row["image_member"],
                    "representative_label_member": row["label_member"],
                    "development_image_members": (
                        [item["image_member"] for item in extraction_rows]
                        if split != "test"
                        else []
                    ),
                }
            )
            split_identity_counts[split] += 1
            evaluation_rows = extraction_rows if split == "train" else [row]
            split_image_counts[split] += len(evaluation_rows)
            for selected in evaluation_rows:
                selected_label_bytes = archive.read(selected["label_member"])
                selected_counts = Counter(
                    int(line.split()[0])
                    for line in selected_label_bytes.decode("utf-8").splitlines()
                    if line.strip()
                )
                split_class_counts[split].update(selected_counts)
            if split == "test":
                continue
            for derivative_index, selected in enumerate(extraction_rows):
                suffix = PurePosixPath(selected["image_member"]).suffix.lower()
                stem = (
                    f"{identity}_{derivative_index:02d}"
                    if len(extraction_rows) > 1
                    else identity
                )
                image_target = output_root / split / "images" / f"{stem}{suffix}"
                label_target = output_root / split / "labels" / f"{stem}.txt"
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(selected["image_member"]) as source, image_target.open("wb") as target:
                    shutil.copyfileobj(source, target)
                label_target.write_bytes(archive.read(selected["label_member"]))

    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "names": {index: name for index, name in enumerate(J25_CLASSES)},
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output_root / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    source_distribution = Counter(component["source_derivatives"] for component in components)
    gates = {
        "exact_451_source_identities": len(components) == 451,
        "exact_271_train_sibling_groups": source_distribution[3] == 271,
        "exact_180_single_source_groups": source_distribution[1] == 180,
        "every_train_sibling_group_has_three_members": all(
            component["source_derivatives"] == 3
            for component in components
            if component["source_split"] == "train"
        ),
        "minimum_train_sibling_similarity_at_least_0_98": min(minimum_similarities) >= 0.98,
        "all_25_classes_in_train": set(split_class_counts["train"]) == set(range(25)),
        "all_25_classes_in_validation": set(split_class_counts["val"]) == set(range(25)),
        "all_25_classes_in_locked_test": set(split_class_counts["test"]) == set(range(25)),
        "development_root_has_no_test_images": not (output_root / "test").exists(),
        "development_yaml_has_no_test_key": "test" not in data_yaml,
        "train_siblings_retained_only_in_train": (
            not retain_train_siblings
            or (
                split_image_counts["train"] == 695
                and split_image_counts["val"] == split_identity_counts["val"]
                and split_image_counts["test"] == split_identity_counts["test"]
            )
        ),
    }
    payload = {
        "format": selected_format,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "source_archive": str(archive_path),
        "source_archive_sha256": archive_sha,
        "source_identities": len(components),
        "source_derivative_group_sizes": dict(sorted(source_distribution.items())),
        "source_identities_by_split": dict(split_identity_counts),
        "images": dict(split_image_counts),
        "extracted_images": {
            "train": split_image_counts["train"],
            "val": split_image_counts["val"],
            "test": 0,
        },
        "instances": {split: int(sum(split_class_counts[split].values())) for split in split_class_counts},
        "minimum_train_sibling_similarity": min(minimum_similarities),
        "assignment_optimizer": optimizer,
        "gates": gates,
        "locked_test_manifest_only": True,
        "test_images_extracted": False,
        "retain_train_siblings": retain_train_siblings,
        "manifest": str(output_root / manifest_name),
        "training_authorized": False,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if payload["decision"] != "PASS":
        raise RuntimeError(f"J25 source split gagal: {[k for k,v in gates.items() if not v]}")
    payload["summary"] = str(summary_path)
    return payload
