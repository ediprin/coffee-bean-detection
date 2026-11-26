"""Auliya v2 baseline bridge. Run in a separate Ultralytics 8.4.210 environment.

Preparation preserves author fold membership; no Faruq checkpoint is accepted.
This is not a claim of reproduction until the real archive has been audited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil

MODELS = {"B0": "yolo26s-seg", "B1": "yolo26s", "B2": "yolo26n"}
COUNTS = {"A": (676, 119, 1351), "B": (1506, 265, 375), "C": (1468, 258, 420)}
SPLITS = ("train", "val", "test")
META_SHA = "798bb53b0cfad35dfcfbcc463fbcb79ff4af61f9807803d6bcc65d37f7c8feaa"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_polygon(line):
    parts = line.split()
    c = int(parts[0])
    v = list(map(float, parts[1:]))
    if not 0 <= c < 21 or len(v) < 6 or len(v) % 2 or any(
        not math.isfinite(x) or not 0 <= x <= 1 for x in v
    ):
        raise ValueError("Invalid normalized polygon: " + line)
    xs, ys = v[::2], v[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Degenerate polygon: " + line)
    return c, (c, (x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0)


def prepare(a):
    from PIL import Image
    import zipfile
    if sha(a.metadata_zip) != META_SHA:
        raise ValueError("Metadata ZIP differs from audited author package")
    if a.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    with zipfile.ZipFile(a.metadata_zip) as z:
        manifest = list(csv.DictReader(z.read("metadata/split_manifest_v4.csv").decode().splitlines()))
        taxonomy = list(csv.DictReader(z.read("taxonomy/label_mapping.csv").decode().splitlines()))
        names = [r["name_21"] for r in sorted(taxonomy, key=lambda r: int(r["id_21"]))]
        assert len(names) == 21 and len(manifest) == 2148
        expected = {}
        for r in manifest:
            rel = f'{r["split"]}/images/{Path(r["file"]).with_suffix(".jpg").name}'
            if rel in expected:
                raise ValueError("Repeated manifest path: " + rel)
            if r["kelompok"] != "-":
                expected[rel] = r
        lists = {}
        for split, count in zip(SPLITS, COUNTS[a.fold]):
            raw = z.read(f"splits/v4_uji{a.fold}_{split}.txt").decode().splitlines()
            entries = [PurePosixPath(x.strip()).as_posix() for x in raw if x.strip()]
            if len(entries) != count or len(set(entries)) != count:
                raise ValueError("Unexpected fold list size: " + split)
            lists[split] = entries
    flat = [x for s in SPLITS for x in lists[s]]
    if len(flat) != len(set(flat)) or set(flat) != set(expected):
        raise ValueError("Fold membership does not match manifest")
    if any((expected[x]["kelompok"] == a.fold) != (s == "test")
           for s in SPLITS for x in lists[s]):
        raise ValueError("Held-out acquisition group mismatch")

    # Audit before writing; the exact same image membership is used for both tasks.
    hashes, records, support = {}, [], {}
    for split in SPLITS:
        support[split] = Counter()
        for rel in lists[split]:
            p = PurePosixPath(rel)
            if len(p.parts) != 3 or p.parts[1] != "images" or p.parts[0] not in ("train", "valid", "test"):
                raise ValueError("Unexpected relative path: " + rel)
            image = a.images_root / rel
            label = a.images_root / p.parts[0] / "labels" / (p.stem + ".txt")
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(f"Missing original image or label: {image} / {label}")
            with Image.open(image) as im:
                size = im.size
                orientation = im.getexif().get(274, 1)
                im.verify()
            digest = sha(image)
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"Exact image hash crosses fold splits: {rel}")
            hashes[digest] = split
            lines = [x for x in label.read_text().splitlines() if x.strip()]
            parsed = [parse_polygon(x) for x in lines]
            if len(parsed) != int(expected[rel]["n"]):
                raise ValueError("Instance count differs from author manifest: " + rel)
            support[split].update(c for c, _ in parsed)
            records.append((split, rel, image, label, parsed, size, orientation, digest))
    if sum(sum(s.values()) for s in support.values()) != 3858:
        raise ValueError("Expected 3858 annotated instances")
    # The metadata does not establish mask alignment with EXIF-transformed pixels.
    # Preserve originals and report orientations; human geometry review is required.
    a.output.mkdir(parents=True)
    for task in ("segment", "detect"):
        root = a.output / task
        for split in SPLITS:
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir()
        for split, rel, image, label, parsed, *_ in records:
            # Prefix original split to avoid collisions if basenames repeat.
            name = PurePosixPath(rel).parts[0] + "__" + image.name
            shutil.copy2(image, root / split / "images" / name)
            target = root / split / "labels" / Path(name).with_suffix(".txt")
            if task == "segment":
                shutil.copy2(label, target)
            else:
                target.write_text("".join(" ".join(map(str, box)) + "\n" for _, box in parsed))
        data = dict(path=str(root.resolve()), train="train/images", val="val/images", nc=21, names=names)
        (root / "train.yaml").write_text(json.dumps(data, indent=2))
        (root / "test.yaml").write_text(json.dumps(dict(data, test="test/images"), indent=2))
    audit = {
        "fold": a.fold, "metadata_sha256": META_SHA,
        "counts": {s: len(lists[s]) for s in SPLITS},
        "class_support": {s: dict(v) for s, v in support.items()},
        "resolution_counts": dict(Counter(str(r[5]) for r in records)),
        "exif_orientation_counts": dict(Counter(str(r[6]) for r in records)),
        "exact_cross_split_hash_overlap": 0,
        "physical_identity_exception_classes": [18, 19, 20],
        "geometry_review_required": True,
        "records": [{"split": s, "path": rel, "image_sha256": d,
                     "label_sha256": sha(lab), "size": size, "orientation": o}
                    for s, rel, _, lab, _, size, o, d in records],
    }
    (a.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print("Prepared author fold", a.fold, "; review image/polygon alignment before training.")


def run(a):
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ != "8.4.210":
        raise RuntimeError("Use isolated author environment ultralytics==8.4.210")
    audit = json.loads((a.output / "audit.json").read_text())
    if audit["fold"] != a.fold or audit["metadata_sha256"] != META_SHA:
        raise ValueError("Wrong prepared fold")
    task = "segment" if a.arm == "B0" else "detect"
    root = a.output / task
    name = f"{a.arm}_uji{a.fold}_s{a.seed}"
    project = a.output / "runs"
    folder = project / name
    kw = dict(imgsz=640, batch=16, workers=4)
    if a.device is not None:
        kw["device"] = a.device
    if a.action == "train":
        if not a.geometry_reviewed:
            raise RuntimeError("Explicit geometry review is required")
        if folder.exists():
            raise FileExistsError("Run already exists; do not overwrite or silently resume")
        YOLO(MODELS[a.arm] + ".pt").train(
            data=str(root / "train.yaml"), epochs=80, patience=25,
            optimizer="AdamW", lr0=0.001, weight_decay=0.0005, cos_lr=True,
            seed=a.seed, hsv_h=0.015, hsv_s=0.4, hsv_v=0.25, translate=0.1,
            scale=0.2, perspective=0.0005, mosaic=0.3, fliplr=0.5, mixup=0.0,
            project=str(project), name=name, exist_ok=False, **kw)
        best = folder / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError("Training did not produce best.pt")
        (folder / "baseline_complete.json").write_text(json.dumps({
            "model": MODELS[a.arm], "seed": a.seed, "fold": a.fold,
            "best_sha256": sha(best), "audit_sha256": sha(a.output / "audit.json"),
            "ultralytics": ultralytics.__version__,
        }, indent=2))
    else:
        if a.split == "test" and not a.frozen_test:
            raise RuntimeError("Test requires explicit frozen-test confirmation")
        best = folder / "weights" / "best.pt"
        done = json.loads((folder / "baseline_complete.json").read_text())
        if done["best_sha256"] != sha(best) or done["audit_sha256"] != sha(a.output / "audit.json"):
            raise ValueError("Checkpoint or dataset audit changed")
        report = folder / f"{a.split}_metrics.json"
        if report.exists():
            raise FileExistsError("Evaluation already recorded")
        result = YOLO(str(best)).val(data=str(root / ("test.yaml" if a.split == "test" else "train.yaml")),
            split=a.split, project=str(project), name=name+"_"+a.split, exist_ok=False, **kw)
        def metrics(m):
            per = {int(c): float(m.ap[i]) for i, c in enumerate(m.ap_class_index)}
            clean = [v for c, v in per.items() if c < 18]
            return {"map5095": float(m.map), "map50": float(m.map50),
                    "per_class_ap5095": per, "classes_missing": sorted(set(range(21))-set(per)),
                    "clean18_ap5095": sum(clean)/18 if set(range(18)) <= set(per) else None}
        out = {"arm": a.arm, "fold": a.fold, "seed": a.seed, "split": a.split,
               "box": metrics(result.box)}
        if task == "segment":
            out["mask"] = metrics(result.seg)
        report.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "train", "evaluate"])
    p.add_argument("--fold", choices=list(COUNTS), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--images-root", type=Path)
    p.add_argument("--metadata-zip", type=Path)
    p.add_argument("--arm", choices=list(MODELS), default="B2")
    p.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--device")
    p.add_argument("--geometry-reviewed", action="store_true")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--frozen-test", action="store_true")
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.action == "prepare":
        if a.images_root is None or a.metadata_zip is None:
            p.error("prepare requires --images-root and --metadata-zip")
        prepare(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
