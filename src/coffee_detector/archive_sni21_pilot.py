from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path

from .archive_vadcp import pack_training_arm, restore_training_arm


BUNDLE_FORMAT = "coffee_detector.sni21_pilot_bundle.v1"


def _pack_files(
    root: Path,
    archive: Path,
    files: list[Path],
    *,
    progress_every: int = 500,
) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    started = time.perf_counter()
    print(f"PACK {root.name}: mulai 0/{len(files)} file", flush=True)
    with tarfile.open(partial, "w") as bundle:
        for index, path in enumerate(files, 1):
            bundle.add(path, arcname=path.relative_to(root), recursive=False)
            if index % max(1, progress_every) == 0 or index == len(files):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-8)
                eta = (len(files) - index) / max(rate, 1e-8)
                print(
                    f"  pack {index}/{len(files)} | ETA {eta / 60:.1f} menit",
                    flush=True,
                )
    os.replace(partial, archive)
    print(
        f"ARCHIVE SIAP: {archive} ({archive.stat().st_size / 2**20:.1f} MB)",
        flush=True,
    )
    return archive


def pack_real_a0(
    root: str | Path,
    archive: str | Path,
    *,
    progress_every: int = 500,
) -> Path:
    """Persist the audited A0 split so a GPU runtime need not rebuild it."""
    root = Path(root).expanduser().resolve()
    archive = Path(archive).expanduser().resolve()
    marker = archive.with_suffix(archive.suffix + ".json")
    if archive.is_file() and marker.is_file():
        print(f"SKIP ARCHIVE A0: {archive}", flush=True)
        return archive
    required_files = [
        root / "data.yaml",
        root / "audit.json",
        root / "post_materialization_audit.json",
    ]
    required_dirs = [
        root / split / kind
        for split in ("train", "val", "test")
        for kind in ("images", "labels")
    ]
    missing = [
        str(path)
        for path in [*required_files, *required_dirs]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("A0 belum lengkap: " + ", ".join(missing))
    files = required_files + sorted(
        path
        for folder in required_dirs
        for path in folder.rglob("*")
        if path.is_file()
    )
    _pack_files(root, archive, files, progress_every=progress_every)
    marker.write_text(
        json.dumps(
            {
                "format": "coffee_detector.sni21_a0_archive.v1",
                "archive": str(archive),
                "files": len(files),
                "bytes": archive.stat().st_size,
                "splits": ["train", "val", "test"],
                "test_status": "stored but locked",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return archive


def _restore_archive(
    archive: Path,
    root: Path,
    *,
    progress_every: int = 500,
) -> Path:
    if (
        (root / "data.yaml").is_file()
        and (root / "post_materialization_audit.json").is_file()
    ):
        print(f"SKIP RESTORE A0: {root}", flush=True)
        return root
    if not archive.is_file():
        raise FileNotFoundError(f"Archive A0 tidak ditemukan: {archive}")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Folder restore A0 tidak kosong: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tarfile.open(archive, "r") as bundle:
        members = bundle.getmembers()
        print(f"RESTORE A0: mulai 0/{len(members)} file", flush=True)
        for index, member in enumerate(members, 1):
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Path archive A0 tidak aman: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    f"Tipe anggota archive A0 tidak aman: {member.name}"
                )
            try:
                bundle.extract(member, root, filter="data")
            except TypeError:  # Python 3.10 compatibility
                bundle.extract(member, root)
            if index % max(1, progress_every) == 0 or index == len(members):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-8)
                eta = (len(members) - index) / max(rate, 1e-8)
                print(
                    f"  restore A0 {index}/{len(members)} | "
                    f"ETA {eta / 60:.1f} menit",
                    flush=True,
                )
    if not (root / "data.yaml").is_file():
        raise RuntimeError(f"Restore A0 tidak lengkap: {root}")
    return root


def pack_sni21_pilot_bundle(
    a0_root: str | Path,
    setup_root: str | Path,
    bundle_root: str | Path,
) -> dict:
    """Persist A0 and the paired 200-scene pilot arms to Google Drive."""
    a0_root = Path(a0_root).expanduser().resolve()
    setup_root = Path(setup_root).expanduser().resolve()
    bundle_root = Path(bundle_root).expanduser().resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    setup = json.loads(
        (setup_root / "setup_summary.json").read_text(encoding="utf-8")
    )
    if int(setup.get("synthetic_images_per_arm", -1)) != 200:
        raise RuntimeError("Bundle hanya menerima pilot 200 scene per arm")
    if setup.get("training_ready") is not True:
        raise RuntimeError("Setup pilot belum menyatakan training_ready")

    archives = {
        "A0": pack_real_a0(a0_root, bundle_root / "A0_real.tar"),
        "A1": pack_training_arm(
            setup_root / "A1", bundle_root / "A1_train.tar"
        ),
        "A2": pack_training_arm(
            setup_root / "A2", bundle_root / "A2_train.tar"
        ),
    }
    payload = {
        "format": BUNDLE_FORMAT,
        "seed": int(setup["seed"]),
        "synthetic_images_per_arm": 200,
        "training_ready": True,
        "test_opened": False,
        "archives": {
            code: {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
            for code, path in archives.items()
        },
    }
    manifest = bundle_root / "bundle_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["manifest"] = str(manifest)
    print(f"BUNDLE PILOT AMAN DI DRIVE: {manifest}", flush=True)
    return payload


def restore_sni21_pilot_bundle(
    bundle_root: str | Path,
    work_root: str | Path = "/content",
) -> dict:
    """Restore the persistent pilot bundle into its protocol-fixed paths."""
    bundle_root = Path(bundle_root).expanduser().resolve()
    work_root = Path(work_root).expanduser().resolve()
    manifest_path = bundle_root / "bundle_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != BUNDLE_FORMAT:
        raise RuntimeError(f"Format bundle tidak dikenal: {manifest_path}")
    if payload.get("training_ready") is not True:
        raise RuntimeError(f"Bundle belum siap training: {manifest_path}")
    archives = {
        code: Path(row["path"]).expanduser().resolve()
        for code, row in payload["archives"].items()
    }
    for code, archive in archives.items():
        if not archive.is_file():
            raise FileNotFoundError(f"Archive {code} hilang: {archive}")

    a0_root = _restore_archive(
        archives["A0"], work_root / "sni21-fullscene-v1"
    )
    setup_root = work_root / "sni21-vadcp-pilot"
    arms = {
        "A1": restore_training_arm(
            archives["A1"], setup_root / "A1"
        ),
        "A2": restore_training_arm(
            archives["A2"], setup_root / "A2"
        ),
    }
    result = {
        "a0_root": str(a0_root),
        "arms": {
            code: {
                "root": str(root),
                "audit": str(root / "metadata" / "vadcp_audit.json"),
            }
            for code, root in arms.items()
        },
        "training_ready": True,
        "test_accessed": False,
    }
    print("RESTORE PILOT SELESAI. TEST TETAP TERKUNCI.", flush=True)
    return result
