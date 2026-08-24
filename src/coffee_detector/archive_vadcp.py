from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path


def _training_files(root: Path) -> list[Path]:
    required = (root / "data.yaml", root / "train", root / "metadata")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Artefak arm belum lengkap: " + ", ".join(missing))
    files = [root / "data.yaml"]
    files.extend(
        sorted(
            path
            for folder in required[1:]
            for path in folder.rglob("*")
            if path.is_file()
        )
    )
    return files


def pack_training_arm(
    root: str | Path,
    archive: str | Path,
    *,
    progress_every: int = 250,
) -> Path:
    root = Path(root).expanduser().resolve()
    archive = Path(archive).expanduser().resolve()
    marker = archive.with_suffix(archive.suffix + ".json")
    if archive.is_file() and marker.is_file():
        print(f"SKIP ARCHIVE: {archive}", flush=True)
        return archive
    files = _training_files(root)
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    started = time.perf_counter()
    print(f"PACK {root.name}: mulai 0/{len(files)} file", flush=True)
    with tarfile.open(partial, "w") as bundle:
        for index, path in enumerate(files, 1):
            bundle.add(path, arcname=path.relative_to(root), recursive=False)
            if index % max(progress_every, 1) == 0 or index == len(files):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-8)
                eta = (len(files) - index) / max(rate, 1e-8)
                print(
                    f"PACK {root.name}: {index}/{len(files)} | "
                    f"{rate:.1f} file/s | ETA {eta / 60:.1f} menit",
                    flush=True,
                )
    os.replace(partial, archive)
    payload = {
        "format": "coffee_detector.vadcp_training_archive.v1",
        "source_root": str(root),
        "archive": str(archive),
        "files": len(files),
        "bytes": archive.stat().st_size,
        "includes": ["data.yaml", "train", "metadata"],
        "excludes": ["val", "test"],
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"ARCHIVE SIAP: {archive} ({archive.stat().st_size / 2**20:.1f} MB)", flush=True)
    return archive


def restore_training_arm(
    archive: str | Path,
    root: str | Path,
    *,
    progress_every: int = 250,
) -> Path:
    archive = Path(archive).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    marker = root / "metadata" / "generation_manifest.json"
    if marker.is_file() and (root / "data.yaml").is_file():
        print(f"SKIP RESTORE: {root}", flush=True)
        return root
    if not archive.is_file():
        raise FileNotFoundError(f"Archive tidak ditemukan: {archive}")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Folder restore tidak kosong: {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tarfile.open(archive, "r") as bundle:
        members = bundle.getmembers()
        print(f"RESTORE {root.name}: mulai 0/{len(members)} file", flush=True)
        for index, member in enumerate(members, 1):
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Path archive tidak aman: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Tipe anggota archive tidak aman: {member.name}")
            try:
                bundle.extract(member, root, filter="fully_trusted")
            except TypeError:  # Python 3.10 compatibility
                bundle.extract(member, root)
            if index % max(progress_every, 1) == 0 or index == len(members):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-8)
                eta = (len(members) - index) / max(rate, 1e-8)
                print(
                    f"RESTORE {root.name}: {index}/{len(members)} | "
                    f"{rate:.1f} file/s | ETA {eta / 60:.1f} menit",
                    flush=True,
                )
    if not marker.is_file() or not (root / "data.yaml").is_file():
        raise RuntimeError(f"Restore tidak lengkap: {root}")
    print(f"RESTORE SIAP: {root}", flush=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist/restore train synthetic VA-DCP.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--root", required=True)
    pack.add_argument("--archive", required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--archive", required=True)
    restore.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "pack":
        pack_training_arm(args.root, args.archive)
    else:
        restore_training_arm(args.archive, args.root)


if __name__ == "__main__":
    main()
