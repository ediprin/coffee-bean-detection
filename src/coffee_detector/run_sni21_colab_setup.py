from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

from .audit_dataset import audit_dataset
from .prepare_sni_fullscene import prepare_sni_fullscene
from .run_sni21_vadcp_setup import run_sni21_vadcp_setup


PROFILES = {
    "smoke": {"synthetic_images": 2, "visual_samples": 2},
    "full": {"synthetic_images": 2000, "visual_samples": 12},
}


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} bukan JSON object: {path}")
    return payload


def extract_archive(
    archive: str | Path,
    target: str | Path,
    *,
    progress_every: int = 500,
) -> Path:
    """Extract one trusted dataset TAR with traversal checks and resume marker."""
    archive = Path(archive).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Arsip tidak ditemukan: {archive}")
    marker = target / ".extract_complete"
    if marker.is_file():
        print(f"REUSE EXTRACT: {target}", flush=True)
        return target
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(
            f"Ekstraksi parsial ditemukan: {target}. "
            "Hapus folder tersebut atau gunakan work-root baru."
        )
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tarfile.open(archive, "r") as bundle:
        members = bundle.getmembers()
        print(f"EXTRACT {archive.name}: mulai 0/{len(members)}", flush=True)
        for index, member in enumerate(members, 1):
            destination = (target / member.name).resolve()
            if destination != target and target not in destination.parents:
                raise RuntimeError(f"Path TAR tidak aman: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(
                    f"Tipe anggota TAR tidak didukung: {member.name}"
                )
            try:
                bundle.extract(member, target, filter="data")
            except TypeError:  # Python 3.10 compatibility
                bundle.extract(member, target)
            if index % max(1, progress_every) == 0 or index == len(members):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-8)
                eta = (len(members) - index) / max(rate, 1e-8)
                print(
                    f"  extract {index}/{len(members)} | "
                    f"ETA {eta / 60:.1f} menit",
                    flush=True,
                )
    marker.write_text("complete\n", encoding="utf-8")
    return target


def find_coco_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidates = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    for candidate in candidates:
        if all(
            (candidate / split / "_annotations.coco.json").is_file()
            for split in ("train", "valid", "test")
        ):
            return candidate
    raise FileNotFoundError(
        f"Root COCO train/valid/test tidak ditemukan di {root}"
    )


def prepare_or_reuse_a0(
    adrian_root: str | Path,
    faruq_root: str | Path,
    crop_manifest: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
) -> tuple[Path, dict]:
    output_root = Path(output_root).expanduser().resolve()
    materialization_path = output_root / "audit.json"
    if materialization_path.is_file():
        materialization = _read_json(
            materialization_path, "Audit materialisasi A0"
        )
        if not materialization.get("training_ready"):
            raise RuntimeError(f"A0 tersimpan tetapi belum siap: {output_root}")
        if materialization.get("test_locked") is not True:
            raise RuntimeError(f"A0 tidak mengunci test: {output_root}")
        print(f"REUSE A0: {output_root}", flush=True)
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise RuntimeError(
                f"A0 parsial ditemukan; gunakan work-root baru: {output_root}"
            )
        print("MATERIALIZE A0 SNI-21...", flush=True)
        materialization = prepare_sni_fullscene(
            adrian_root,
            faruq_root,
            crop_manifest,
            output_root,
            seed=seed,
            link_mode="hardlink",
        )
        if not materialization.get("training_ready"):
            raise RuntimeError("Materialisasi A0 selesai tetapi belum siap")

    post_path = output_root / "post_materialization_audit.json"
    if post_path.is_file():
        post = _read_json(post_path, "Audit pascamaterialisasi A0")
        print(f"REUSE POST AUDIT: {post_path}", flush=True)
    else:
        print("POST-AUDIT A0...", flush=True)
        post = audit_dataset(
            output_root,
            post_path,
            near_threshold=-1,
        )
    if not post.get("safe_for_training"):
        raise RuntimeError(f"Post-audit A0 belum aman: {post_path}")
    if int(post.get("cross_split_duplicate_components", -1)) != 0:
        raise RuntimeError(f"Post-audit A0 menemukan leakage: {post_path}")
    print(f"A0 SIAP: {post.get('images_by_split', {})}", flush=True)
    return output_root, post


def run_sni21_colab_setup(
    adrian_archive: str | Path,
    faruq_archive: str | Path,
    crop_dataset_root: str | Path,
    work_root: str | Path = "/content",
    *,
    profile: str = "smoke",
    seed: int = 42,
) -> dict:
    """One-call Colab setup. It never trains and never evaluates test."""
    if profile not in PROFILES:
        raise ValueError(
            f"Profile tidak dikenal: {profile}; pilih {sorted(PROFILES)}"
        )
    crop_root = Path(crop_dataset_root).expanduser().resolve()
    for required in ("manifest.csv", "audit.json", "complete.json"):
        if not (crop_root / required).is_file():
            raise FileNotFoundError(
                f"Crop package belum lengkap: {crop_root / required}"
            )
    if not (crop_root / "shards").is_dir():
        raise FileNotFoundError(
            f"Folder shard crop tidak ditemukan: {crop_root / 'shards'}"
        )

    work_root = Path(work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    raw_root = work_root / "sni21-raw"
    a0_root = work_root / "sni21-fullscene-v1"
    library_root = work_root / "sni21-object-library"
    shard_cache = work_root / "sni21-shard-cache"
    setup_root = work_root / f"sni21-vadcp-{profile}"

    print("=== SNI-21 COLAB SETUP ===", flush=True)
    print(f"PROFILE  : {profile}", flush=True)
    print("TRAINING : TIDAK AKAN DIJALANKAN", flush=True)
    print("TEST     : TIDAK AKAN DIAKSES", flush=True)
    print("[1/4] Ekstraksi/reuse sumber...", flush=True)
    adrian_extract = extract_archive(
        adrian_archive, raw_root / "adrian_detection"
    )
    faruq_extract = extract_archive(
        faruq_archive, raw_root / "faruq_segmentation"
    )
    adrian_root = find_coco_root(adrian_extract)
    faruq_root = find_coco_root(faruq_extract)

    print("[2/4] Materialisasi/reuse A0...", flush=True)
    a0_root, post = prepare_or_reuse_a0(
        adrian_root,
        faruq_root,
        crop_root / "manifest.csv",
        a0_root,
        seed=seed,
    )

    profile_config = PROFILES[profile]
    print(f"[3/4] Setup A1/A2 profile={profile}...", flush=True)
    setup = run_sni21_vadcp_setup(
        a0_root,
        crop_root,
        setup_root,
        synthetic_images=profile_config["synthetic_images"],
        seed=seed,
        objects_min=220,
        objects_max=300,
        canvas_size=1024,
        max_normal_assets=300,
        max_defect_assets_per_class=60,
        shard_cache_root=shard_cache,
        object_library_root=library_root,
        visual_samples=profile_config["visual_samples"],
    )

    result = {
        "format": "coffee_detector.sni21_colab_setup.v1",
        "profile": profile,
        "seed": seed,
        "work_root": str(work_root),
        "a0_root": str(a0_root),
        "a0_images": post.get("images_by_split", {}),
        "setup_root": str(setup_root),
        "setup_summary": str(setup_root / "setup_summary.json"),
        "training_ready": setup["training_ready"],
        "training_executed": False,
        "test_accessed": False,
        "arms": setup["arms"],
    }
    result_path = work_root / f"sni21-colab-{profile}-summary.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[4/4] Ringkasan selesai.", flush=True)
    print(f"TRAINING_READY : {result['training_ready']}", flush=True)
    print("TRAINING       : BELUM DIJALANKAN", flush=True)
    print("TEST           : TIDAK DIAKSES", flush=True)
    print(f"SUMMARY        : {result_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-call resumable SNI-21 Colab setup; no training."
    )
    parser.add_argument("--adrian-archive", required=True)
    parser.add_argument("--faruq-archive", required=True)
    parser.add_argument("--crop-dataset-root", required=True)
    parser.add_argument("--work-root", default="/content")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_sni21_colab_setup(
        args.adrian_archive,
        args.faruq_archive,
        args.crop_dataset_root,
        args.work_root,
        profile=args.profile,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
