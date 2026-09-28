"""Resolve shared-Drive inputs for the DefectosCafeVerde routability audit."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


ARCHIVE_NAME = "defectoscafeverde-grouped-physical-v1.tar"
ARCHIVE_BYTES = 686_474_752
ARCHIVE_SHA256 = "53fb2233f1f0d1c77cb24eca2d720f86e0a16835b8a69f4e8f3176fae1aacef2"
CHUNK_IDS = (
    "1pAwWlkABja1Epaw1U8Uxre2JorfXwE3P", "1UoZkqJbHstFcA9WeXW0KNeisGasqkci6",
    "1kY_xwFOHW17eQQ7x1TDgM6F4ySRyg2gD", "1HzW5oJnMOoO7HwWY4Ic-dK8SE-3uy1rH",
    "1ydHf2EqZIJiA_I0udSL68QzbdscRvKtu", "1ip1VVRQ6n1NW7Zr5DCU3iajy54Zgsd2B",
    "10cobK2wuX_VsqOd8DfSiwHZcfoXVj4YO", "12NpEYi7qfIzzHzItlrw1hFb3DB22N2DQ",
    "15GWk7s87nQGlEBCxe3PU7lxYJsXG1rQY", "1miE6MBbwkCme0HX1JF08uDkBDjot-DVZ",
    "1pF9TmGrcK56reCN6zRF4xzIaxtZqD6sk",
)
RESULT_REL = {
    "D0DIRECT": "experiments/defectoscafeverde-cwcf-direct-v1/val_reports/D0DIRECT_seed42_result.json",
    "AF2DIRECT": "experiments/defectoscafeverde-af2-direct-v1/val_reports/AF2DIRECT_seed42_result.json",
    "DCWCF1": "experiments/defectoscafeverde-cwcf-direct-v1/val_reports/DCWCF1_seed42_result.json",
    "LIFRPF1": "experiments/defectoscafeverde-lif-rpf-v1/val_reports/LIFRPF1_seed42_result.json",
    "RAFC1": "experiments/defectoscafeverde-rafc-v1/val_reports/RAFC1_seed42_result.json",
}
CHECKPOINT_REL = {
    "D0DIRECT": "experiments/defectoscafeverde-cwcf-direct-v1/D0DIRECT/D0DIRECT_seed42/weights/best.pt",
    "AF2DIRECT": "experiments/defectoscafeverde-af2-direct-v1/AF2DIRECT/AF2DIRECT_seed42/weights/best.pt",
    "DCWCF1": "experiments/defectoscafeverde-cwcf-direct-v1/DCWCF1/DCWCF1_seed42/weights/best.pt",
    "LIFRPF1": "experiments/defectoscafeverde-lif-rpf-v1/LIFRPF1/LIFRPF1_seed42/weights/best.pt",
    "RAFC1": "experiments/defectoscafeverde-rafc-v1/RAFC1/RAFC1_seed42/weights/best.pt",
}
RESULT_IDS = {
    "D0DIRECT": "14syRwc1tGNs2Lq27q-1X6B0VQwv88qWQ", "AF2DIRECT": "1VBNlbUR_0x9q3HAgpauYN0EMoe8jl86x",
    "DCWCF1": "1HDVDNQtqEoOn3bSetgqdo1Fvwmqxuv4O", "LIFRPF1": "16fel4LqgF4Y4DkgaHom7NQlWv1TDmdKc",
    "RAFC1": "1OWaWDbxYnc5zl6_W8Ipq9SJhid6Mtd-s",
}
CHECKPOINT_IDS = {
    "D0DIRECT": "1maWHfcfTkM7CB5V4NpBP9WlI-D74TuKB", "AF2DIRECT": "1sWK4iapXN1G8QT5zq0qSI8EpKOfcFaEW",
    "DCWCF1": "1Q5JIdWvNVwqAxNkoh6S9Du1ZYK6NIgqE", "LIFRPF1": "14D_cuU2LJuqZXrjYYVSdKZkD-0Iyvz6g",
    "RAFC1": "1vQb-G6shm7KogNU0TErwzdjAEyqHJmCb",
}
CHECKPOINT_SHA = {
    "D0DIRECT": "895ed890344ea618bcb693a665e1c3791dd78179eb6cc0a13f8c545c0dbfee2c",
    "AF2DIRECT": "f843ca03b76f272010e32f3f6afa4a2994ac3b5d1c3a1b3fc5ee2bcc5021dbe3",
    "DCWCF1": "28c6088f1ad6b68c7c093e15d1d0594427ed311224bb5c984dd6eb43df13a4c4",
    "LIFRPF1": "6b093fa71917a1fce930d731a1e1af727edf3303c51c63c8ac7319480ae7a4ed",
    "RAFC1": "d4574651998aef26ca8bb48437e3495abd37da81f5e55e561909f65aa1269dac",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _service():
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build

    auth.authenticate_user()
    credentials, _ = google.auth.default()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _download(service, file_id: str, target: Path, expected_sha: str | None = None) -> Path:
    from googleapiclient.http import MediaIoBaseDownload

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (expected_sha is None or _sha256(target) == expected_sha):
        return target
    target.unlink(missing_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    try:
        with target.open("wb") as stream:
            loader = MediaIoBaseDownload(stream, request, chunksize=16 * 1024 * 1024)
            done = False
            while not done:
                _, done = loader.next_chunk()
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if expected_sha and _sha256(target) != expected_sha:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"SHA Drive API tidak cocok: {target.name}")
    return target


def prepare_inputs(work_root: str | Path = "/content", drive_root: str | Path = "/content/drive"):
    work = Path(work_root)
    drive = Path(drive_root)
    output_project = drive / "MyDrive" / "Coffee_Bean_Detection"
    output_project.mkdir(parents=True, exist_ok=True)
    cache = work / "defectoscafeverde-routability-inputs"
    cache.mkdir(parents=True, exist_ok=True)
    mount_roots = [drive / "MyDrive", *(drive / ".shortcut-targets-by-id").glob("*")]
    project_roots = []
    for root in mount_roots:
        for candidate in (root, root / "Coffee_Bean_Detection"):
            if candidate.exists() and candidate not in project_roots:
                project_roots.append(candidate)

    service = None

    def resolve(relative: str, file_id: str, target: Path, expected_sha=None):
        nonlocal service
        for root in project_roots:
            candidate = root / relative
            if candidate.is_file() and (expected_sha is None or _sha256(candidate) == expected_sha):
                return candidate, "mounted"
        service = service or _service()
        return _download(service, file_id, target, expected_sha), "drive_api"

    results, checkpoints, sources = {}, {}, {}
    for model in RESULT_REL:
        results[model], result_source = resolve(
            RESULT_REL[model], RESULT_IDS[model], cache / f"{model}_result.json"
        )
        checkpoints[model], checkpoint_source = resolve(
            CHECKPOINT_REL[model], CHECKPOINT_IDS[model], cache / f"{model}_best.pt", CHECKPOINT_SHA[model]
        )
        sources[model] = {"result": result_source, "checkpoint": checkpoint_source}

    archive = None
    for root in project_roots:
        for candidate in (root / ARCHIVE_NAME, root / "bundles" / ARCHIVE_NAME):
            if candidate.is_file() and candidate.stat().st_size == ARCHIVE_BYTES and _sha256(candidate) == ARCHIVE_SHA256:
                archive = candidate
                archive_source = "mounted"
                break
        if archive is not None:
            break
    if archive is None:
        service = service or _service()
        parts = []
        for index, file_id in enumerate(CHUNK_IDS):
            expected_size = 67_108_864 if index < 10 else 15_386_112
            part = cache / f"{ARCHIVE_NAME}.chunk-{index:03d}"
            if not part.is_file() or part.stat().st_size != expected_size:
                part.unlink(missing_ok=True)
                _download(service, file_id, part)
                print(f"DOWNLOAD BUNDLE: {index + 1}/{len(CHUNK_IDS)}", flush=True)
            parts.append(part)
        if sum(path.stat().st_size for path in parts) != ARCHIVE_BYTES:
            raise RuntimeError("Ukuran bundle Drive API tidak lengkap")
        archive = cache / ARCHIVE_NAME
        if not archive.is_file() or _sha256(archive) != ARCHIVE_SHA256:
            archive.unlink(missing_ok=True)
            with archive.open("wb") as target:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if _sha256(archive) != ARCHIVE_SHA256:
            raise RuntimeError("SHA bundle hasil gabung tidak cocok")
        archive_source = "drive_api_chunks"
    return output_project, archive, results, checkpoints, {
        "archive": archive_source,
        "artifacts": sources,
    }

