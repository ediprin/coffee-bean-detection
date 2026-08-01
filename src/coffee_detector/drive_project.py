"""Resolve the persistent Google Drive project without creating a new root.

Colab accounts expose shared folders through different paths.  The resolver
therefore discovers the project by its committed Drive-side marker files, not
by assuming a particular shortcut name.  It deliberately never creates the
project root: a missing or ambiguous root is an error instead of an invitation
to scatter artifacts across My Drive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


PROJECT_DIRECTORY = "Coffee_Bean_Detection"
PROJECT_MARKERS = ("artifact_index.json", "PROJECT_INDEX.md")
DEFAULT_DRIVE_SEARCH_ROOTS = (
    Path("/content/drive/MyDrive"),
    Path("/content/drive/.shortcut-targets-by-id"),
)


def _candidate_rank(path: Path) -> tuple[int, int, str]:
    normalized = path.as_posix()
    if "/02_RISET_DAN_PROYEK/" in normalized:
        preference = 0
    elif normalized.startswith("/content/drive/MyDrive/"):
        preference = 1
    else:
        preference = 2
    return preference, len(path.parts), normalized


def resolve_drive_project_root(
    search_roots: Iterable[str | Path] = DEFAULT_DRIVE_SEARCH_ROOTS,
) -> Path:
    """Return the marked project root exposed to the current Colab account.

    Both ``artifact_index.json`` and ``PROJECT_INDEX.md`` must exist.  An empty
    same-named directory is ignored.  No directory is created by this function.
    """

    candidates: dict[str, Path] = {}
    for raw_root in search_roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        for marker in root.rglob(PROJECT_MARKERS[0]):
            candidate = marker.parent
            if (
                candidate.name == PROJECT_DIRECTORY
                and all((candidate / name).is_file() for name in PROJECT_MARKERS)
            ):
                candidates[candidate.as_posix()] = candidate

    if not candidates:
        raise FileNotFoundError(
            "Folder proyek Drive yang valid tidak ditemukan. Tambahkan shortcut "
            "folder Coffee_Bean_Detection yang berisi artifact_index.json dan "
            "PROJECT_INDEX.md ke My Drive; notebook tidak akan membuat root baru."
        )

    return sorted(candidates.values(), key=_candidate_rank)[0]


def require_project_artifact(project_root: str | Path, relative_path: str | Path) -> Path:
    """Resolve one exact artifact inside the marked project, or fail clearly."""

    root = Path(project_root).expanduser()
    if not all((root / marker).is_file() for marker in PROJECT_MARKERS):
        raise FileNotFoundError(f"Bukan root proyek Drive yang valid: {root}")
    artifact = root / relative_path
    if not artifact.is_file():
        raise FileNotFoundError(f"Artefak proyek tidak ditemukan: {artifact}")
    return artifact
