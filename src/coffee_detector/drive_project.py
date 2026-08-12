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


def _contains_required_artifacts(
    candidate: Path, required: tuple[Path, ...]
) -> bool:
    """Return whether one project root contains every caller-required file."""

    return all((candidate / relative).is_file() for relative in required)


def resolve_drive_project_root(
    search_roots: Iterable[str | Path] = DEFAULT_DRIVE_SEARCH_ROOTS,
    *,
    required_relative_paths: Iterable[str | Path] = (),
) -> Path:
    """Return the marked project root exposed to the current Colab account.

    Both ``artifact_index.json`` and ``PROJECT_INDEX.md`` must exist.  An empty
    same-named directory is ignored.  No directory is created by this function.
    """

    roots = [Path(raw_root).expanduser() for raw_root in search_roots]
    required = tuple(Path(item) for item in required_relative_paths)
    candidates: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        # DriveFS may expose a shortcut for direct access but refuse to descend
        # into it during rglob().  Probe the canonical shortcut path explicitly
        # before relying on recursive discovery.
        direct = root / PROJECT_DIRECTORY
        if direct.is_dir() and (
            all((direct / name).is_file() for name in PROJECT_MARKERS)
            or (required and _contains_required_artifacts(direct, required))
        ):
            candidates[direct.as_posix()] = direct
        for marker in root.rglob(PROJECT_MARKERS[0]):
            candidate = marker.parent
            if all((candidate / name).is_file() for name in PROJECT_MARKERS):
                candidates[candidate.as_posix()] = candidate

    # Marker files establish that a directory is a project root, but they do
    # not establish that this account sees a complete copy of the project.
    # This matters when MyDrive contains an old/incomplete directory while a
    # shared shortcut exposes the complete project.  A caller that supplies
    # required paths must receive one root containing *all* of them.
    if required:
        candidates = {
            key: candidate
            for key, candidate in candidates.items()
            if _contains_required_artifacts(candidate, required)
        }

    # Google Drive FUSE can retain a cached shortcut view after files are moved
    # into the shared folder.  In that case marker files may lag behind larger
    # artifacts that were already visible to the runtime.  Exact, caller-supplied
    # artifacts provide a safe fallback without accepting an empty same-named
    # folder or creating a replacement project root.
    if not candidates and required:
        for anchor in required:
            for root in roots:
                if not root.is_dir():
                    continue
                for match in root.rglob(anchor.name):
                    if not match.is_file():
                        continue
                    candidate = match
                    for _ in anchor.parts:
                        candidate = candidate.parent
                    if _contains_required_artifacts(candidate, required):
                        candidates[candidate.as_posix()] = candidate

    if not candidates:
        raise FileNotFoundError(
            "Folder proyek Drive yang valid tidak ditemukan. Tambahkan shortcut "
            "folder Coffee_Bean_Detection yang berisi artifact_index.json dan "
            "PROJECT_INDEX.md ke My Drive. Jika shortcut baru dipindahkan, force-remount "
            "Drive; notebook tidak akan membuat root baru."
        )

    return sorted(candidates.values(), key=_candidate_rank)[0]


def require_project_artifact(project_root: str | Path, relative_path: str | Path) -> Path:
    """Resolve one exact artifact inside the marked project, or fail clearly."""

    root = Path(project_root).expanduser()
    artifact = root / relative_path
    if not artifact.is_file():
        raise FileNotFoundError(f"Artefak proyek tidak ditemukan: {artifact}")
    return artifact
