from pathlib import Path

import pytest

from coffee_detector.drive_project import (
    require_project_artifact,
    resolve_drive_project_root,
)


NOTEBOOKS = Path(__file__).parents[1] / "notebooks"
PERSISTENT_NOTEBOOKS = (
    "Faruq_Mask_Geometry_Audit_Colab.ipynb",
    "Faruq_Mask_Geometry_Repair_Colab.ipynb",
    "Faruq_Grouped_Development_Colab.ipynb",
    "Faruq_V3_YOLO26n_Baseline_Colab.ipynb",
    "SNI21_Source_Separation_Colab.ipynb",
    "SNI21_Source_Domain_Evaluation_Colab.ipynb",
)


def _marked_project(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "artifact_index.json").write_text("{}", encoding="utf-8")
    (path / "PROJECT_INDEX.md").write_text("# Project", encoding="utf-8")
    return path


def test_resolver_ignores_unmarked_same_named_directory(tmp_path: Path) -> None:
    (tmp_path / "Coffee_Bean_Detection").mkdir()

    with pytest.raises(FileNotFoundError, match="tidak akan membuat root baru"):
        resolve_drive_project_root([tmp_path])


def test_resolver_prefers_canonical_research_tree(tmp_path: Path) -> None:
    _marked_project(tmp_path / "Coffee_Bean_Detection")
    canonical = _marked_project(
        tmp_path / "02_RISET_DAN_PROYEK" / "Coffee_Bean_Detection"
    )

    assert resolve_drive_project_root([tmp_path]) == canonical


def test_require_project_artifact_uses_exact_relative_path(tmp_path: Path) -> None:
    project = _marked_project(tmp_path / "Coffee_Bean_Detection")
    artifact = project / "bundles" / "dataset.tar"
    artifact.parent.mkdir()
    artifact.write_bytes(b"data")

    assert require_project_artifact(project, "bundles/dataset.tar") == artifact
    with pytest.raises(FileNotFoundError, match="Artefak proyek"):
        require_project_artifact(project, "bundles/missing.tar")


def test_persistent_notebooks_never_create_or_hardcode_project_root() -> None:
    forbidden = (
        "Path('/content/drive/MyDrive/Coffee_Bean_Detection')",
        'Path("/content/drive/MyDrive/Coffee_Bean_Detection")',
        "PROJECT_ROOT.mkdir(",
    )
    for name in PERSISTENT_NOTEBOOKS:
        source = (NOTEBOOKS / name).read_text(encoding="utf-8")
        assert "resolve_drive_project_root" in source, name
        for token in forbidden:
            assert token not in source, f"{name}: {token}"
