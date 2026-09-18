import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_arm_notebooks_are_parallel_safe_and_never_extract_test() -> None:
    for arm in ("D0DIRECT", "AF2DIRECT"):
        path = ROOT / f"notebooks/Coffee_Standard_J25_{arm}_Seed42_Colab.ipynb"
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        for cell in payload["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), str(path), "exec")
        assert f"ARM='{arm}'" in source
        assert "audit_j25_thesis_provenance(ARCHIVE,PROVENANCE)" in source
        assert "resolve_drive_project_root(required_relative_paths=" not in source
        assert "prepare_j25_source_split" in source
        assert "CONTRACT=DATA/'coffee_standard_j25_source_split_summary.json'" in source
        assert "run_coffee_standard_j25_af2_direct" in source
        assert "last.pt tersimpan di Drive" in source
        assert "test is never extracted" in source


def test_decision_notebook_is_training_free() -> None:
    path = ROOT / "notebooks/Coffee_Standard_J25_AF2_Direct_Decision_Colab.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "build_decision" in source
    assert "--authorize-training" not in source
    assert "Jangan buka test" in source
