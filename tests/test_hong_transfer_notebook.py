import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Hong_YOLO26_Full_Transfer_Colab.ipynb"


def test_hong_notebook_is_static_first_resumable_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )

    assert "static_architecture_audit" in source
    assert source.index("static_architecture_audit") < source.index(
        "run_hong_yolo26_transfer"
    )
    assert "RUN_SCREEN = False" in source
    assert "seed=42" in source
    assert "last.pt" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "resolve_drive_project_root(required_relative_paths=" in source
