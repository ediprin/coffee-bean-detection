import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks/Faruq_V3_Multilevel_Head_Training_Colab.ipynb"
)


def _source() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        ast.parse("".join(cell["source"]))
    return "\n".join("".join(cell["source"]) for cell in cells)


def test_training_notebook_is_resumable_and_test_locked() -> None:
    source = _source()
    assert "resolve_drive_project_root" in source
    assert "faruq-development-v3-grouped.tar" in source
    assert "static_audit.json" in source
    assert "--authorize-training" in source
    assert "MHC0" in source and "MHF1" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "force_remount=True" in source
    assert "PROJECT_ROOT.mkdir" not in source
