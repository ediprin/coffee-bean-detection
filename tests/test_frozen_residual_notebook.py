import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks/Faruq_V3_Frozen_Residual_Colab.ipynb"


def test_frozen_residual_notebook_is_valid_resumable_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "resolve_drive_project_root" in source
    assert "force_remount=True" in source
    assert "static_frozen_residual_audit" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "--authorize-training" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "PROJECT_ROOT.mkdir" not in source
