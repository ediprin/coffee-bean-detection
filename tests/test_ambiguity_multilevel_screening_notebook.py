import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks/Faruq_V3_ACMC_Screening_Colab.ipynb"


def test_acmc_screening_notebook_is_resumable_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "force_remount=True" in source
    assert "resolve_drive_project_root" in source
    assert "ACMC1_seed42/weights/last.pt" in source
    assert "--authorize-training" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "test_images_accessed'] is False" in source
    assert "PROJECT_ROOT.mkdir" not in source
