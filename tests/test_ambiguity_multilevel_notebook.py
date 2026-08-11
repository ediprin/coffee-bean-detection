import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks/Faruq_V3_ACMC_Static_Audit_Colab.ipynb"


def test_acmc_static_notebook_is_test_locked_and_never_trains() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    for cell in cells:
        ast.parse("".join(cell["source"]))
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "force_remount=True" in source
    assert "resolve_drive_project_root" in source
    assert "static_ambiguity_multilevel_audit" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "assert result['training_executed'] is False" in source
    assert "assert result['dataset_accessed'] is False" in source
    assert "assert result['test_images_accessed'] is False" in source
    assert ".train(" not in source
    assert "PROJECT_ROOT.mkdir" not in source
