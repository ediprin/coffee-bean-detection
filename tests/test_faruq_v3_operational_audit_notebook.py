import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "Faruq_V3_Operational_Audit_Colab.ipynb"
)


def test_operational_audit_notebook_is_validation_only_and_training_free() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    assert "resolve_drive_project_root" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "faruq_v3_operational_audit" in source
    assert "--split', 'val'" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "training_executed" in source
    assert "test_images_accessed" in source
    assert "model.train" not in source
    assert "--open-test" not in source

