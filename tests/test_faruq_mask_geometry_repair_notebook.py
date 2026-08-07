import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "Faruq_Mask_Geometry_Repair_Colab.ipynb"
)


def test_faruq_repair_notebook_is_test_locked_and_persistent() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    assert "os.chdir('/content')" in source
    assert "sys.modules.pop(module_name, None)" in source
    assert "userdata.get('ROBOFLOW_API_KEY')" in source
    assert "repair_faruq_mask_geometry" in source
    assert "audit_faruq_mask_geometry(REPAIRED_ROOT" in source
    assert "faruq-development-v2.tar" in source
    assert "training_executed" in source
    assert "test_images_accessed" in source
    assert "training_ready" in source
    assert "model.train" not in source
    assert "--open-test" not in source
