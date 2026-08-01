import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "Faruq_Mask_Geometry_Audit_Colab.ipynb"
)


def test_faruq_mask_geometry_notebook_is_audit_only() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")

    assert "os.chdir('/content')" in source
    assert "sys.modules.pop(module_name, None)" in source
    assert "userdata.get('roboflow')" in source
    assert "robusta_sni_dataset-hr9ci" in source
    assert "coco-segmentation" in source
    assert "audit_faruq_mask_geometry" in source
    assert "training_executed" in source
    assert "inference_executed" in source
    assert "test_images_accessed" in source
    assert "model.train" not in source
    assert "--open-test" not in source
    assert "q4xouf9aR81dzO9HuZMi" not in source
