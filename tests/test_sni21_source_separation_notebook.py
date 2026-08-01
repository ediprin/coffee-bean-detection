import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "SNI21_Source_Separation_Colab.ipynb"


def test_source_separation_notebook_is_valid_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    assert "restore_real_a0_development" in source
    assert "os.chdir('/content')" in source
    assert "separate_sni21_sources" in source
    assert "test_images_accessed" in source
    assert "training_executed" in source
    assert "assert not (COMBINED_ROOT / 'test').exists()" in source
    assert "model.train" not in source
    assert "--open-test" not in source
