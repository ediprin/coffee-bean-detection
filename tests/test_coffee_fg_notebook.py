import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "CoffeeFG_YOLO26_Colab.ipynb"
)


def _load_notebook() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    return payload, source


def test_coffee_fg_notebook_code_cells_are_valid_python() -> None:
    payload, _ = _load_notebook()
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")


def test_coffee_fg_notebook_locks_test_and_gates_refiner() -> None:
    _, source = _load_notebook()
    assert "restore_real_a0_development" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "--evaluation-split', 'val'" in source
    assert "--open-test" not in source
    assert "--models', 'D0', 'D1'" in source
    assert "RUN_REFINER = False" in source
    assert "classification_refinement_rational" in source
    assert "recommended_refiners" in source


def test_coffee_fg_notebook_persists_and_resumes() -> None:
    _, source = _load_notebook()
    assert "Coffee_Bean_Detection" in source
    assert "experiments/coffee-fg-v2" in source
    assert "A0_real.tar" in source
    assert "last.pt" in source
