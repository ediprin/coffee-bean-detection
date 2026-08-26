import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "B0_FRESH": "Faruq_V3_DLRBC_B0_Fresh_Seed42_Colab.ipynb",
    "LRLIN_FRESH": "Faruq_V3_DLRBC_LRLIN_Fresh_Seed42_Colab.ipynb",
    "DLRBC_FRESH": "Faruq_V3_DLRBC_Quadratic_Fresh_Seed42_Colab.ipynb",
}
DECISION_NOTEBOOK = "Faruq_V3_DLRBC_Fresh_Seed42_Decision_Colab.ipynb"


def _load(name: str) -> dict:
    return json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))


def test_each_training_arm_has_one_parallel_notebook() -> None:
    for arm, filename in NOTEBOOKS.items():
        notebook = _load(filename)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert f"ARM='{arm}'" in source
        assert "--authorize-training" in source
        assert "time.sleep(120)" in source
        assert "last.pt" in source
        assert "test" in source.lower()
        assert "progress=True" not in source


def test_all_training_notebook_code_compiles() -> None:
    for filename in (*NOTEBOOKS.values(), DECISION_NOTEBOOK):
        notebook = _load(filename)
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                ast.parse("".join(cell.get("source", [])), filename=f"{filename}:{index}")


def test_decision_notebook_does_not_train() -> None:
    notebook = _load(DECISION_NOTEBOOK)
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "build_fresh_dlrbc_decision" in source
    assert "--authorize-training" not in source
    assert ".train(" not in source
