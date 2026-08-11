import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "Faruq_Grouped_Development_Colab.ipynb"
)


def test_faruq_grouped_notebook_is_test_locked_and_archives_only_after_gate() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    assert "faruq-development-v2.tar" in source
    assert "group_faruq_development" in source
    assert "cross_split_parent_identities" in source
    assert "cross_split_exact_hashes" in source
    assert "if summary['training_ready'] and not V3_ARCHIVE.is_file()" in source
    assert "faruq-development-v3-grouped.tar" in source
    assert "training_executed" in source
    assert "test_images_accessed" in source
    assert "model.train" not in source
    assert "--open-test" not in source
