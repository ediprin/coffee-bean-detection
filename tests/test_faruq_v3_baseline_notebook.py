import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "Faruq_V3_YOLO26n_Baseline_Colab.ipynb"
)


def test_faruq_v3_baseline_notebook_is_validation_only_and_resumable() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    assert "faruq-development-v3-grouped.tar" in source
    assert "run_faruq_v3_baseline" in source
    assert "D0_seed42/weights/last.pt" in source
    assert "--seed', '42'" in source
    assert "--device', '0'" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "evaluation_split" in source
    assert "test_images_accessed" in source
    assert "metrics/mAP50-95(B)" in source
    assert "min(per_class, key=per_class.get)" in source
    assert "--open-test" not in source
