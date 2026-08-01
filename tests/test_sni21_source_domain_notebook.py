import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "SNI21_Source_Domain_Evaluation_Colab.ipynb"


def test_source_domain_notebook_is_validation_only() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    assert "restore_real_a0_development" in source
    assert "separate_sni21_sources" in source
    assert "evaluate_sni21_source_domains" in source
    assert "A0_seed42/weights/best.pt" in source
    assert "torch.cuda.is_available()" in source
    assert "device=DEVICE" in source
    assert "training_executed" in source
    assert "test_images_accessed" in source
    assert "model.train" not in source
    assert "--open-test" not in source
