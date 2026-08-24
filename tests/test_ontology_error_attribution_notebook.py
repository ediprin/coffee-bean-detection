import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Ontology_Error_Attribution_Colab.ipynb"


def test_attribution_notebook_reuses_checkpoints_without_training_or_test() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_ontology_error_attribution" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "C0_seed42/weights/best.pt" in source
    assert "S0_seed42/weights/best.pt" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "model.train" not in source
    assert "test/images" not in source
