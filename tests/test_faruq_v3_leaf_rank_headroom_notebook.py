import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Leaf_Rank_Headroom_Colab.ipynb"


def test_leaf_rank_notebook_is_validation_only_without_training() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_faruq_v3_leaf_rank_headroom" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "model.train(" not in source
    assert "candidate_count=500" in source
    assert "sys.modules.pop(module_name, None)" in source

