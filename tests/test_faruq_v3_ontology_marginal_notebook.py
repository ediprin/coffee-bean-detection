import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Faruq_V3_Ontology_Marginal_Screening_Colab.ipynb"


def test_screening_notebook_is_resume_safe_and_test_locked() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    assert "run_faruq_v3_ontology_marginal" in source
    assert "D0_seed42/weights/best.pt" in source
    assert "assert not (DATA_ROOT / 'test').exists()" in source
    assert "--seed', '42'" in source
    assert "C0=\"" not in source
    assert "for code in ('C0', 'S0')" in source
    assert "Rerun notebook untuk resume" in source
    assert "seed tambahan" in source
