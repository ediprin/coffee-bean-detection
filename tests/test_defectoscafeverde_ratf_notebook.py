import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/DefectosCafeVerde_RATF1_Seed42_Colab.ipynb"


def test_ratf_notebook_contract_and_test_lock():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "BRANCH='codex/defectoscafeverde-ratf1'" in source
    assert "run_defectoscafeverde_ratf" in source
    assert "--authorize-training" in source
    assert "RATF1: {epochs}/50" not in source
    assert "last.pt tersimpan di Drive setiap epoch" in source
    assert "if (DATA/'test').exists(): raise RuntimeError" in source
    assert "payload.pop('test',None)" in source
    assert "split='test'" not in source
    assert "build_decision(D0,RESULT" in source
    assert "D0 tidak terlihat lewat mount" in source
    assert "D0_HEADLINE" in source
    assert "PROVISIONAL_MISSING_RAW_D0_REFERENCE" in source
    assert "Drive API" in source
