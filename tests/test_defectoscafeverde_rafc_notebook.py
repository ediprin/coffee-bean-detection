from __future__ import annotations

import json
from pathlib import Path


def test_rafc_colab_contract():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/DefectosCafeVerde_RAFC1_Seed42_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), str(path), "exec")
    assert "codex/defectoscafeverde-rafc" in code
    assert "run_defectoscafeverde_rafc" in code
    assert "defectoscafeverde-rafc-v1" in code
    assert "--authorize-training" in code
    assert "time.sleep(120)" in code
    assert "TEST TEREXPOSE" in code
    assert "test" not in code.split("allowed_files=")[1].split("\n", 1)[0]
