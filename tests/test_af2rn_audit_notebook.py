from __future__ import annotations

import json
from pathlib import Path


def test_af2rn_audit_notebook_compiles_and_blocks_training_test() -> None:
    path = Path("notebooks/Faruq_V3_AF2RN_Static_Observability_Colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:{index}", "exec")
    assert "run_af2rn_static_audit" in code
    assert "run_af2rn_observability_audit" in code
    assert "assert not (DATA/'test').exists()" in code
    assert "model.train" not in code
    assert "authorize-training" not in code
