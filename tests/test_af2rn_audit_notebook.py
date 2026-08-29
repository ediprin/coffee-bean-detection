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


def test_af2rn_kaggle_notebook_is_fail_fast_and_train_only() -> None:
    path = Path("notebooks/Faruq_V3_AF2RN_Static_Observability_Kaggle.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    code = "\n".join(code_cells)
    for index, source in enumerate(code_cells):
        compile(source, f"{path}:{index}", "exec")

    assert "af2_spectral_kaggle_manifest.json" in code_cells[0]
    assert "D0_seed42_best.pt" in code_cells[0]
    assert "git','clone" not in code_cells[0]
    assert "pip','install" not in code_cells[0]
    assert "prepare_af2rn_kaggle_input" in code
    assert "run_af2rn_static_audit" in code
    assert "run_af2rn_observability_audit" in code
    assert "validation_files_read') is not False" in code
    assert "model.train" not in code
    assert "authorize-training" not in code


def test_af2rn_kaggle_training_notebook_gates_before_training() -> None:
    path = Path("notebooks/Faruq_V3_AF2RN_Seed42_Training_Kaggle.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    code = "\n".join(code_cells)
    for index, source in enumerate(code_cells):
        compile(source, f"{path}:{index}", "exec")

    assert "af2_spectral_kaggle_manifest.json" in code_cells[0]
    assert "D0_seed42_best.pt" in code_cells[0]
    assert "lfdet_afab_seed42_screening.json" in code_cells[0]
    assert "git','clone" not in code_cells[0]
    assert code.index("run_af2rn_static_audit") < code.index("--authorize-training")
    assert code.index("run_af2rn_observability_audit") < code.index(
        "--authorize-training"
    )
    assert "run_faruq_v3_af2rn" in code
    assert "AF2RN: {epochs}/50 epoch" in code
    assert "weights/last.pt" in code
    assert "run_af2rn_seed42_decision" in code
    assert "--authorize-test" not in code
