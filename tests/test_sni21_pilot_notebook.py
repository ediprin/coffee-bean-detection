import ast
import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "SNI21_VADCP_Pilot_Train_Colab.ipynb"
)


def _load_notebook() -> tuple[dict, str]:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )
    return payload, source


def test_pilot_notebook_code_cells_are_valid_python() -> None:
    payload, _ = _load_notebook()
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell-{index}")


def test_pilot_notebook_locks_protocol_and_test() -> None:
    _, source = _load_notebook()
    assert "profile='pilot'" in source
    assert "RUN_TRAINING = False" in source
    assert "if 'pilot' not in globals()" in source
    assert "sni21-colab-pilot-summary.json" in source
    assert "seeds=(42,)" in source
    assert "evaluation_split='val'" in source
    assert "open_test=False" in source
    assert "--open-test" not in source
    assert "A0_yolo26n_screen.yaml" in source
    assert "A1_yolo26n_screen.yaml" in source
    assert "A2_yolo26n_screen.yaml" in source


def test_pilot_notebook_preflights_persistent_backups() -> None:
    _, source = _load_notebook()
    assert "userdata.get('HF_TOKEN')" in source
    assert "api.whoami()" in source
    assert "api.upload_file(" in source
    assert "private=True" in source
    assert "MyDrive" in source
    assert "hf_repo_id=HF_REPO_ID" in source
