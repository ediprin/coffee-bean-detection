import ast
import json
from pathlib import Path

import torch

from coffee_detector.analysis.coffee_standard_j25_yolov8n_cwcf_target_audit import (
    CONFUSION_FAMILY_NAMES,
    PROTOCOL,
    TARGET_CLASS,
    _attribute_compatibility_2d,
    _component_id,
    _rank,
)
from coffee_detector.j25_cwcf import build_j25_attribute_matrix
from coffee_detector.j25_cwcf.model import ATTRIBUTE_NAMES


ROOT = Path(__file__).resolve().parents[1]


def test_target_audit_contract_is_fixed_to_black_broken_family():
    assert PROTOCOL == "coffee-standard-j25-yolov8n-cwcf-target-audit-v1"
    assert TARGET_CLASS == "Biji Hitam Pecah"
    assert CONFUSION_FAMILY_NAMES == (
        "Biji Hitam Pecah",
        "Biji Hitam Penuh",
        "Biji Hitam Sebagian",
        "Biji Pecah",
    )


def test_component_id_collapses_train_sibling_suffix():
    identity = "a" * 64
    assert _component_id(Path(identity + "_00.jpg")) == identity
    assert _component_id(Path(identity + "_02.png")) == identity
    assert _component_id(Path(identity + ".jpg")) == identity


def test_exact_target_attribute_code_ranks_target_first_or_tied_first():
    matrix = build_j25_attribute_matrix()
    target = 7
    code = matrix[target]
    logits = torch.where(code > 0.5, torch.tensor(8.0), torch.tensor(-8.0))
    compatibility = _attribute_compatibility_2d(logits[None], matrix)[0]
    target_rank = _rank(compatibility, target)
    assert target_rank == 1
    assert ATTRIBUTE_NAMES.index("black") == 5
    assert ATTRIBUTE_NAMES.index("broken") == 6


def test_protocol_and_notebook_are_diagnostic_only():
    protocol = (
        ROOT
        / "docs/COFFEE_STANDARD_J25_YOLOV8N_CWCF_TARGET_AUDIT_PROTOCOL_2026-09-27.md"
    ).read_text(encoding="utf-8")
    assert "no training" in protocol.lower()
    assert "locked test" in protocol.lower()
    assert "P(black)" in protocol
    assert "P(broken)" in protocol

    notebook = json.loads(
        (
            ROOT
            / "notebooks/Coffee_Standard_J25_YOLOv8n_CWCF_Target_Audit_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8n-cwcf-target-audit'" in code
    assert "authorize_diagnostic=True" in code
    assert ".train(" not in code
    assert "--authorize-training" not in code
    assert "test/images" not in code
