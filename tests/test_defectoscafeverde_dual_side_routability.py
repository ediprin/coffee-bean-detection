import json
from pathlib import Path

import torch

from coffee_detector.analysis.defectoscafeverde_dual_side_routability import (
    _fit_ridge_difference,
    _selector_view,
    run_routability,
    view_features,
)
from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import _sha256
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation import (
    FORMAT as ARM_FORMAT,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation_decision import (
    FORMAT as DECISION_FORMAT,
)


def test_shared_difference_selector_is_swap_equivariant():
    torch.manual_seed(13)
    logits = torch.randn(20, 2, 12)
    features = view_features(logits.reshape(-1, 12)).reshape(20, 2, -1)
    difference = (features[:, 0] - features[:, 1]).numpy()
    targets = torch.where(torch.arange(20) % 2 == 0, 1.0, -1.0).numpy()
    model = _fit_ridge_difference(difference, targets)
    paper = torch.zeros(20, dtype=torch.long).numpy()

    original = _selector_view(difference, paper, model)
    swapped = _selector_view(-difference, 1 - paper, model)

    assert (swapped == 1 - original).all()


def test_train_only_oof_audit_authorizes_learnable_side_reliability(tmp_path: Path):
    pairs, classes = 120, 12
    labels = torch.arange(pairs) % classes
    logits = torch.full((pairs, 2, classes), -5.0)
    ious = torch.full((pairs, 2), 0.9)
    for index in range(pairs):
        target = int(labels[index])
        correct_side = index % 2
        wrong_side = 1 - correct_side
        logits[index, correct_side, target] = 3.0
        logits[index, correct_side, (target + 1) % classes] = -1.0
        logits[index, wrong_side, (target + 1) % classes] = 4.0
        logits[index, wrong_side, (target + 2) % classes] = 3.9
    cache_path = tmp_path / "train_pair_cache.pt"
    torch.save(
        {
            "split": "train",
            "view_logits": logits,
            "labels": labels,
            "ious": ious,
            "group_ids": [f"group-{index}" for index in range(pairs)],
            "names": {index: f"class-{index}" for index in range(classes)},
        },
        cache_path,
    )
    arm_path = tmp_path / "seed123.json"
    arm_path.write_text(
        json.dumps(
            {
                "format": ARM_FORMAT,
                "seed": 123,
                "status": "COMPLETE",
                "run_contract": {"train_cache_sha256": _sha256(cache_path)},
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "format": DECISION_FORMAT,
                "decision": "PASS",
                "next": "PROMOTE_DVF1_DUAL_VIEW_FUSION",
                "artifacts": {"123": {"sha256": _sha256(arm_path)}},
                "training_executed": False,
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    result = run_routability(
        cache_path,
        arm_path,
        decision_path,
        tmp_path / "audit.json",
    )

    assert result["decision"] == "AUTHORIZE_DVSR1_VALIDATION_SCREEN"
    assert all(result["criteria"].values())
    assert result["metrics"]["oof_selector_accuracy"] == 1.0
    assert result["validation_images_accessed"] is False
    assert result["test_images_accessed"] is False


def test_protocol_and_colab_lock_validation_and_test():
    protocol = Path(
        "docs/DEFECTOSCAFEVERDE_DVSR_ROUTABILITY_PROTOCOL_2026-10-06.md"
    ).read_text(encoding="utf-8")
    notebook = json.loads(
        Path(
            "notebooks/DefectosCafeVerde_DVSR_Routability_Audit_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "before this\ntrain-only audit is executed" in protocol
    assert "Validation and test caches are forbidden" in protocol
    assert "validation access: forbidden" in protocol
    compile(source, "dvsr_audit_colab", "exec")
    assert "dual_side_routability" in source
    assert "train_pair_cache.pt" in source
    assert "val_pair_cache.pt" not in source
    assert "test_images_accessed" in source
