import json
from pathlib import Path

import torch

from coffee_detector.analysis.defectoscafeverde_dual_side_routability import (
    FORMAT as AUDIT_FORMAT,
    _fit_ridge_difference,
    view_features,
)
from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    _sha256,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation import (
    FORMAT as ARM_FORMAT,
    METRICS,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation_decision import (
    FORMAT as DECISION_FORMAT,
)
from coffee_detector.experiments.run_defectoscafeverde_dvsr1_screen import (
    _endpoint_arrays,
    run_dvsr1_screen,
)


def test_frozen_selector_validation_screen_can_pass(tmp_path: Path):
    pairs, classes = 120, 12
    labels = torch.arange(pairs) % classes
    logits = torch.full((pairs, 2, classes), -5.0)
    ious = torch.full((pairs, 2), 0.9)
    correct = torch.zeros((pairs, 2), dtype=torch.bool)
    for index in range(pairs):
        target = int(labels[index])
        correct_side = index % 2
        wrong_side = 1 - correct_side
        correct[index, correct_side] = True
        logits[index, correct_side, target] = 3.0
        logits[index, correct_side, (target + 1) % classes] = -1.0
        logits[index, wrong_side, (target + 1) % classes] = 4.0
        logits[index, wrong_side, (target + 2) % classes] = 3.9
    cache = {
        "split": "val",
        "view_logits": logits,
        "labels": labels,
        "ious": ious,
        "group_ids": [f"val-{index}" for index in range(pairs)],
        "names": {index: f"class-{index}" for index in range(classes)},
    }
    cache_path = tmp_path / "val_pair_cache.pt"
    torch.save(cache, cache_path)
    features = view_features(logits.reshape(-1, classes)).reshape(pairs, 2, -1)
    difference = (features[:, 0] - features[:, 1]).numpy()
    targets = torch.where(correct[:, 0], 1.0, -1.0).numpy()
    selector = _fit_ridge_difference(difference, targets)
    model = {"scale": selector["scale"], "weights": selector["weights"]}
    reference, _ = _endpoint_arrays(cache, model)
    arm_path = tmp_path / "DVF1_seed123_result.json"
    arm_path.write_text(
        json.dumps(
            {
                "format": ARM_FORMAT,
                "seed": 123,
                "status": "COMPLETE",
                "values": {
                    "PAPER_MAX_CONFIDENCE": reference["PAPER_MAX_CONFIDENCE"],
                    "PAIR_ORACLE": reference["PAIR_ORACLE"],
                },
                "run_contract": {"validation_cache_sha256": _sha256(cache_path)},
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    decision_path = tmp_path / "DVF1_three_seed_confirmation.json"
    decision_path.write_text(
        json.dumps(
            {
                "format": DECISION_FORMAT,
                "decision": "PASS",
                "next": "PROMOTE_DVF1_DUAL_VIEW_FUSION",
                "aggregate": {
                    metric: {"dvf1_mean": 0.90} for metric in METRICS
                },
                "artifacts": {"123": {"sha256": _sha256(arm_path)}},
                "training_executed": False,
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "dvsr_routability.json"
    audit_path.write_text(
        json.dumps(
            {
                "format": AUDIT_FORMAT,
                "decision": "AUTHORIZE_DVSR1_VALIDATION_SCREEN",
                "final_train_selector": {
                    "scale": selector["scale"].tolist(),
                    "weights": selector["weights"].tolist(),
                },
                "contract": {
                    "confirmation_arm_sha256": _sha256(arm_path),
                    "confirmation_decision_sha256": _sha256(decision_path),
                },
                "validation_images_accessed": False,
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )

    result = run_dvsr1_screen(
        audit_path,
        cache_path,
        arm_path,
        decision_path,
        tmp_path / "result.json",
    )

    assert result["decision"] == "PASS"
    assert result["next"] == "PROMOTE_DVSR1_SIDE_RELIABILITY"
    assert result["values"]["DVSR1"]["physical_pair_accuracy"] == 1.0
    assert result["selector_fitting_executed"] is False
    assert result["validation_images_accessed"] is True
    assert result["test_images_accessed"] is False


def test_dvsr1_protocol_and_notebook_are_frozen():
    protocol = Path(
        "docs/DEFECTOSCAFEVERDE_DVSR1_PROTOCOL_2026-10-06.md"
    ).read_text(encoding="utf-8")
    notebook = json.loads(
        Path(
            "notebooks/DefectosCafeVerde_DVSR1_Validation_Screen_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "before DVSR1 accesses validation" in protocol
    assert "No class logits are synthesized" in protocol
    assert "test extraction or access: forbidden" in protocol
    compile(source, "dvsr1_colab", "exec")
    assert "run_defectoscafeverde_dvsr1_screen" in source
    assert "dvsr_routability.json" in source
    assert "--train-audit" in source
    assert "test_images_accessed" in source
