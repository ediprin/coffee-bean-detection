import json
from pathlib import Path

from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    FORMAT as DVF1_FORMAT,
    _sha256,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation import (
    ALLOWED_SEEDS,
    FORMAT as ARM_FORMAT,
    PROTOCOL,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation_decision import (
    run_decision,
)


METRICS = (
    "physical_pair_accuracy",
    "macro_class_accuracy",
    "bottom3_class_accuracy",
    "worst_class_accuracy",
)


def _endpoint(physical: float, macro: float, bottom3: float, worst: float) -> dict:
    values = dict(zip(METRICS, (physical, macro, bottom3, worst)))
    values.update({"accuracy_by_class": {}, "classes_without_ground_truth": []})
    return values


def test_confirmation_decision_passes_consistent_three_seed_gain(tmp_path: Path):
    paper = _endpoint(0.94, 0.92, 0.80, 0.76)
    seed42 = tmp_path / "seed42.json"
    seed42.write_text(
        json.dumps(
            {
                "format": DVF1_FORMAT,
                "seed": 42,
                "decision": "PASS",
                "values": {
                    "PAPER_MAX_CONFIDENCE": paper,
                    "DVF1": _endpoint(0.950, 0.930, 0.825, 0.76),
                },
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    arms = {}
    for seed, candidate in (
        (123, _endpoint(0.949, 0.929, 0.820, 0.75)),
        (2026, _endpoint(0.951, 0.931, 0.830, 0.77)),
    ):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "format": ARM_FORMAT,
                    "protocol": PROTOCOL,
                    "seed": seed,
                    "status": "COMPLETE",
                    "values": {
                        "PAPER_MAX_CONFIDENCE": paper,
                        "DVF1_CANDIDATE": candidate,
                    },
                    "run_contract": {
                        "dvf1_seed42_result_sha256": _sha256(seed42)
                    },
                    "test_images_accessed": False,
                }
            ),
            encoding="utf-8",
        )
        arms[seed] = path

    result = run_decision(
        seed42,
        arms[123],
        arms[2026],
        tmp_path / "decision.json",
    )

    assert result["decision"] == "PASS"
    assert result["next"] == "PROMOTE_DVF1_DUAL_VIEW_FUSION"
    assert all(result["criteria"].values())
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False


def test_confirmation_protocol_and_notebooks_are_frozen_and_test_locked():
    protocol = Path(
        "docs/DEFECTOSCAFEVERDE_DVF1_CONFIRMATION_PROTOCOL_2026-10-06.md"
    ).read_text(encoding="utf-8")

    assert "before training confirmation seeds 123 or 2026" in protocol
    assert "No validation early stopping" in protocol
    assert "Test is not extracted or accessed" in protocol
    assert ALLOWED_SEEDS == (123, 2026)

    for name in (
        "DefectosCafeVerde_DVF1_Confirmation_Seed123_Colab.ipynb",
        "DefectosCafeVerde_DVF1_Confirmation_Seed2026_Colab.ipynb",
        "DefectosCafeVerde_DVF1_Confirmation_Decision_Colab.ipynb",
    ):
        payload = json.loads((Path("notebooks") / name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell["cell_type"] == "code"
        )
        compile(source, name, "exec")
        assert "codex/defectoscafeverde-dual-view-fusion" in source
        assert "test_images_accessed" in source
    for seed in ALLOWED_SEEDS:
        source = Path(
            f"notebooks/DefectosCafeVerde_DVF1_Confirmation_Seed{seed}_Colab.ipynb"
        ).read_text(encoding="utf-8")
        assert "--authorize-training" in source
