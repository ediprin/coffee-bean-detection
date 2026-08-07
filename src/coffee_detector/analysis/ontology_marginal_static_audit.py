from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics.nn.tasks import DetectionModel

from coffee_detector.dataset import write_json
from coffee_detector.ontology_marginal import OntologyDetectionModel, OntologyMarginalConfig
from coffee_detector.train import load_experiment


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG = REPO_ROOT / "configs/coffee_fg/D0_yolo26n_p3.yaml"
CONTROL_CONFIG = REPO_ROOT / "configs/structured_ontology/C0_yolo26n_identity_control.yaml"
SEMANTIC_CONFIG = REPO_ROOT / "configs/structured_ontology/S0_yolo26n_semantic_marginal.yaml"


def _model_path(config: dict) -> Path:
    path = Path(config["model"])
    return path if path.is_absolute() else REPO_ROOT / path


def _parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def audit_ontology_marginal_static(output: str | Path) -> dict:
    baseline_config = load_experiment(BASELINE_CONFIG)
    control_config = load_experiment(CONTROL_CONFIG)
    semantic_config = load_experiment(SEMANTIC_CONFIG)
    baseline = DetectionModel(_model_path(baseline_config), nc=21, verbose=False)
    control = OntologyDetectionModel(
        _model_path(control_config),
        nc=21,
        verbose=False,
        ontology_marginal=control_config["ontology_marginal"],
    )
    semantic = OntologyDetectionModel(
        _model_path(semantic_config),
        nc=21,
        verbose=False,
        ontology_marginal=semantic_config["ontology_marginal"],
    )
    baseline_state = baseline.state_dict()
    control.load_state_dict(baseline_state, strict=True)
    semantic.load_state_dict(baseline_state, strict=True)
    baseline.eval()
    control.eval()
    semantic.eval()
    torch.manual_seed(42)
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        baseline_detection = baseline(image)[0]
        control_detection = control(image)[0]
        semantic_detection = semantic(image)[0]

    frozen_control = OntologyMarginalConfig.from_mapping(
        control_config["ontology_marginal"]
    )
    frozen_semantic = OntologyMarginalConfig.from_mapping(
        semantic_config["ontology_marginal"]
    )
    gates = {
        "same_model_yaml": baseline_config["model"]
        == control_config["model"]
        == semantic_config["model"],
        "same_pretrained_weights": baseline_config["weights"]
        == control_config["weights"]
        == semantic_config["weights"],
        "same_training_schedule": baseline_config["train"]
        == control_config["train"]
        == semantic_config["train"],
        "same_parameter_count": _parameters(baseline)
        == _parameters(control)
        == _parameters(semantic),
        "same_state_dict_schema": set(baseline_state)
        == set(control.state_dict())
        == set(semantic.state_dict()),
        "identical_control_inference": torch.equal(
            baseline_detection, control_detection
        ),
        "identical_semantic_inference": torch.equal(
            baseline_detection, semantic_detection
        ),
        "same_task_masks": frozen_control.tasks == frozen_semantic.tasks,
        "same_task_weights": frozen_control.task_weights
        == frozen_semantic.task_weights,
        "same_auxiliary_gain": frozen_control.auxiliary_gain
        == frozen_semantic.auxiliary_gain,
        "correct_control_modes": frozen_control.mode == "identity_control"
        and frozen_semantic.mode == "semantic",
    }
    payload = {
        "protocol": "sni21-ontology-marginal-static-audit-v1",
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "models": {
            "D0": {"parameters": _parameters(baseline)},
            "C0": {"parameters": _parameters(control)},
            "S0": {"parameters": _parameters(semantic)},
        },
        "maximum_inference_difference": {
            "C0_vs_D0": float((control_detection - baseline_detection).abs().max()),
            "S0_vs_D0": float((semantic_detection - baseline_detection).abs().max()),
        },
        "training_executed": False,
        "inference_mode": "synthetic_tensor_static_equivalence_only",
        "dataset_accessed": False,
        "test_images_accessed": False,
        "training_authorized": False,
        "next_action": (
            "freeze_seed42_validation_screening_runner"
            if all(gates.values())
            else "repair_implementation_before_any_training"
        ),
    }
    write_json(payload, Path(output).expanduser().resolve())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit ontology-marginal YOLO")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = audit_ontology_marginal_static(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
