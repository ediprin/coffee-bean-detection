from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .model import (
    GeometryConditionedDetectionModel,
    GeometryConditioningConfig,
    _is_size_class,
    load_geometry_conditioned_weights,
)


def static_geometry_conditioning_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    class_names: Mapping[int, str] | Sequence[str],
    output: str | Path,
    *,
    nc: int,
    image_size: int = 128,
) -> dict:
    from ultralytics import YOLO

    source = YOLO(str(Path(d0_checkpoint).expanduser().resolve())).model.eval()
    config = GeometryConditioningConfig()
    control = GeometryConditionedDetectionModel(
        str(Path(model_yaml).resolve()),
        nc=nc,
        verbose=False,
        geometry_conditioning=config,
        signal_mode="zero",
        class_names=class_names,
    ).eval()
    geometry = GeometryConditionedDetectionModel(
        str(Path(model_yaml).resolve()),
        nc=nc,
        verbose=False,
        geometry_conditioning=config,
        signal_mode="geometry",
        class_names=class_names,
    ).eval()
    load_geometry_conditioned_weights(control, source)
    load_geometry_conditioned_weights(geometry, source)

    control_params = sum(parameter.numel() for parameter in control.parameters())
    geometry_params = sum(parameter.numel() for parameter in geometry.parameters())
    native_params = sum(parameter.numel() for parameter in source.parameters())

    image = torch.rand(1, 3, image_size, image_size)
    with torch.inference_mode():
        native = source(image)
        control_zero = control(image)
        geometry_zero = geometry(image)

    native_pred = native[1]["one2one"]
    control_pred = control_zero[1]["one2one"]
    geometry_pred = geometry_zero[1]["one2one"]
    control_head = control.model[-1]
    geometry_head = geometry.model[-1]
    final_control = control_head.adapter.network[-1]
    final_geometry = geometry_head.adapter.network[-1]

    names = (
        [str(class_names[index]) for index in range(nc)]
        if isinstance(class_names, Mapping)
        else [str(value) for value in class_names]
    )
    expected_mask = torch.tensor(
        [1.0 if _is_size_class(name) else 0.0 for name in names],
        dtype=geometry_head.adapter.class_mask.dtype,
    ).view(1, nc, 1)

    gates = {
        "same_parameter_count": control_params == geometry_params,
        "control_initial_boxes_equal_native": torch.equal(
            native_pred["boxes"], control_pred["boxes"]
        ),
        "geometry_initial_boxes_equal_native": torch.equal(
            native_pred["boxes"], geometry_pred["boxes"]
        ),
        "control_initial_scores_equal_native": torch.equal(
            native_pred["scores"], control_pred["scores"]
        ),
        "geometry_initial_scores_equal_native": torch.equal(
            native_pred["scores"], geometry_pred["scores"]
        ),
        "control_final_projection_zero": bool(
            torch.count_nonzero(final_control.weight) == 0
            and torch.count_nonzero(final_control.bias) == 0
        ),
        "geometry_final_projection_zero": bool(
            torch.count_nonzero(final_geometry.weight) == 0
            and torch.count_nonzero(final_geometry.bias) == 0
        ),
        "same_class_mask": torch.equal(
            control_head.adapter.class_mask, geometry_head.adapter.class_mask
        ),
        "mask_matches_size_defined_classes": torch.equal(
            geometry_head.adapter.class_mask.cpu(), expected_mask.cpu()
        ),
        "at_least_two_size_classes_targeted": int(expected_mask.sum().item()) >= 2,
    }
    payload = {
        "protocol": "faruq-v3-geometry-conditioning-static-v1",
        "training_executed": False,
        "test_images_accessed": False,
        "test_opened": False,
        "models": {
            "D0": {"parameters": native_params},
            "GEO-C0": {"parameters": control_params},
            "GEO1": {"parameters": geometry_params},
        },
        "added_parameters": geometry_params - native_params,
        "target_size_classes": [
            name for name in names if _is_size_class(name)
        ],
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload
