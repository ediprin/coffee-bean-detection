from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .model import (
    FAMILIES,
    Family35x3GeometryAdapter,
    GeometryFactorizationConfig,
    GeometryFactorizedDetectionModel,
    Shared60GeometryAdapter,
    family_class_indices,
    load_geometry_factorized_weights,
)


def _final_layers_zero(adapter) -> bool:
    if isinstance(adapter, Shared60GeometryAdapter):
        layers = [adapter.network[-1]]
    elif isinstance(adapter, Family35x3GeometryAdapter):
        layers = [adapter.networks[family][-1] for family in FAMILIES]
    else:
        raise TypeError(type(adapter).__name__)
    return all(
        bool(torch.count_nonzero(layer.weight) == 0 and torch.count_nonzero(layer.bias) == 0)
        for layer in layers
    )


def static_geometry_factorization_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    class_names: Mapping[int, str] | Sequence[str],
    output: str | Path,
    *,
    nc: int,
    image_size: int = 128,
) -> dict:
    from ultralytics import YOLO

    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    source = YOLO(str(d0_checkpoint)).model.eval()
    config = GeometryFactorizationConfig()
    shared = GeometryFactorizedDetectionModel(
        str(Path(model_yaml).resolve()),
        nc=nc,
        verbose=False,
        geometry_factorization=config,
        mode="shared60",
        class_names=class_names,
    ).eval()
    family = GeometryFactorizedDetectionModel(
        str(Path(model_yaml).resolve()),
        nc=nc,
        verbose=False,
        geometry_factorization=config,
        mode="family35x3",
        class_names=class_names,
    ).eval()
    load_geometry_factorized_weights(shared, source)
    load_geometry_factorized_weights(family, source)

    native_params = sum(parameter.numel() for parameter in source.parameters())
    shared_params = sum(parameter.numel() for parameter in shared.parameters())
    family_params = sum(parameter.numel() for parameter in family.parameters())
    expected_indices = family_class_indices(class_names, nc)

    image = torch.rand(1, 3, image_size, image_size)
    with torch.inference_mode():
        native = source(image)
        shared_zero = shared(image)
        family_zero = family(image)
    native_pred = native[1]["one2one"]
    shared_pred = shared_zero[1]["one2one"]
    family_pred = family_zero[1]["one2one"]
    shared_head = shared.model[-1]
    family_head = family.model[-1]

    shared_targets = tuple(int(v) for v in shared_head.adapter.target_indices.cpu().tolist())
    family_targets = tuple(
        int(v)
        for fam in FAMILIES
        for v in getattr(family_head.adapter, f"indices_{fam}").cpu().tolist()
    )
    expected_flat = tuple(index for fam in FAMILIES for index in expected_indices[fam])

    gates = {
        "shared_added_parameters_exactly_849": shared_params - native_params == 849,
        "family_added_parameters_exactly_849": family_params - native_params == 849,
        "exact_parameter_match": shared_params == family_params,
        "shared_initial_boxes_equal_native": torch.equal(native_pred["boxes"], shared_pred["boxes"]),
        "family_initial_boxes_equal_native": torch.equal(native_pred["boxes"], family_pred["boxes"]),
        "shared_initial_scores_equal_native": torch.equal(native_pred["scores"], shared_pred["scores"]),
        "family_initial_scores_equal_native": torch.equal(native_pred["scores"], family_pred["scores"]),
        "shared_final_projection_zero": _final_layers_zero(shared_head.adapter),
        "family_final_projections_zero": _final_layers_zero(family_head.adapter),
        "shared_targets_exactly_nine": len(shared_targets) == 9 and len(set(shared_targets)) == 9,
        "family_targets_exactly_nine": len(family_targets) == 9 and len(set(family_targets)) == 9,
        "same_target_indices": shared_targets == family_targets == expected_flat,
    }
    names = (
        [str(class_names[index]) for index in range(nc)]
        if isinstance(class_names, Mapping)
        else [str(value) for value in class_names]
    )
    payload = {
        "protocol": "faruq-v3-geometry-family-factorization-static-v1",
        "training_executed": False,
        "test_images_accessed": False,
        "test_opened": False,
        "models": {
            "D0": {"parameters": native_params},
            "GEO-SHARED60": {"parameters": shared_params, "added_parameters": shared_params - native_params},
            "GEO-FAM35x3": {"parameters": family_params, "added_parameters": family_params - native_params},
        },
        "family_indices": {family_name: list(indices) for family_name, indices in expected_indices.items()},
        "target_classes": [names[index] for index in expected_flat],
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload
