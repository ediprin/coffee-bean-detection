from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from .model import (
    DistributionShiftConvBlock,
    HongSPPFAttention,
    HongTransferConfig,
    PartialConvBlock,
    inject_hong_transfer,
)


def _tensor_hash(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _module_hashes(module: nn.Module) -> dict[str, str]:
    return {name: _tensor_hash(value) for name, value in module.state_dict().items()}


def _shape(value: Any) -> Any:
    if isinstance(value, Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    return type(value).__name__


def _prediction_paths(model: nn.Module) -> list[str]:
    head_index = len(model.model) - 1
    return [
        f"model.{head_index}.{branch}.{level}.2"
        for branch in ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
        for level in range(3)
    ]


def _module_record(model: nn.Module, path: str) -> dict[str, Any]:
    module = model.get_submodule(path)
    convolution = next(
        (child for child in module.modules() if isinstance(child, nn.Conv2d)),
        None,
    )
    return {
        "path": path,
        "type": type(module).__name__,
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
        "in_channels": getattr(convolution, "in_channels", None),
        "out_channels": getattr(convolution, "out_channels", None),
        "kernel_size": list(convolution.kernel_size) if convolution is not None else None,
        "stride": list(convolution.stride) if convolution is not None else None,
    }


def _load_pretrained(model: nn.Module, weights: str | Path | None) -> dict[str, Any]:
    if weights is None:
        return {
            "requested": False,
            "source": None,
            "eligible_shape_matched_keys": [],
            "shape_mismatched_keys": [],
        }
    from ultralytics import YOLO

    source = YOLO(str(weights)).model
    source_state = source.state_dict()
    target_state = model.state_dict()
    eligible = sorted(
        key
        for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    )
    mismatched = sorted(
        key
        for key, value in source_state.items()
        if key in target_state and value.shape != target_state[key].shape
    )
    model.load(source)
    return {
        "requested": True,
        "source": str(weights),
        "eligible_shape_matched_keys": eligible,
        "shape_mismatched_keys": mismatched,
        "source_only_keys": sorted(set(source_state) - set(target_state)),
        "target_only_keys_before_injection": sorted(set(target_state) - set(source_state)),
    }


def build_hong_model(
    model_yaml: str | Path,
    *,
    nc: int,
    weights: str | Path | None = None,
    config: HongTransferConfig | dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(model_yaml), nc=int(nc), verbose=False)
    pretrained = _load_pretrained(model, weights)
    paths = _prediction_paths(model)
    before_hashes = {path: _module_hashes(model.get_submodule(path)) for path in paths}
    before_state = set(model.state_dict())
    before_parameters = sum(parameter.numel() for parameter in model.parameters())
    architecture = inject_hong_transfer(model, config)
    after_hashes = {path: _module_hashes(model.get_submodule(path)) for path in paths}
    after_state = set(model.state_dict())
    if before_hashes != after_hashes:
        raise RuntimeError("Terminal prediction layers berubah saat Hong injection")
    transfer = {
        **pretrained,
        "new_keys_after_injection": sorted(after_state - before_state),
        "removed_keys_after_injection": sorted(before_state - after_state),
        "preserved_prediction_hashes": before_hashes,
        "parameters_before_injection": int(before_parameters),
        "parameters_after_injection": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    return model, {"architecture": architecture, "pretrained_transfer": transfer}


def static_architecture_audit(
    model_yaml: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    weights: str | Path | None = None,
    config: HongTransferConfig | dict[str, Any] | None = None,
    image_size: int = 128,
) -> dict[str, Any]:
    """Run the no-training static gate required by the frozen protocol."""

    frozen = HongTransferConfig.from_mapping(config)
    model, records = build_hong_model(
        model_yaml, nc=nc, weights=weights, config=frozen
    )
    model.cpu()
    hooks: dict[str, list[int]] = {}
    handles = []
    for index in (*frozen.dsconv_layer_indices, frozen.sppf_layer_index, 16, 19, 22):
        path = f"model.{index}"
        handles.append(
            model.get_submodule(path).register_forward_hook(
                lambda _module, _inputs, result, path=path: hooks.__setitem__(
                    path, list(result.shape)
                )
            )
        )

    train_input = torch.randn(2, 3, image_size, image_size)
    model.train()
    train_output = model(train_input)
    if set(train_output) != {"one2many", "one2one"}:
        raise RuntimeError(f"Kontrak train berubah: {set(train_output)}")
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=2)
    dummy_batch = {
        "img": train_input,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.3], [0.4, 0.4, 0.2, 0.2]]
        ),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    detection_loss, loss_items = model.loss(dummy_batch, train_output)
    detection_loss.sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    if not finite_gradients:
        raise RuntimeError("Backward Hong menghasilkan gradient non-finite/kosong")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    eval_inputs = {
        "batch1": torch.randn(1, 3, image_size, image_size),
        "batch2": torch.randn(2, 3, image_size, image_size),
    }
    eval_shapes = {}
    with torch.inference_mode():
        for name, inputs in eval_inputs.items():
            eval_shapes[name] = _shape(model(inputs))
    for handle in handles:
        handle.remove()

    state = model.state_dict()
    clone, _ = build_hong_model(model_yaml, nc=nc, weights=None, config=frozen)
    missing, unexpected = clone.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Reload state tidak identik: missing={missing}, unexpected={unexpected}"
        )
    clone.eval()
    with torch.inference_mode():
        left = model(eval_inputs["batch1"])[0]
        right = clone(eval_inputs["batch1"])[0]
    reload_equal = bool(torch.allclose(left, right, atol=1.0e-6, rtol=1.0e-5))
    if not reload_equal:
        raise RuntimeError("Output berubah setelah state-dict reload")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "hong_state.pt"
        torch.save(state, checkpoint)
        serialized_bytes = checkpoint.stat().st_size
        resume_checkpoint = Path(directory) / "hong_resume.pt"
        torch.save(
            {
                "epoch": 0,
                "model": copy.deepcopy(model),
                "optimizer": optimizer.state_dict(),
            },
            resume_checkpoint,
        )
        restored = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        restored_model = restored["model"].float().eval()
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0e-4)
        restored_optimizer.load_state_dict(restored["optimizer"])
        with torch.inference_mode():
            restored_output = restored_model(eval_inputs["batch1"])[0]
        checkpoint_resume_equal = bool(
            torch.allclose(left, restored_output, atol=1.0e-6, rtol=1.0e-5)
        )
        if not checkpoint_resume_equal:
            raise RuntimeError("Full checkpoint/optimizer resume mengubah output")

    # Ultralytics fuses native Conv/BN blocks during inference. Verify that the
    # custom DSConv survives that operation instead of silently becoming Conv.
    fused = copy.deepcopy(model)
    fused.fuse(verbose=False)
    dsconv_after_fuse = sum(
        isinstance(module, DistributionShiftConvBlock)
        for module in fused.modules()
    )
    if dsconv_after_fuse != len(frozen.dsconv_layer_indices):
        raise RuntimeError("DSConv hilang pada automatic model.fuse()")

    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            model(eval_inputs["batch1"])
    latency_ms = (time.perf_counter() - started) * 1000.0 / 3.0

    architecture = records["architecture"]
    module_paths = (
        architecture["dsconv_paths"]
        + [architecture["sppf_attention_path"]]
        + [
            path
            for paths in architecture["pconv_paths"].values()
            for path in paths
        ]
    )
    payload = {
        "protocol": "HONG-YOLO26-TRANSFER-v1.2.0",
        "training_executed": False,
        "test_images_accessed": False,
        "model_yaml": str(Path(model_yaml).resolve()),
        "weights": str(weights) if weights is not None else None,
        "nc": int(nc),
        "image_size": int(image_size),
        **records,
        "module_records": [_module_record(model, path) for path in module_paths],
        "feature_shapes": hooks,
        "train_output_schema": _shape(train_output),
        "native_detection_loss_shape": list(detection_loss.shape),
        "native_detection_loss_items": [float(value) for value in loss_items.detach()],
        "eval_output_schema": eval_shapes,
        "finite_gradients": finite_gradients,
        "state_reload_equal": reload_equal,
        "checkpoint_resume_equal": checkpoint_resume_equal,
        "state_dict_bytes": int(serialized_bytes),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "cpu_latency_smoke_ms_batch1": float(latency_ms),
        "module_counts": {
            "dsconv_blocks": sum(
                isinstance(module, DistributionShiftConvBlock)
                for module in model.modules()
            ),
            "sppf_attention": sum(
                isinstance(module, HongSPPFAttention) for module in model.modules()
            ),
            "pconv_blocks": sum(
                isinstance(module, PartialConvBlock) for module in model.modules()
            ),
            "dsconv_blocks_after_fuse": int(dsconv_after_fuse),
        },
        "static_gate": "PASS",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["output"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Static Hong-to-YOLO26 architecture audit")
    parser.add_argument("--model-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--nc", type=int, default=21)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    result = static_architecture_audit(
        args.model_yaml,
        args.output,
        nc=args.nc,
        weights=args.weights,
        image_size=args.image_size,
    )
    print(json.dumps(result["module_counts"], indent=2))
    print("STATIC GATE:", result["static_gate"])
    print("SAVED:", result["output"])


if __name__ == "__main__":
    main()
