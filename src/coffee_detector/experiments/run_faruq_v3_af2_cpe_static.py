from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_cpe import AF2CPEDetectionModel
from coffee_detector.fsce_cpe.loss import cpe_supervised_contrastive_loss
from coffee_detector.fsce_cpe.model import load_fsce_cpe_detector_weights

ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_nc(source: torch.nn.Module) -> int:
    """Read class count from the actual Detect head instead of assuming model.nc exists."""
    detector = getattr(source, "model", None)
    if isinstance(detector, (torch.nn.Sequential, torch.nn.ModuleList)) and len(detector):
        head = detector[-1]
        if hasattr(head, "nc"):
            return int(head.nc)
    yaml_payload = getattr(source, "yaml", None)
    if isinstance(yaml_payload, dict) and "nc" in yaml_payload:
        return int(yaml_payload["nc"])
    names = getattr(source, "names", None)
    if isinstance(names, (dict, list, tuple)):
        return len(names)
    raise RuntimeError("Tidak dapat menentukan nc dari checkpoint AF2")


def _torch_device(device: str | torch.device) -> torch.device:
    """Normalize Ultralytics-style device strings for direct PyTorch use."""
    if isinstance(device, torch.device):
        return device
    text = str(device).strip()
    if text.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {text} diminta tetapi torch.cuda.is_available() == False")
        return torch.device(f"cuda:{text}")
    return torch.device(text)


def _flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [item for key in sorted(value) for item in _flatten_tensors(value[key])]
    if isinstance(value, (tuple, list)):
        return [item for child in value for item in _flatten_tensors(child)]
    return []


def _state_dict_exact(left: torch.nn.Module, right: torch.nn.Module) -> tuple[bool, dict]:
    a, b = left.state_dict(), right.state_dict()
    same_keys = tuple(a.keys()) == tuple(b.keys())
    mismatched = []
    if same_keys:
        for key in a:
            if not torch.equal(a[key], b[key]):
                mismatched.append(key)
                if len(mismatched) >= 8:
                    break
    return same_keys and not mismatched, {
        "same_keys": same_keys,
        "mismatched_keys_preview": mismatched,
        "num_items_left": len(a),
        "num_items_right": len(b),
    }


def _compare_outputs(left, right, *, rtol: float = 1e-5, atol: float = 1e-6) -> dict:
    tensors_a, tensors_b = _flatten_tensors(left), _flatten_tensors(right)
    same_count = bool(tensors_a) and len(tensors_a) == len(tensors_b)
    same_shapes = same_count and all(a.shape == b.shape for a, b in zip(tensors_a, tensors_b))
    if not same_shapes:
        return {
            "same_tensor_count": same_count,
            "same_shapes": same_shapes,
            "tensor_count_left": len(tensors_a),
            "tensor_count_right": len(tensors_b),
            "bitwise_equal": False,
            "allclose": False,
            "rtol": rtol,
            "atol": atol,
            "max_abs_diff": None,
            "max_rel_diff": None,
        }

    bitwise_equal = all(torch.equal(a, b) for a, b in zip(tensors_a, tensors_b))
    allclose = all(torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True) for a, b in zip(tensors_a, tensors_b))
    max_abs = 0.0
    max_rel = 0.0
    for a, b in zip(tensors_a, tensors_b):
        af, bf = a.detach().float(), b.detach().float()
        diff = (af - bf).abs()
        if diff.numel():
            max_abs = max(max_abs, float(diff.max().cpu()))
            denom = torch.maximum(af.abs(), bf.abs()).clamp_min(atol)
            max_rel = max(max_rel, float((diff / denom).max().cpu()))
    return {
        "same_tensor_count": True,
        "same_shapes": True,
        "tensor_count_left": len(tensors_a),
        "tensor_count_right": len(tensors_b),
        "bitwise_equal": bitwise_equal,
        "allclose": allclose,
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
    }


def run_static_audit(checkpoint: str | Path, output: str | Path, *, device: str = "cpu") -> dict:
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    torch_device = _torch_device(device)
    print(f"[STATIC] checkpoint={checkpoint}", flush=True)
    print(f"[STATIC] device={device} -> torch_device={torch_device}", flush=True)

    configs = []
    for arm in ("AF2CPE0", "AF2CPE5"):
        path = ROOT / f"configs/af2_cpe/{arm}_yolo26n.yaml"
        print(f"[STATIC] read config {path}", flush=True)
        configs.append(yaml.safe_load(path.read_text(encoding="utf-8")))
    left, right = configs
    left_same = {k: v for k, v in left["cpe"].items() if k != "loss_weight"}
    right_same = {k: v for k, v in right["cpe"].items() if k != "loss_weight"}
    config_matched = left["afab"] == right["afab"] and left["train"] == right["train"] and left_same == right_same

    print("[STATIC] load AF2 checkpoint", flush=True)
    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model
    nc = _source_nc(source)
    print(f"[STATIC] checkpoint class count nc={nc}", flush=True)

    models = []
    for index, payload in enumerate(configs):
        print(f"[STATIC] build wrapped model {index}", flush=True)
        model = AF2CPEDetectionModel(
            str(ROOT / payload["model"]), nc=nc, verbose=False,
            afab=payload["afab"], cpe=payload["cpe"],
        ).to(torch_device)
        print(f"[STATIC] transfer AF2 weights into wrapped model {index}", flush=True)
        transfer = load_fsce_cpe_detector_weights(model, source)
        print(f"[STATIC] transfer {index}: {transfer}", flush=True)
        models.append(model)

    # The only intentional candidate difference is a training-only scalar loss
    # weight. Copy the complete tensor state, then verify it is exactly equal.
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    shared_state_exact, state_diag = _state_dict_exact(models[0], models[1])
    print(f"[STATIC] shared tensor state exact={shared_state_exact}", flush=True)

    calls = [0, 0]
    hooks = []
    for index, model in enumerate(models):
        head = model.model[-1]

        def _count_projection(_module, _inputs, _output, *, i=index):
            calls[i] += 1

        hooks.append(head.cpe_projection.register_forward_hook(_count_projection))
        model.eval()

    print("[STATIC] inference equivalence check", flush=True)
    torch.manual_seed(42)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    sample = torch.rand(1, 3, 64, 64, device=torch_device)
    with torch.no_grad():
        a = models[0](sample)
        b = models[1](sample)
    for hook in hooks:
        hook.remove()

    inference_diag = _compare_outputs(a, b, rtol=1e-5, atol=1e-6)
    inference_equivalent = bool(inference_diag["allclose"])
    print(
        "[STATIC] inference "
        f"bitwise_equal={inference_diag['bitwise_equal']} "
        f"allclose={inference_diag['allclose']} "
        f"max_abs_diff={inference_diag['max_abs_diff']} "
        f"max_rel_diff={inference_diag['max_rel_diff']}",
        flush=True,
    )

    print("[STATIC] projection gradient check", flush=True)
    projection = models[0].model[-1].cpe_projection
    projection.train()
    features = [torch.randn(2, layer.in_channels, 2, 2, device=torch_device) for layer in projection.projections]
    embeddings = projection(features).reshape(-1, projection.config.embedding_dim)
    labels = torch.tensor([0, 0, 1, 1] * 6, device=torch_device, dtype=torch.long)[: embeddings.shape[0]]
    if labels.shape[0] != embeddings.shape[0]:
        raise RuntimeError(
            f"Static synthetic labels tidak sejajar: labels={labels.shape[0]} embeddings={embeddings.shape[0]}"
        )
    base = cpe_supervised_contrastive_loss(embeddings, labels, temperature=projection.config.temperature)
    if not bool(torch.isfinite(base)):
        raise RuntimeError(f"CPE static loss tidak finite: {float(base.detach().cpu())}")

    grad = {}
    for name, weight in (("control", 0.0), ("candidate", 0.5)):
        projection.zero_grad(set_to_none=True)
        (base * weight).backward(retain_graph=True)
        values = [p.grad for p in projection.parameters()]
        grad[name] = {
            "finite": all(g is not None and bool(torch.isfinite(g).all()) for g in values),
            "sum_abs": sum(float(g.abs().sum().detach().cpu()) for g in values if g is not None),
        }

    checks = {
        "configs_differ_only_loss_weight": config_matched and left["cpe"]["loss_weight"] == 0.0 and right["cpe"]["loss_weight"] == 0.5,
        "shared_tensor_state_exact": shared_state_exact,
        "shared_state_inference_equivalent": inference_equivalent,
        "projection_not_called_in_inference": calls == [0, 0],
        "control_projection_gradient_zero": grad["control"]["finite"] and grad["control"]["sum_abs"] == 0.0,
        "candidate_projection_gradient_finite_nonzero": grad["candidate"]["finite"] and grad["candidate"]["sum_abs"] > 0.0,
    }
    result = {
        "format": "coffee_detector.af2_cpe.static_audit.v2",
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_nc": nc,
        "requested_device": str(device),
        "torch_device": str(torch_device),
        "checks": checks,
        "state_diagnostic": state_diag,
        "inference_diagnostic": inference_diag,
        "gradient": grad,
        "projection_inference_calls": calls,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "training_authorized": all(checks.values()),
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[STATIC] decision={result['decision']} output={output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_static_audit(args.af2_checkpoint, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
