from __future__ import annotations

"""Environment-locked launcher for the AF2 x D-FINE-N seed-42 screen.

v3 adds a hardware/runtime gate before any dataset preparation or training.
The scientific experiment is unchanged from v2.
"""

import importlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import torch


EXPECTED_TORCH_PREFIX = "2.5.1+cu124"
EXPECTED_TORCHVISION = "0.20.1"


def runtime_gate() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Enable a Kaggle GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
    required_arch = f"sm_{capability[0]}{capability[1]}"
    arch_list = list(torch.cuda.get_arch_list())
    torchvision_version = metadata.version("torchvision")

    gates = {
        "torch_exact_pinned_build": str(torch.__version__).startswith(EXPECTED_TORCH_PREFIX),
        "torchvision_exact_pinned_version": torchvision_version.split("+")[0] == EXPECTED_TORCHVISION,
        "gpu_arch_compiled_into_torch": required_arch in arch_list,
    }
    if not all(gates.values()):
        raise RuntimeError(
            "PyTorch/GPU runtime contract failed before training: "
            + json.dumps(
                {
                    "torch": torch.__version__,
                    "torchvision": torchvision_version,
                    "cuda_runtime": torch.version.cuda,
                    "gpu": gpu_name,
                    "compute_capability": capability,
                    "required_arch": required_arch,
                    "arch_list": arch_list,
                    "gates": gates,
                },
                indent=2,
            )
        )

    x = torch.arange(4096, dtype=torch.float32, device="cuda").reshape(64, 64)
    y = (x @ x.T).mean()
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(y).item())
    if not finite:
        raise RuntimeError("CUDA smoke test produced a non-finite result")

    payload = {
        "format": "coffee_detector.dfine_kaggle_runtime.v1",
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_name,
        "compute_capability": list(capability),
        "required_arch": required_arch,
        "arch_list": arch_list,
        "cuda_smoke_pass": True,
        "cuda_smoke_value": float(y.detach().cpu().item()),
        "gates": {**gates, "cuda_smoke_kernel_executes": True},
        "decision": "PASS",
    }
    print("RUNTIME PREFLIGHT PASS")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    runtime = runtime_gate()

    scripts = Path(__file__).resolve().parent
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    if "run_dfine_af2_kaggle_screen_v2" in sys.modules:
        del sys.modules["run_dfine_af2_kaggle_screen_v2"]
    v2 = importlib.import_module("run_dfine_af2_kaggle_screen_v2")

    original_strict_preflight = v2._strict_preflight

    def strict_preflight_with_runtime(**kwargs):
        result = original_strict_preflight(**kwargs)
        result["runtime"] = runtime
        result["gates"]["runtime_preflight_pass"] = runtime["decision"] == "PASS"
        result["decision"] = "PASS" if all(result["gates"].values()) else "FAIL"
        result["training_authorized"] = result["decision"] == "PASS"
        Path(kwargs["output"]).expanduser().resolve().write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    v2._strict_preflight = strict_preflight_with_runtime
    v2.main()


if __name__ == "__main__":
    main()
