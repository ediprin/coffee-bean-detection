from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LocalBilinearAdapter(nn.Module):
    """Spatially preserving bilinear residual adapter.

    Unlike the global HBP classifier, this block never pools ``H x W``. It is
    therefore safe to place only in a YOLO classification branch while the box
    regression branch continues to receive the original feature map.
    """

    def __init__(self, channels: int, rank: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0 or rank <= 0:
            raise ValueError("channels dan rank harus positif")
        self.channels = int(channels)
        self.rank = min(int(rank), self.channels)
        self.eps = float(eps)
        self.left = nn.Conv2d(self.channels, self.rank, kernel_size=1, bias=False)
        self.right = nn.Conv2d(self.channels, self.rank, kernel_size=1, bias=False)
        self.output = nn.Conv2d(self.rank, self.channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0))
        nn.init.kaiming_normal_(self.left.weight, mode="fan_out", nonlinearity="linear")
        nn.init.kaiming_normal_(self.right.weight, mode="fan_out", nonlinearity="linear")
        # Zero initialization preserves the pretrained baseline at injection.
        nn.init.zeros_(self.output.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"LocalBilinearAdapter memerlukan BCHW, diterima {features.shape}")
        bilinear = self.left(features) * self.right(features)
        bilinear = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + self.eps)
        bilinear = F.normalize(bilinear, p=2, dim=1, eps=self.eps)
        return features + self.scale * self.output(bilinear)


class LocalHBPClassBranch(nn.Module):
    """Wrap one YOLO class branch without changing its output contract."""

    def __init__(self, channels: int, branch: nn.Module, rank: int = 64) -> None:
        super().__init__()
        self.adapter = LocalBilinearAdapter(channels, rank=rank)
        self.branch = branch

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.branch(self.adapter(features))


def _first_conv_in_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada classification branch {type(module).__name__}")


def _patch_module_list(branches: nn.ModuleList, rank: int) -> int:
    patched = 0
    for index, branch in enumerate(list(branches)):
        if isinstance(branch, LocalHBPClassBranch):
            continue
        channels = _first_conv_in_channels(branch)
        branches[index] = LocalHBPClassBranch(channels, branch, rank=rank)
        patched += 1
    return patched


def inject_local_hbp(model: nn.Module, rank: int = 64) -> int:
    """Inject adapters only into Ultralytics ``Detect.cv3`` class branches.

    The function intentionally leaves ``cv2`` (box regression) untouched.
    It accepts a lightweight fake model as well, which keeps the contract easy
    to unit-test without importing Ultralytics.
    """

    layers = getattr(model, "model", None)
    if layers is None or len(layers) == 0:
        raise TypeError("Model tidak memiliki urutan layer `.model`")
    head = layers[-1]
    class_branches = getattr(head, "cv3", None)
    if not isinstance(class_branches, nn.ModuleList):
        raise TypeError("Detection head tidak memiliki ModuleList `cv3`; versi Ultralytics tidak kompatibel")
    patched = _patch_module_list(class_branches, rank)

    # End-to-end heads may keep a separate one-to-one classification branch.
    one_to_one = getattr(head, "one2one_cv3", None)
    if isinstance(one_to_one, nn.ModuleList):
        patched += _patch_module_list(one_to_one, rank)
    head.local_hbp_rank = int(rank)
    return patched


def make_local_hbp_trainer(rank: int = 64):
    """Build an Ultralytics trainer class that injects before optimizer setup."""

    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("Ultralytics belum terpasang. Jalankan `pip install -e .`.") from error

    class LocalHBPDetectionTrainer(DetectionTrainer):
        local_hbp_rank = int(rank)

        def get_model(self, cfg=None, weights=None, verbose=True):
            detector = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            patched = inject_local_hbp(detector, rank=self.local_hbp_rank)
            already_patched = any(
                isinstance(branch, LocalHBPClassBranch)
                for branch in getattr(detector.model[-1], "cv3", [])
            )
            if patched == 0 and not already_patched:
                raise RuntimeError("Tidak ada classification branch yang berhasil dipasang local-HBP")
            return detector

    LocalHBPDetectionTrainer.__name__ = f"LocalHBPDetectionTrainerRank{int(rank)}"
    return LocalHBPDetectionTrainer
