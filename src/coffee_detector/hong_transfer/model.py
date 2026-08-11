from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HongTransferConfig:
    """Frozen reconstruction choices for the Hong-to-YOLO26 transfer."""

    dsconv_bits: int = 4
    dsconv_block_size: int = 128
    dsconv_layer_indices: tuple[int, ...] = (1, 3, 17)
    sppf_layer_index: int = 9
    attention_reduction: int = 16
    pconv_ratio: float = 0.25

    @classmethod
    def from_mapping(
        cls, payload: "HongTransferConfig" | dict[str, Any] | None
    ) -> "HongTransferConfig":
        if isinstance(payload, cls):
            return payload
        values = dict(payload or {})
        if "dsconv_layer_indices" in values:
            values["dsconv_layer_indices"] = tuple(
                int(index) for index in values["dsconv_layer_indices"]
            )
        result = cls(**values)
        if not 2 <= result.dsconv_bits <= 8:
            raise ValueError("hong_transfer.dsconv_bits harus 2..8")
        if result.dsconv_block_size <= 0:
            raise ValueError("hong_transfer.dsconv_block_size harus positif")
        if not result.dsconv_layer_indices:
            raise ValueError("Minimal satu layer DSConv harus dibekukan")
        if result.sppf_layer_index < 0:
            raise ValueError("sppf_layer_index tidak boleh negatif")
        if result.attention_reduction <= 0:
            raise ValueError("attention_reduction harus positif")
        if not 0.0 < result.pconv_ratio <= 1.0:
            raise ValueError("pconv_ratio harus berada pada (0, 1]")
        return result


class DistributionShiftConv2d(nn.Module):
    """Trainable VQK/KDS/CDS reconstruction of DSConv.

    The low-bit VQK is simulated with a straight-through estimator. KDS is
    stored per output-channel/input-block/kernel position and CDS per output
    channel. This faithfully represents the paper equations in PyTorch, but
    does not claim integer-kernel latency without a deployment backend.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        bias: bool = False,
        bits: int = 4,
        block_size: int = 128,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel DSConv harus positif")
        if not 2 <= bits <= 8:
            raise ValueError("bits DSConv harus 2..8")
        if block_size <= 0:
            raise ValueError("block_size DSConv harus positif")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(int(value) for value in kernel_size)
        self.stride = tuple(int(value) for value in stride)
        self.padding = tuple(int(value) for value in padding)
        self.dilation = tuple(int(value) for value in dilation)
        self.groups = 1
        self.bits = int(bits)
        self.block_size = int(block_size)
        self.num_blocks = math.ceil(self.in_channels / self.block_size)
        self.quant_min = -(2 ** (self.bits - 1))
        self.quant_max = 2 ** (self.bits - 1) - 1

        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        shift_shape = (
            self.out_channels,
            self.num_blocks,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        self.kds_scale = nn.Parameter(torch.ones(shift_shape))
        self.kds_bias = nn.Parameter(torch.zeros(shift_shape))
        self.cds_scale = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.cds_bias = nn.Parameter(torch.zeros(self.out_channels, 1, 1, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        self.initialize_shifts_from_weight()

    def block_ranges(self) -> list[tuple[int, int]]:
        return [
            (
                block * self.block_size,
                min((block + 1) * self.block_size, self.in_channels),
            )
            for block in range(self.num_blocks)
        ]

    @torch.no_grad()
    def initialize_shifts_from_weight(self) -> None:
        """Use the DSConv least-squares KDS initialization."""

        for block_index, (start, stop) in enumerate(self.block_ranges()):
            block = self.weight[:, start:stop]
            maximum = block.abs().amax(dim=1, keepdim=True)
            step = (maximum / float(self.quant_max)).clamp_min(1.0e-8)
            quantized = torch.round(block / step).clamp(
                self.quant_min, self.quant_max
            )
            numerator = (block * quantized).sum(dim=1)
            denominator = quantized.square().sum(dim=1).clamp_min(1.0e-8)
            self.kds_scale[:, block_index].copy_(numerator / denominator)
        self.kds_bias.zero_()
        self.cds_scale.fill_(1.0)
        self.cds_bias.zero_()

    @classmethod
    def from_conv2d(
        cls,
        convolution: nn.Conv2d,
        *,
        bits: int,
        block_size: int,
    ) -> "DistributionShiftConv2d":
        if convolution.groups != 1:
            raise ValueError("DSConv Hong hanya mengganti convolution groups=1")
        result = cls(
            convolution.in_channels,
            convolution.out_channels,
            tuple(convolution.kernel_size),
            stride=tuple(convolution.stride),
            padding=tuple(convolution.padding),
            dilation=tuple(convolution.dilation),
            bias=convolution.bias is not None,
            bits=bits,
            block_size=block_size,
        ).to(device=convolution.weight.device, dtype=convolution.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(convolution.weight)
            if result.bias is not None and convolution.bias is not None:
                result.bias.copy_(convolution.bias)
            result.initialize_shifts_from_weight()
        return result

    def _quantized_block(self, block: Tensor) -> Tensor:
        maximum = block.detach().abs().amax(dim=1, keepdim=True)
        step = (maximum / float(self.quant_max)).clamp_min(1.0e-8)
        scaled = block / step
        rounded = torch.round(scaled).clamp(self.quant_min, self.quant_max)
        return scaled + (rounded - scaled).detach()

    def reconstructed_weight(self) -> Tensor:
        reconstructed = []
        for block_index, (start, stop) in enumerate(self.block_ranges()):
            quantized = self._quantized_block(self.weight[:, start:stop])
            scale = self.kds_scale[:, block_index].unsqueeze(1)
            shift = self.kds_bias[:, block_index].unsqueeze(1)
            reconstructed.append(scale * quantized + shift)
        kernel = torch.cat(reconstructed, dim=1)
        return self.cds_scale * kernel + self.cds_bias

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(
            inputs,
            self.reconstructed_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            groups=1,
        )


class DistributionShiftConvBlock(nn.Module):
    """Ultralytics Conv-compatible DSConv/BN/activation block.

    It deliberately is not an Ultralytics ``Conv`` subclass. Otherwise the
    automatic fusion path would silently fuse BN into the latent FP kernel and
    discard the VQK/KDS/CDS reconstruction.
    """

    def __init__(
        self,
        convolution: DistributionShiftConv2d,
        batch_norm: nn.Module,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.conv = convolution
        self.bn = batch_norm
        self.act = activation

    @classmethod
    def from_ultralytics_conv(
        cls, module: nn.Module, *, bits: int, block_size: int
    ) -> "DistributionShiftConvBlock":
        convolution = getattr(module, "conv", None)
        if not isinstance(convolution, nn.Conv2d):
            raise TypeError("Target DSConv bukan wrapper Conv2d Ultralytics")
        result = cls(
            DistributionShiftConv2d.from_conv2d(
                convolution, bits=bits, block_size=block_size
            ),
            copy.deepcopy(module.bn),
            copy.deepcopy(module.act),
        )
        _copy_graph_metadata(module, result)
        return result

    def forward(self, inputs: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(inputs)))


class HongSPPFAttention(nn.Module):
    """Hong SPPF-Attention while preserving pretrained SPPF projections."""

    def __init__(self, base_sppf: nn.Module, reduction: int = 16) -> None:
        super().__init__()
        for name in ("cv1", "cv2", "m"):
            if not hasattr(base_sppf, name):
                raise TypeError(f"SPPF target tidak memiliki {name}")
        self.cv1 = base_sppf.cv1
        self.cv2 = base_sppf.cv2
        self.m = base_sppf.m
        output_conv = getattr(self.cv2, "conv", None)
        if not isinstance(output_conv, nn.Conv2d):
            raise TypeError("SPPF cv2 bukan Conv2d")
        channels = int(output_conv.out_channels)
        hidden = max(1, channels // int(reduction))
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        _copy_graph_metadata(base_sppf, self)

    def forward(self, inputs: Tensor) -> Tensor:
        features = [self.cv1(inputs)]
        features.extend(self.m(features[-1]) for _ in range(3))
        fused = self.cv2(torch.cat(features, dim=1))
        if fused.shape != inputs.shape:
            raise RuntimeError(
                "Residual SPPF-Attention membutuhkan bentuk input/output sama: "
                f"{tuple(inputs.shape)} != {tuple(fused.shape)}"
            )
        channel_gate = torch.sigmoid(
            self.channel_mlp(F.adaptive_avg_pool2d(fused, 1))
        )
        channel_refined = fused * channel_gate
        spatial_descriptor = torch.cat(
            (
                channel_refined.mean(dim=1, keepdim=True),
                channel_refined.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = torch.sigmoid(self.spatial(spatial_descriptor))
        return channel_refined * spatial_gate + inputs


class PartialSpatialConv(nn.Module):
    """Apply a kxk convolution only to the first r fraction of channels."""

    def __init__(
        self,
        channels: int,
        *,
        ratio: float = 0.25,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.ratio = float(ratio)
        self.active_channels = max(1, int(math.floor(self.channels * self.ratio)))
        self.conv = nn.Conv2d(
            self.active_channels,
            self.active_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        active, bypass = torch.split(
            inputs,
            (self.active_channels, self.channels - self.active_channels),
            dim=1,
        )
        active = self.conv(active)
        return torch.cat((active, bypass), dim=1) if bypass.shape[1] else active


class PartialConvBlock(nn.Module):
    """PConv spatial subset followed by a 1x1 cross-channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        ratio: float,
        batch_norm: nn.Module,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self.partial = PartialSpatialConv(in_channels, ratio=ratio)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = batch_norm
        self.act = activation

    @classmethod
    def from_full_conv(
        cls, source: nn.Module, *, ratio: float
    ) -> "PartialConvBlock":
        convolution = getattr(source, "conv", None)
        if not isinstance(convolution, nn.Conv2d) or tuple(convolution.kernel_size) != (3, 3):
            raise TypeError("PConv box branch memerlukan wrapper Conv 3x3")
        result = cls(
            convolution.in_channels,
            convolution.out_channels,
            ratio=ratio,
            batch_norm=copy.deepcopy(source.bn),
            activation=copy.deepcopy(source.act),
        ).to(device=convolution.weight.device, dtype=convolution.weight.dtype)
        active = result.partial.active_channels
        with torch.no_grad():
            # There is no exact factorization from a dense c_out x c_in x 3x3
            # kernel to PConv. Start the partial spatial operator as identity
            # and transfer the source kernel's central cross-channel mixing to
            # the following pointwise layer. The audit labels this as a derived
            # initialization, not an exactly loaded pretrained tensor.
            result.partial.conv.weight.zero_()
            diagonal = torch.arange(active, device=convolution.weight.device)
            result.partial.conv.weight[diagonal, diagonal, 1, 1] = 1.0
            result.pointwise.weight.copy_(
                convolution.weight[:, :, 1:2, 1:2]
            )
        return result

    @classmethod
    def from_depthwise_pointwise(
        cls, source: nn.Module, *, ratio: float
    ) -> "PartialConvBlock":
        if not isinstance(source, nn.Sequential) or len(source) != 2:
            raise TypeError("PConv class branch memerlukan DWConv+PWConv pair")
        depthwise = getattr(source[0], "conv", None)
        pointwise = getattr(source[1], "conv", None)
        if (
            not isinstance(depthwise, nn.Conv2d)
            or depthwise.groups != depthwise.in_channels
            or tuple(depthwise.kernel_size) != (3, 3)
            or not isinstance(pointwise, nn.Conv2d)
            or tuple(pointwise.kernel_size) != (1, 1)
        ):
            raise TypeError("Struktur class branch bukan DWConv3x3+PWConv1x1")
        result = cls(
            depthwise.in_channels,
            pointwise.out_channels,
            ratio=ratio,
            batch_norm=copy.deepcopy(source[1].bn),
            activation=copy.deepcopy(source[1].act),
        ).to(device=depthwise.weight.device, dtype=depthwise.weight.dtype)
        active = result.partial.active_channels
        with torch.no_grad():
            result.partial.conv.weight.zero_()
            diagonal = torch.arange(active, device=depthwise.weight.device)
            result.partial.conv.weight[diagonal, diagonal].copy_(
                depthwise.weight[:active, 0]
            )
            result.pointwise.weight.copy_(pointwise.weight)
        return result

    def forward(self, inputs: Tensor) -> Tensor:
        return self.act(self.bn(self.pointwise(self.partial(inputs))))


def _copy_graph_metadata(source: nn.Module, target: nn.Module) -> None:
    for name in ("i", "f", "type", "np"):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def _patch_head_branches(head: nn.Module, ratio: float) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    for branch_name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branches = getattr(head, branch_name, None)
        if not isinstance(branches, nn.ModuleList) or len(branches) != 3:
            raise TypeError(f"Head YOLO26 tidak memiliki tiga level {branch_name}")
        paths[branch_name] = []
        for level, branch in enumerate(branches):
            if not isinstance(branch, nn.Sequential) or len(branch) < 3:
                raise TypeError(f"Struktur {branch_name}.{level} tidak kompatibel")
            for block_index in (0, 1):
                current = branch[block_index]
                if isinstance(current, PartialConvBlock):
                    continue
                replacement = (
                    PartialConvBlock.from_full_conv(current, ratio=ratio)
                    if branch_name.endswith("cv2")
                    else PartialConvBlock.from_depthwise_pointwise(
                        current, ratio=ratio
                    )
                )
                branch[block_index] = replacement
                paths[branch_name].append(
                    f"model.23.{branch_name}.{level}.{block_index}"
                )
    return paths


def inject_hong_transfer(
    model: nn.Module,
    config: HongTransferConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject the full Hong package into a pretrained YOLO26 model once."""

    frozen = HongTransferConfig.from_mapping(config)
    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Model tidak memiliki graph Ultralytics `.model`")
    existing = getattr(model, "hong_transfer_config", None)
    if existing is not None:
        if HongTransferConfig.from_mapping(existing) != frozen:
            raise RuntimeError("Checkpoint Hong memakai konfigurasi berbeda")
        return dict(getattr(model, "hong_transfer_audit"))

    dsconv_paths = []
    for index in frozen.dsconv_layer_indices:
        if index >= len(layers):
            raise IndexError(f"Index DSConv di luar graph: {index}")
        source = layers[index]
        if isinstance(source, DistributionShiftConvBlock):
            raise RuntimeError("DSConv sudah terpasang tanpa marker konfigurasi")
        replacement = DistributionShiftConvBlock.from_ultralytics_conv(
            source,
            bits=frozen.dsconv_bits,
            block_size=frozen.dsconv_block_size,
        )
        layers[index] = replacement
        dsconv_paths.append(f"model.{index}")

    sppf_index = frozen.sppf_layer_index
    if sppf_index >= len(layers):
        raise IndexError(f"Index SPPF di luar graph: {sppf_index}")
    layers[sppf_index] = HongSPPFAttention(
        layers[sppf_index], reduction=frozen.attention_reduction
    )

    head = layers[-1]
    if type(head).__name__ != "Detect" or not getattr(head, "end2end", False):
        raise TypeError("Hong transfer dikunci untuk Detect end-to-end YOLO26")
    prediction_paths = [
        f"model.{len(layers) - 1}.{branch}.{level}.2"
        for branch in ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
        for level in range(3)
    ]
    pconv_paths = _patch_head_branches(head, frozen.pconv_ratio)

    audit = {
        "format": "coffee_detector.hong_transfer.architecture.v1",
        "config": asdict(frozen),
        "dsconv_paths": dsconv_paths,
        "sppf_attention_path": f"model.{sppf_index}",
        "pconv_paths": pconv_paths,
        "preserved_prediction_paths": prediction_paths,
        "end2end": bool(head.end2end),
        "levels": int(head.nl),
        "reg_max": int(head.reg_max),
        "initialization": {
            "dsconv": (
                "latent FP kernel copied exactly; KDS initialized by blockwise "
                "least squares; KDS/CDS bias zero and CDS scale one"
            ),
            "sppf_attention": (
                "SPPF cv1/cv2/max-pool reused exactly; channel and spatial "
                "attention layers newly initialized"
            ),
            "pconv_box": (
                "partial spatial kernel initialized as identity; pointwise "
                "projection derived from the source 3x3 centre coefficients; "
                "BN and activation copied"
            ),
            "pconv_class": (
                "active-channel spatial diagonals copied from source DWConv; "
                "pointwise projection, BN, and activation copied exactly"
            ),
            "prediction_layers": "terminal 1x1 modules and hashes preserved exactly",
        },
    }
    model.hong_transfer_config = asdict(frozen)
    model.hong_transfer_audit = audit
    return audit
