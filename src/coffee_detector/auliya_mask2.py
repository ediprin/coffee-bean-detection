"""Stride-2 residual prototype masks for the Auliya HAAR model (Ultralytics 8.4.210)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.block import Proto26
from ultralytics.nn.modules.head import Segment26
from ultralytics.utils.ops import process_mask
from ultralytics.utils.torch_utils import initialize_weights

from coffee_detector.auliya_frequency_model import FrequencySegmentationModel

MASK2_ARM = "HAAR_MASK2"
MASK2_PROTOCOL = "auliya-official-fold-a-haar-mask2-v1"


class SubpixelProto26(Proto26):
    """Keep the original prototypes and add a zero-start, local subpixel correction."""

    def __init__(self, head: Segment26):
        original = head.proto
        channels = (original.feat_fuse.conv.in_channels, *(m.conv.in_channels for m in original.feat_refine))
        super().__init__(channels, head.npr, head.nm, head.nc)
        initialize_weights(self)  # match native BatchNorm epsilon/momentum, which are not state tensors
        self.detail = nn.Conv2d(head.nm, 4 * head.nm, 3, padding=1, groups=head.nm, bias=False)
        nn.init.zeros_(self.detail.weight)
        incompatible = self.load_state_dict(original.state_dict(), strict=False)
        if incompatible.missing_keys != ["detail.weight"] or incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected prototype state changes: {incompatible}")

    def forward(self, x, return_semantic=True):
        value = super().forward(x, return_semantic=return_semantic)
        proto, semantic = value if isinstance(value, tuple) else (value, None)
        fine = F.interpolate(proto, scale_factor=2, mode="nearest") + F.pixel_shuffle(self.detail(proto), 2)
        return (fine, semantic) if isinstance(value, tuple) else fine


class HaarMask2Model(FrequencySegmentationModel):
    """Same HAAR frontend and detector heads, changing only mask prototype resolution."""

    def __init__(self, cfg, *, ch=3, nc=None, verbose=True):
        super().__init__(cfg, arm="HAAR", ch=ch, nc=nc, verbose=verbose)
        self.frequency_arm = MASK2_ARM
        head = self.model[-1]
        if type(head) is not Segment26 or type(head.proto) is not Proto26:
            raise TypeError("Expected an unmodified YOLO26 Segment26/Proto26 head")
        with torch.random.fork_rng(devices=[]):
            head.proto = SubpixelProto26(head)


def prototypes_at_grid(proto: torch.Tensor, input_hw: tuple[int, int], ratio: int) -> torch.Tensor:
    """Map prototype logits to the selected common metric raster, before cropping/thresholding."""
    hw = tuple(map(int, input_hw))
    if ratio not in (2, 4) or any(i % 4 for i in hw):
        raise ValueError(f"Unsupported mask evaluation grid: {hw}, stride {ratio}")
    expected = tuple(i // ratio for i in hw)
    if proto.shape[-2:] == expected:
        return proto
    if proto.shape[-2:] == tuple(i // 2 for i in hw) and ratio == 4:
        # Algebraically a 2x2 mean, expressed around one sample so nearest-
        # repeated logits reduce to the original bitwise at zero residual.
        a = proto[..., ::2, ::2]
        return a + 0.25 * ((proto[..., ::2, 1::2] - a)
            + (proto[..., 1::2, ::2] - a) + (proto[..., 1::2, 1::2] - a))
    if proto.shape[-2:] == tuple(i // 4 for i in hw) and ratio == 2:
        return F.interpolate(proto, size=expected, mode="bilinear", align_corners=False)
    raise ValueError(f"Unexpected prototype dimensions {tuple(proto.shape[-2:])} for input {hw}")


class CommonMaskGridValidator(SegmentationValidator):
    """Use identical crops and direct polygon targets at stride 2 or stride 4 for either model."""

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        self.input_hw = tuple(batch["img"].shape[-2:])
        return batch

    def postprocess(self, preds):
        if self.args.tile or self.args.save_json or self.args.save_txt:
            raise ValueError("This validation-only audit does not support tiling or exporting masks")
        det, proto = preds[0] if isinstance(preds[0], tuple) else preds
        proto = prototypes_at_grid(proto, self.input_hw, self.args.mask_ratio)
        results = DetectionValidator.postprocess(self, det)
        for i, pred in enumerate(results):
            pred["masks"] = process_mask(proto[i], pred.pop("extra"), pred["bboxes"], shape=self.input_hw)
        return results

    def _prepare_batch(self, si, batch):
        # Upstream's segmentation implementation hardcodes stride 4. Refuse
        # resampled targets here: these must be newly rasterized from polygons.
        prepared = DetectionValidator._prepare_batch(self, si, batch)
        masks = batch["masks"]
        hw = tuple(s // self.args.mask_ratio for s in prepared["imgsz"])
        if masks.shape[-2:] != hw:
            raise RuntimeError(f"Dataloader masks {tuple(masks.shape[-2:])} do not match eval grid {hw}")
        if self.args.overlap_mask:
            n = prepared["cls"].shape[0]
            indices = torch.arange(1, n + 1, device=masks.device).view(n, 1, 1)
            prepared["masks"] = (masks[si] == indices).float()
        else:
            prepared["masks"] = masks[batch["batch_idx"] == si]
        return prepared
