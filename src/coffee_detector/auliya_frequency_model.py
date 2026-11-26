"""Parameter-free input operators for a matched YOLO segmentation comparison."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.tasks import SegmentationModel

from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer


ARMS = ("RGB", "AF2", "HAAR")


class HaarLuminanceEnhancer(nn.Module):
    """Boost signed Haar detail at two scales using a shared RGB offset."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        work = image.float()
        y = 0.2126 * work[:, :1] + 0.7152 * work[:, 1:2] + 0.0722 * work[:, 2:]
        y = F.pad(y, (0, -width % 4, 0, -height % 4), mode="replicate")
        # Average pooling and replication reconstruct the low-pass Haar approximation.
        ll1 = F.avg_pool2d(y, kernel_size=2)
        ll2 = F.avg_pool2d(ll1, kernel_size=2)
        coarse1 = ll1.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        coarse2 = ll2.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
        delta = (0.25 * (y - coarse1) + 0.125 * (coarse1 - coarse2))[..., :height, :width]
        # Bound the common offset before adding it, preserving channel differences.
        delta = delta.maximum(-work.amin(dim=1, keepdim=True)).minimum(
            1.0 - work.amax(dim=1, keepdim=True)
        )
        return (work + delta).to(image.dtype)


class FrequencySegmentationModel(SegmentationModel):
    """Apply the chosen fixed operator before the first YOLO layer."""

    def __init__(self, cfg, *, arm: str, ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.frequency_arm = arm
        if arm == "RGB":
            self.input_enhancer = nn.Identity()
        elif arm == "AF2":
            self.input_enhancer = AFABInputEnhancer(AFABConfig(mode="af2"))
        elif arm == "HAAR":
            self.input_enhancer = HaarLuminanceEnhancer()
        else:
            raise ValueError(f"Unknown arm: {arm}")

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        # Ultralytics infers strides via dummy input during super().__init__.
        if hasattr(self, "input_enhancer"):
            x = self.input_enhancer(x)
        return super().predict(x, profile=profile, visualize=visualize, augment=augment, embed=embed)
