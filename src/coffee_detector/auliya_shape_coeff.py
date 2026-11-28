"""Mask-coefficient shape supervision and a bounded, class-7-only residual."""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.head import Segment26
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss


ARM = "SHAPECOEF1"
FOCUS, SHAPE_CLASSES = 7, frozenset({7, 12, 13, 14})
CENTER, SPAN, ASSIST_WEIGHT = .96, .04, .10
MAX_RESIDUAL = math.log(4)
MIN_PIXELS = 24


def targets_from_masks(batch, *, overlap):
    """Compute solidity on the actual augmented native mask raster (CPU, train only)."""
    classes = batch['cls'].reshape(-1).cpu().numpy().astype(int)
    image_id = batch['batch_idx'].reshape(-1).cpu().numpy().astype(int)
    masks = batch['masks'].cpu().numpy()
    targets = np.zeros(len(classes), dtype=np.float32)
    valid = np.zeros(len(classes), dtype=bool)
    if (not overlap and masks.shape[0] != len(classes)) or (overlap and len(masks) != len(batch['img'])):
        raise ValueError("Mask instances do not follow native training batch layout")
    counts = {}
    for i, (cls, im) in enumerate(zip(classes, image_id, strict=True)):
        counts[im] = counts.get(im, 0) + 1
        if cls not in SHAPE_CLASSES:
            continue
        mask = (masks[im] == counts[im]) if overlap else (masks[i] > 0)
        y, x = np.nonzero(mask)
        if y.size < MIN_PIXELS:
            continue
        crop = mask[y.min():y.max()+1, x.min():x.max()+1].astype(np.uint8)
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            continue
        contour = contours[0]
        area, hull = cv2.contourArea(contour), cv2.contourArea(cv2.convexHull(contour))
        if min(area, hull) <= 0:
            continue
        targets[i] = np.clip((area/hull - CENTER)/SPAN, -1, 1)
        valid[i] = True
    return torch.from_numpy(targets), torch.from_numpy(valid)


class ShapeProjection(nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        # Avoid perturbing the RNG stream of the matched native initialization.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv1d(coefficients, 1, kernel_size=1)
            nn.init.normal_(self.projection.weight, std=.02)
            nn.init.zeros_(self.projection.bias)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, coefficients):
        shape = self.projection(coefficients.detach()).squeeze(1).tanh()
        return shape, shape * (self.gain.tanh() * MAX_RESIDUAL)


class ShapeSegmentHead(Segment26):
    @property
    def one2many(self):
        return {**super().one2many, "projector": self.shape_one2many}

    @property
    def one2one(self):
        return {**super().one2one, "projector": self.shape_one2one}

    def forward_head(self, x, box_head, cls_head, mask_head, projector=None):
        pred = super().forward_head(x, box_head, cls_head, mask_head)
        if projector is None or "mask_coefficient" not in pred:
            return pred
        shape, correction = projector(pred['mask_coefficient'])
        scores = pred['scores']
        pred['scores'] = torch.cat((scores[:, :FOCUS], scores[:, FOCUS:FOCUS+1] + correction.unsqueeze(1),
                                   scores[:, FOCUS+1:]), dim=1)
        if self.training:
            pred['shape_readout'] = shape
        return pred

    def fuse(self):
        super().fuse()
        self.shape_one2many = None


def shape_regression(shape, foreground, target_gt_index, batch):
    if 'shape_target' not in batch or not frontend_data_available(batch):
        return shape.float().sum() * 0.
    if not foreground.any():
        return shape.float().sum() * 0.
    image_id = batch['batch_idx'].reshape(-1).to(device=shape.device, dtype=torch.long)
    sizes = torch.bincount(image_id, minlength=shape.shape[0])
    offsets = sizes.cumsum(0) - sizes
    gt = (offsets[:, None] + target_gt_index.long())[foreground]
    valid = batch['shape_valid'].to(shape.device)[gt]
    target = batch['shape_target'].to(shape.device)[gt]
    errors = F.smooth_l1_loss(shape[foreground].float(), target.float(), reduction='none', beta=.5)
    return (errors * valid).sum() / valid.sum().clamp_min(1)


def frontend_data_available(batch):
    return 'shape_valid' in batch and bool(batch['shape_target'].numel())


class ShapeSegmentLoss(v8SegmentationLoss):
    def get_assigned_targets_and_loss(self, preds, batch):
        assignment, loss, items = super().get_assigned_targets_and_loss(preds, batch)
        if 'shape_readout' not in preds or 'shape_target' not in batch:
            return assignment, loss, items
        extra = shape_regression(preds['shape_readout'], assignment[0], assignment[1], batch)
        total = torch.stack((loss[0], loss[1] + self.hyp.cls * ASSIST_WEIGHT * extra, loss[2]))
        return assignment, total, total.detach()


class ShapeCoeffModel(SegmentationModel):
    def __init__(self, cfg, *, ch=3, nc=21, verbose=True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        if type(head) is not Segment26 or not head.end2end or nc != 21:
            raise TypeError("Expected the 21-class native YOLO26-seg dual head")
        head.__class__ = ShapeSegmentHead
        head.shape_one2many = ShapeProjection(head.nm)
        head.shape_one2one = ShapeProjection(head.nm)
        self.frequency_arm = ARM

    def init_criterion(self):
        return E2ELoss(self, ShapeSegmentLoss)
