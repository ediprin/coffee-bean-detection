from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GDSClsConfig, GDSClsDetectHead


def axis_aligned_grid_distance(
    candidate_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    *,
    grid_size: int = 7,
) -> torch.Tensor:
    """Horizontal specialization of Zhao et al. Eq. (1)-(3).

    Both boxes are divided into k x k uniformly spaced bin centers. Because the
    coffee transfer uses horizontal boxes rather than rotated boxes, ordered grid
    correspondence is used instead of the source paper's Hungarian matching.
    Distances are summed and normalized by k^2 times candidate box area, matching
    the source equation's normalization form. This is explicitly a transfer
    specialization rather than a literal rotated-anchor reproduction.
    """
    if candidate_xyxy.shape != target_xyxy.shape or candidate_xyxy.ndim != 2 or candidate_xyxy.shape[1] != 4:
        raise ValueError("grid distance memerlukan dua tensor [N,4] berukuran sama")
    if grid_size <= 0:
        raise ValueError("grid_size harus positif")
    if not len(candidate_xyxy):
        return candidate_xyxy.new_zeros((0,))

    candidate = candidate_xyxy.float()
    target = target_xyxy.float()
    fractions = (
        torch.arange(grid_size, device=candidate.device, dtype=candidate.dtype) + 0.5
    ) / float(grid_size)

    c_w = (candidate[:, 2] - candidate[:, 0]).clamp_min(1e-6)
    c_h = (candidate[:, 3] - candidate[:, 1]).clamp_min(1e-6)
    t_w = (target[:, 2] - target[:, 0]).clamp_min(1e-6)
    t_h = (target[:, 3] - target[:, 1]).clamp_min(1e-6)

    c_x = candidate[:, 0, None] + fractions[None, :] * c_w[:, None]
    c_y = candidate[:, 1, None] + fractions[None, :] * c_h[:, None]
    t_x = target[:, 0, None] + fractions[None, :] * t_w[:, None]
    t_y = target[:, 1, None] + fractions[None, :] * t_h[:, None]

    dx2 = (c_x - t_x).square()[:, :, None]
    dy2 = (c_y - t_y).square()[:, None, :]
    total = torch.sqrt(dx2 + dy2 + 1e-12).sum(dim=(1, 2))
    normalization = float(grid_size * grid_size) * c_w * c_h
    return total / normalization


class GDSAuxDetectionLoss:
    """Native TAL loss plus GDS-positive classification supervision.

    Only the one-to-many loss (tal_topk > 1) receives the auxiliary term. Native
    TAL foreground assignments, box loss, and DFL are untouched. Among those
    positives, decoded boxes satisfying D_grid < T receive an additional one-hot
    classification BCE. This tests the GDS classification-potential hypothesis
    without pretending YOLO26 has the predefined anchor-box geometry of S2A-Net.
    """

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics.utils.tal import make_anchors

        class _BoundGDSAuxLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, GDSClsDetectHead):
                    raise TypeError("GDS auxiliary loss memerlukan GDSClsDetectHead")
                self.gds_config = GDSClsConfig.from_mapping(head.config)
                self.enable_gds_aux = int(tal_topk) > 1

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                if not self.enable_gds_aux:
                    return assignments, loss, loss.detach()

                fg_mask, target_gt_idx, target_bboxes, _, _ = assignments
                if not bool(fg_mask.any()):
                    return assignments, loss, loss.detach()

                pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
                decoded = self.bbox_decode(anchor_points, pred_distri) * stride_tensor

                distances = axis_aligned_grid_distance(
                    decoded[fg_mask].detach(),
                    target_bboxes[fg_mask].detach(),
                    grid_size=int(self.gds_config.grid_size),
                )
                selected = distances.lt(float(self.gds_config.threshold))
                if not bool(selected.any()):
                    return assignments, loss, loss.detach()

                dtype = pred_scores.dtype
                batch_size = pred_scores.shape[0]
                image_size = (
                    torch.tensor(
                        preds["feats"][0].shape[2:], device=self.device, dtype=dtype
                    )
                    * self.stride[0]
                )
                targets = torch.cat(
                    (
                        batch["batch_idx"].view(-1, 1),
                        batch["cls"].view(-1, 1),
                        batch["bboxes"],
                    ),
                    1,
                )
                targets = self.preprocess(
                    targets.to(self.device),
                    batch_size,
                    scale_tensor=image_size[[1, 0, 1, 0]],
                )
                gt_labels = targets[..., 0].long()
                assigned_labels = gt_labels.gather(1, target_gt_idx.long())
                selected_labels = assigned_labels[fg_mask][selected]
                selected_logits = pred_scores[fg_mask][selected]
                one_hot = F.one_hot(selected_labels, num_classes=self.nc).to(selected_logits.dtype)
                auxiliary = F.binary_cross_entropy_with_logits(
                    selected_logits, one_hot, reduction="mean"
                )
                loss[1] = loss[1] + (
                    float(self.gds_config.auxiliary_weight) * auxiliary * float(self.hyp.cls)
                )
                return assignments, loss, loss.detach()

        return _BoundGDSAuxLoss()
