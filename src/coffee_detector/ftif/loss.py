from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import FTIFConfig, FTIFDetectHead


def bidirectional_alignment_loss(
    similarity: torch.Tensor,
    fg_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """LFDet Eq. (20) transfer on native YOLO dense assignments.

    similarity: [B,N,K] cosine/tau from Eq. (19)
    fg_mask: [B,N] native TAL positives
    labels: [B,N] assigned leaf class for positive locations

    Paper-derived loss forms:
      * positive I->T: softmax CE over category texts;
      * negative I->T: sigmoid BCE because background has no text match;
      * T->I: binary sigmoid/BCE, separated into positive/negative terms;
      * Eq. (20) averages the two directional positive and negative groups.

    Transfer-level averaging choice: each positive/negative term is mean-reduced
    independently to avoid the massive dense background count changing its
    coefficient implicitly.
    """
    if similarity.ndim != 3:
        raise ValueError("FTIF similarity harus [B,N,K]")
    if fg_mask.shape != similarity.shape[:2] or labels.shape != fg_mask.shape:
        raise ValueError("Shape FTIF assignment tidak sejajar")
    num_classes = similarity.shape[-1]
    zero = similarity.sum() * 0.0

    if bool(fg_mask.any()):
        positive_logits = similarity[fg_mask]
        positive_labels = labels[fg_mask].long()
        pos_i2t = F.cross_entropy(positive_logits.float(), positive_labels, reduction="mean")
        matched = positive_logits.gather(1, positive_labels[:, None])
        pos_t2i = F.binary_cross_entropy_with_logits(
            matched.float(), torch.ones_like(matched, dtype=torch.float32), reduction="mean"
        )
    else:
        pos_i2t = zero
        pos_t2i = zero

    background = ~fg_mask
    if bool(background.any()):
        negative_visual = similarity[background]
        neg_i2t = F.binary_cross_entropy_with_logits(
            negative_visual.float(),
            torch.zeros_like(negative_visual, dtype=torch.float32),
            reduction="mean",
        )
    else:
        neg_i2t = zero

    # Text-to-image binary alignment: every matched (visual,class) cell is a
    # positive; all other class/location cells are negatives.
    target = torch.zeros_like(similarity, dtype=torch.bool)
    if bool(fg_mask.any()):
        b_index, n_index = fg_mask.nonzero(as_tuple=True)
        target[b_index, n_index, labels[fg_mask].long()] = True
    if bool(target.any()):
        pos_t2i_full = F.binary_cross_entropy_with_logits(
            similarity[target].float(),
            torch.ones_like(similarity[target], dtype=torch.float32),
            reduction="mean",
        )
        # Keep the explicit T->I positive calculation from the full matrix.
        pos_t2i = pos_t2i_full
    if bool((~target).any()):
        neg_t2i = F.binary_cross_entropy_with_logits(
            similarity[~target].float(),
            torch.zeros_like(similarity[~target], dtype=torch.float32),
            reduction="mean",
        )
    else:
        neg_t2i = zero

    total = 0.5 * (pos_i2t + pos_t2i) + 0.5 * (neg_i2t + neg_t2i)
    terms = {
        "positive_i2t": pos_i2t.detach(),
        "negative_i2t": neg_i2t.detach(),
        "positive_t2i": pos_t2i.detach(),
        "negative_t2i": neg_t2i.detach(),
    }
    return total, terms


class FTIFDetectionLoss:
    """Native Ultralytics loss plus optional LFDet bidirectional alignment."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundFTIFDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, FTIFDetectHead):
                    raise TypeError("FTIF loss memerlukan FTIFDetectHead")
                self.ftif_head = head
                self.ftif_config = FTIFConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                similarity = preds.get("ftif_similarity")
                if not self.ftif_config.bidirectional_alignment:
                    if similarity is not None:
                        raise RuntimeError("Similarity tidak boleh dibuat saat alignment nonaktif")
                    return assignments, loss, loss.detach()
                if similarity is None:
                    raise RuntimeError("FTIF alignment aktif tetapi similarity matrix hilang")

                fg_mask, target_gt_idx = assignments[:2]
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                if similarity.shape[:2] != pred_scores.shape[:2] or similarity.shape[2] != self.nc:
                    raise RuntimeError("FTIF similarity tidak sejajar dengan dense predictions")

                batch_size = pred_scores.shape[0]
                image_size = (
                    torch.tensor(
                        preds["feats"][0].shape[2:],
                        device=self.device,
                        dtype=pred_scores.dtype,
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
                auxiliary, _ = bidirectional_alignment_loss(
                    similarity, fg_mask, assigned_labels
                )
                loss[1] = loss[1] + float(self.ftif_config.alignment_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundFTIFDetectionLoss()
