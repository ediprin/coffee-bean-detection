from __future__ import annotations

import torch
import torch.nn.functional as F

from coffee_detector.sni21_ontology import SNI21_CLASSES, load_sni21_ontology

from .config import AF2ComplementConfig


def balanced_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Class-balanced supervised contrastive loss with a safe zero fallback."""

    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("embeddings/labels harus [N,D] dan [N]")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Jumlah embeddings dan labels tidak sama")
    if temperature <= 0:
        raise ValueError("temperature harus positif")
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    features = F.normalize(embeddings.float(), dim=1)
    similarity = features @ features.T / float(temperature)
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    nonself = ~identity
    positive = labels[:, None].eq(labels[None, :]) & nonself
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0

    log_denominator = torch.logsumexp(similarity.masked_fill(identity, -torch.inf), dim=1)
    log_probability = similarity - log_denominator[:, None]
    per_anchor = -(
        log_probability.masked_fill(~positive, 0.0).sum(dim=1)
        / positive.sum(dim=1).clamp_min(1)
    )
    _, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)
    weights = counts[inverse].float().reciprocal()
    weights = weights[valid] / weights[valid].sum().clamp_min(1e-12)
    return (per_anchor[valid] * weights).sum().to(embeddings.dtype)


def _family_mapping() -> torch.Tensor:
    ontology = load_sni21_ontology()
    values = sorted({str(row["entity_family"]) for row in ontology["classes"].values()})
    value_to_id = {value: index for index, value in enumerate(values)}
    return torch.tensor(
        [
            value_to_id[str(ontology["classes"][class_name]["entity_family"])]
            for class_name in SNI21_CLASSES
        ],
        dtype=torch.long,
    )


def _aggregate_gt_logits(
    pred_scores: torch.Tensor,
    foreground: torch.Tensor,
    target_gt_index: torch.Tensor,
    gt_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average assigned positive logits so each GT object contributes once."""

    embeddings, labels = [], []
    for image_index in range(pred_scores.shape[0]):
        mask = foreground[image_index]
        if not bool(mask.any()):
            continue
        assigned = target_gt_index[image_index, mask].long()
        scores = pred_scores[image_index, mask]
        for gt_index in torch.unique(assigned, sorted=True):
            selected = assigned.eq(gt_index)
            embeddings.append(scores[selected].mean(dim=0))
            labels.append(gt_labels[image_index, gt_index])
    if not embeddings:
        return pred_scores.new_zeros((0, pred_scores.shape[-1])), gt_labels.new_zeros((0,))
    return torch.stack(embeddings), torch.stack(labels).long()


class AF2ComplementDetectionLoss:
    """Native YOLO loss plus balanced leaf/family contrastive supervision."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundAF2ComplementLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                self.complement = AF2ComplementConfig.from_mapping(model.af2_complement_config)
                self.apply_auxiliary = self.complement.mode == "bhcl" and tal_topk2 is None
                self.family_mapping = _family_mapping()

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                if not self.apply_auxiliary:
                    return assignments, loss, loss.detach()
                foreground, target_gt_index = assignments[:2]
                if not bool(foreground.any()):
                    return assignments, loss, loss.detach()

                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                image_size = (
                    torch.tensor(
                        preds["feats"][0].shape[2:],
                        device=self.device,
                        dtype=pred_scores.dtype,
                    )
                    * self.stride[0]
                )
                targets = torch.cat(
                    (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
                    1,
                )
                targets = self.preprocess(
                    targets.to(self.device),
                    pred_scores.shape[0],
                    scale_tensor=image_size[[1, 0, 1, 0]],
                )
                gt_labels = targets[..., 0].long()
                embeddings, labels = _aggregate_gt_logits(
                    pred_scores, foreground, target_gt_index, gt_labels
                )
                if not labels.numel():
                    return assignments, loss, loss.detach()

                leaf = balanced_supervised_contrastive_loss(
                    embeddings,
                    labels,
                    temperature=self.complement.contrastive_temperature,
                )
                families = self.family_mapping.to(labels.device)[labels]
                family = balanced_supervised_contrastive_loss(
                    embeddings,
                    families,
                    temperature=self.complement.contrastive_temperature,
                )
                mix = float(self.complement.family_gain)
                auxiliary = (1.0 - mix) * leaf + mix * family
                loss[1] = loss[1] + float(self.complement.contrastive_gain) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundAF2ComplementLoss()
