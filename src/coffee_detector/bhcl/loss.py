from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import BHCLConfig, BHCLDetectHead


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    """Fail at the first non-finite BHCL tensor with actionable diagnostics."""
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    finite_values = tensor.detach()[finite]
    minimum = float(finite_values.min()) if finite_values.numel() else float("nan")
    maximum = float(finite_values.max()) if finite_values.numel() else float("nan")
    raise FloatingPointError(
        f"BHCL non-finite tensor: {name}; shape={tuple(tensor.shape)}; "
        f"finite={int(finite.sum())}/{tensor.numel()}; min_finite={minimum:.6g}; "
        f"max_finite={maximum:.6g}"
    )


def hierarchical_contrastive_loss(
    embeddings: torch.Tensor,
    leaf_labels: torch.Tensor,
    prototype_bank,
    config: BHCLConfig,
) -> torch.Tensor:
    """HCL Eq. (7) without prototype-based class balancing.

    Eq. (6) uses all foreground representations except the anchor in the
    denominator; positives are defined separately at each hierarchy level.
    TAL normally supplies multiple positive locations per GT. If a level has no
    non-self positive for an anchor, that undefined term contributes zero while
    the outer normalization remains |I|, a defensive transfer rule because the
    source paper generates paired augmented views and therefore assumes positives.

    The denominator is evaluated with logsumexp rather than exp->sum->log. This
    is algebraically equivalent but avoids avoidable overflow in long real runs.
    """
    if embeddings.ndim != 2 or embeddings.shape[1] != config.embedding_dim:
        raise ValueError("HCL embedding dimension tidak valid")
    leaf_labels = leaf_labels.to(device=embeddings.device, dtype=torch.long).reshape(-1)
    if leaf_labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Jumlah embedding dan label HCL berbeda")
    if not len(leaf_labels):
        return embeddings.sum() * 0.0

    _require_finite("hcl_embeddings_input", embeddings)
    z = F.normalize(embeddings.float(), dim=1, eps=1e-8)
    _require_finite("hcl_embeddings_normalized", z)
    hierarchy = prototype_bank.hierarchy
    coarse_labels = hierarchy.coarse_labels(leaf_labels)
    levels = (coarse_labels, leaf_labels)
    weights = prototype_bank.level_weights.to(device=z.device, dtype=z.dtype)
    n = len(z)
    tau = float(config.temperature)
    total = z.new_zeros(())

    for start in range(0, n, int(config.anchor_chunk_size)):
        stop = min(start + int(config.anchor_chunk_size), n)
        anchors = z[start:stop]
        row = torch.arange(stop - start, device=z.device)
        source = torch.arange(start, stop, device=z.device)
        logits = anchors @ z.transpose(0, 1) / tau
        _require_finite("hcl_pair_logits", logits)
        denominator_logits = logits.clone()
        denominator_logits[row, source] = -torch.inf
        log_denominator = torch.logsumexp(denominator_logits, dim=1)
        _require_finite("hcl_log_denominator", log_denominator)

        for level_index, labels in enumerate(levels):
            class_count = int(labels.max()) + 1
            one_hot = F.one_hot(labels, num_classes=class_count).to(dtype=z.dtype)
            counts = one_hot.sum(dim=0)
            category_sum = one_hot.transpose(0, 1) @ z
            anchor_labels = labels[start:stop]
            similarity_sum = (
                anchors @ category_sum.transpose(0, 1)
            )[row, anchor_labels]
            self_similarity = (anchors * z[source]).sum(dim=1)
            positive_count = counts[anchor_labels] - 1.0
            valid = positive_count > 0
            if bool(valid.any()):
                mean_positive_logit = (
                    (similarity_sum[valid] - self_similarity[valid])
                    / positive_count[valid]
                    / tau
                )
                contribution = log_denominator[valid] - mean_positive_logit
                _require_finite(f"hcl_level{level_index}_contribution", contribution)
                total = total + weights[level_index] * contribution.sum()
    output = total / float(n)
    _require_finite("hcl_output", output)
    return output


def balanced_level_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    temperature: float,
    anchor_chunk_size: int,
) -> torch.Tensor:
    """BHCL Eq. (8) averaged over P'_l(i) for one hierarchy level.

    The original denominator
      sum_c (sum_{j in c,j!=i} exp(s_ij/tau) + exp(s_ip_c/tau))/(n_c+1)
    is computed exactly in log-space by assigning each pair/prototype its
    category divisor before logsumexp. This preserves the objective while
    removing the numerically fragile explicit exponentiation chain.
    """
    if embeddings.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("BHCL embeddings/prototypes harus matriks")
    if embeddings.shape[1] != prototypes.shape[1]:
        raise ValueError("Dimensi BHCL embedding/prototype berbeda")
    labels = labels.long().reshape(-1)
    n = embeddings.shape[0]
    class_count = prototypes.shape[0]
    if labels.shape[0] != n:
        raise ValueError("Jumlah embedding dan label BHCL tidak sama")
    if n == 0:
        return embeddings.sum() * 0.0
    if int(labels.min()) < 0 or int(labels.max()) >= class_count:
        raise ValueError("Label hierarchy di luar rentang")

    work_embeddings = embeddings.float()
    work_prototypes = prototypes.float()
    _require_finite("bhcl_level_embeddings", work_embeddings)
    _require_finite("bhcl_level_prototypes", work_prototypes)

    one_hot = F.one_hot(labels, num_classes=class_count).to(dtype=work_embeddings.dtype)
    counts = one_hot.sum(dim=0)
    category_embedding_sum = one_hot.transpose(0, 1) @ work_embeddings
    divisors = counts + 1.0
    log_divisors = torch.log(divisors)
    pair_log_divisors = log_divisors[labels]
    tau = float(temperature)
    total = work_embeddings.new_zeros(())

    for start in range(0, n, int(anchor_chunk_size)):
        stop = min(start + int(anchor_chunk_size), n)
        anchors = work_embeddings[start:stop]
        anchor_labels = labels[start:stop]
        row_indices = torch.arange(stop - start, device=work_embeddings.device)
        source_indices = torch.arange(start, stop, device=work_embeddings.device)

        pair_logits = anchors @ work_embeddings.transpose(0, 1) / tau
        prototype_logits = anchors @ work_prototypes.transpose(0, 1) / tau
        _require_finite("bhcl_pair_logits", pair_logits)
        _require_finite("bhcl_prototype_logits", prototype_logits)

        adjusted_pairs = pair_logits - pair_log_divisors.unsqueeze(0)
        adjusted_pairs[row_indices, source_indices] = -torch.inf
        adjusted_prototypes = prototype_logits - log_divisors.unsqueeze(0)
        log_denominator = torch.logsumexp(
            torch.cat((adjusted_pairs, adjusted_prototypes), dim=1),
            dim=1,
        )
        _require_finite("bhcl_log_denominator", log_denominator)

        similarity_sum_all = anchors @ category_embedding_sum.transpose(0, 1)
        own_similarity_sum = similarity_sum_all[row_indices, anchor_labels]
        self_similarity = (anchors * work_embeddings[source_indices]).sum(dim=1)
        own_prototype_similarity = (anchors * work_prototypes[anchor_labels]).sum(dim=1)
        positive_similarity_sum = own_similarity_sum - self_similarity + own_prototype_similarity
        positive_count = counts[anchor_labels].clamp_min(1.0)
        mean_positive_logit = positive_similarity_sum / positive_count / tau
        contribution = log_denominator - mean_positive_logit
        _require_finite("bhcl_level_contribution", contribution)
        total = total + contribution.sum()

    output = total / float(n)
    _require_finite("bhcl_level_output", output)
    return output


def balanced_hierarchical_contrastive_loss(
    embeddings: torch.Tensor,
    leaf_labels: torch.Tensor,
    prototype_bank,
    config: BHCLConfig,
) -> torch.Tensor:
    """BHCL Eqs. (8)-(10) with positive aggregation of -log pair losses."""
    if embeddings.ndim != 2 or embeddings.shape[1] != config.embedding_dim:
        raise ValueError(
            f"BHCL embeddings harus [N,{config.embedding_dim}], diterima {tuple(embeddings.shape)}"
        )
    leaf_labels = leaf_labels.to(device=embeddings.device, dtype=torch.long).reshape(-1)
    if leaf_labels.shape[0] != embeddings.shape[0]:
        raise ValueError("Jumlah embedding dan leaf label BHCL berbeda")
    if not len(leaf_labels):
        return embeddings.sum() * 0.0

    _require_finite("bhcl_embeddings_input", embeddings)
    z = F.normalize(embeddings.float(), dim=1, eps=1e-8)
    _require_finite("bhcl_embeddings_normalized", z)
    prototype_bank.update(z.detach(), leaf_labels)
    hierarchy = prototype_bank.hierarchy
    coarse_labels = hierarchy.coarse_labels(leaf_labels)
    level_labels = (coarse_labels, leaf_labels)
    weights = prototype_bank.level_weights.to(device=z.device, dtype=z.dtype)

    output = z.new_zeros(())
    for level_index, labels in enumerate(level_labels, start=1):
        prototypes = prototype_bank.normalized(level_index).to(device=z.device, dtype=z.dtype)
        _require_finite(f"bhcl_level{level_index}_prototype_bank", prototypes)
        output = output + weights[level_index - 1] * balanced_level_loss(
            z,
            labels,
            prototypes,
            temperature=config.temperature,
            anchor_chunk_size=config.anchor_chunk_size,
        )
    _require_finite("bhcl_output", output)
    return output


class BHCLDetectionLoss:
    """Native Ultralytics loss plus HCL or BHCL on TAL positives."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundBHCLDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, BHCLDetectHead):
                    raise TypeError("HCL/BHCL loss memerlukan BHCLDetectHead")
                self.bhcl_head = head
                self.bhcl_config = BHCLConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                _require_finite("native_detection_loss_before_bhcl", loss)
                embeddings = preds.get("bhcl_embeddings")
                if embeddings is None:
                    return assignments, loss, loss.detach()

                _require_finite("dense_bhcl_embeddings", embeddings)
                fg_mask, target_gt_idx = assignments[:2]
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                _require_finite("prediction_scores", pred_scores)
                if embeddings.shape[:2] != pred_scores.shape[:2]:
                    raise RuntimeError("HCL/BHCL dense embeddings tidak sejajar dengan predictions")
                if not bool(fg_mask.any()):
                    return assignments, loss, loss.detach()

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
                positive_embeddings = embeddings[fg_mask]
                positive_labels = assigned_labels[fg_mask]
                _require_finite("positive_bhcl_embeddings", positive_embeddings)
                if self.bhcl_config.variant == "hcl":
                    auxiliary = hierarchical_contrastive_loss(
                        positive_embeddings,
                        positive_labels,
                        self.bhcl_head.bhcl_prototypes,
                        self.bhcl_config,
                    )
                else:
                    auxiliary = balanced_hierarchical_contrastive_loss(
                        positive_embeddings,
                        positive_labels,
                        self.bhcl_head.bhcl_prototypes,
                        self.bhcl_config,
                    )
                _require_finite("bhcl_auxiliary_loss", auxiliary)
                loss[1] = loss[1] + float(self.bhcl_config.loss_weight) * auxiliary.to(loss.dtype)
                _require_finite("detection_loss_after_bhcl", loss)
                return assignments, loss, loss.detach()

        return _BoundBHCLDetectionLoss()
