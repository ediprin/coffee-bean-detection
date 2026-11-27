import pytest
import torch

from coffee_detector.auliya_rival_margin import RIVAL_ARM, RIVAL_MARGIN, positive_rival_margin
from coffee_detector.experiments.compare_auliya_rival_margin import paired_objects


def test_hinge_only_gradients_for_true_and_strongest_wrong_at_positives():
    # Image 0 has two GTs, image 1 has none, image 2 has one.
    logits = torch.tensor(
        [[[1., 1., -3.], [4., 9., 5.]],
         [[9., 8., 7.], [0., 0., 0.]],
         [[-1., 2., 1.], [1., -3., 1.]]], requires_grad=True
    )
    foreground = torch.tensor([[True, False], [False, False], [True, False]])
    padded_gt = torch.tensor([[1, 0], [0, 0], [0, 0]])
    batch = {"batch_idx": torch.tensor([0, 0, 2]), "cls": torch.tensor([[2], [1], [2]])}
    loss = positive_rival_margin(logits.transpose(1, 2), foreground, padded_gt, batch)
    assert loss.item() == pytest.approx(RIVAL_MARGIN + .5)
    loss.backward()
    assert torch.equal(logits.grad[~foreground], torch.zeros(4, 3))
    assert logits.grad[0, 0, 1] == -.5 and logits.grad[0, 0, 0] == .5
    assert logits.grad[2, 0, 2] == -.5 and logits.grad[2, 0, 1] == .5


def test_empty_and_safely_separated_positives_get_zero_penalty():
    scores = torch.tensor([[[10.], [5.], [0.]]], requires_grad=True)
    idx = torch.zeros((1, 1), dtype=torch.long)
    fg = torch.ones((1, 1), dtype=torch.bool)
    batch = {"batch_idx": torch.tensor([0]), "cls": torch.tensor([[0]])}
    empty = {key: value[:0] for key, value in batch.items()}
    no_bg = positive_rival_margin(scores, ~fg, idx, empty)
    well_separated = positive_rival_margin(scores, fg, idx, batch)
    assert no_bg == 0 and well_separated == 0
    (no_bg + well_separated).backward()
    assert not scores.grad.count_nonzero()


def test_rival_margin_uses_same_paired_classification_evaluator():
    key = ("image", 0)
    truth = {key: 0}
    pred = {"RGB": {key: 1}, RIVAL_ARM: {key: 0}}
    r = paired_objects(truth, pred, {0: "true", 1: "rival"}, ["image"], arms=("RGB", RIVAL_ARM))
    assert r["rescues"] == 1 and r["regressions"] == 0
