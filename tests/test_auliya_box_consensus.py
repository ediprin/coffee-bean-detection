import numpy as np
import torch

from coffee_detector.analysis.auliya_box_consensus import matched_indices, vote_labels


def test_vote_swaps_duplicate_anchor_class_slots_without_touching_inputs():
    boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.], [20., 20., 30., 30.]])
    probs = torch.tensor([[.40, .10, .01], [.02, .90, .01], [.02, .03, .50]])
    indices = torch.tensor([0, 0, 2])
    classes = torch.tensor([0, 1, 2])
    originals = [t.clone() for t in (boxes, probs, indices, classes)]

    assert vote_labels(boxes, probs, indices, classes).tolist() == [1, 0, 2]
    assert all(torch.equal(a, b) for a, b in zip((boxes, probs, indices, classes), originals, strict=True))


def test_vote_ignores_low_score_neighbors_and_does_not_vote_for_itself():
    boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]])
    probs = torch.tensor([[.0002, .0001], [0., .0009]])
    indexes, labels = torch.tensor([0]), torch.tensor([0])

    assert torch.equal(vote_labels(boxes, probs, indexes, labels), labels)
    assert vote_labels(boxes, probs, indexes[:0], labels[:0]).numel() == 0


def test_matching_skips_low_score_and_prioritizes_valid_pair_count():
    ious = np.array([[.95, .80, .65], [.99, .70, .01]])
    conf = np.array([.10, .6, .6])

    assert matched_indices(ious, conf) == {0: 2, 1: 1}
