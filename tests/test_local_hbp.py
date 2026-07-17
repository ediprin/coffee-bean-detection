import torch
from torch import nn

from coffee_detector.models.local_hbp import (
    LocalBilinearAdapter,
    LocalHBPClassBranch,
    inject_local_hbp,
)


class FakeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cv2 = nn.ModuleList([nn.Sequential(nn.Conv2d(16, 4, 1))])
        self.cv3 = nn.ModuleList([nn.Sequential(nn.Conv2d(16, 6, 1))])


class FakeDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList([nn.Identity(), FakeHead()])


def test_adapter_is_identity_at_initialization() -> None:
    adapter = LocalBilinearAdapter(16, rank=8)
    features = torch.randn(2, 16, 11, 13)

    output = adapter(features)

    assert output.shape == features.shape
    assert torch.equal(output, features)


def test_injection_changes_only_classification_branch() -> None:
    detector = FakeDetector()
    box_branch = detector.model[-1].cv2[0]
    features = torch.randn(2, 16, 9, 9)
    baseline_class = detector.model[-1].cv3[0](features)

    patched = inject_local_hbp(detector, rank=8)

    assert patched == 1
    assert detector.model[-1].cv2[0] is box_branch
    assert isinstance(detector.model[-1].cv3[0], LocalHBPClassBranch)
    candidate_class = detector.model[-1].cv3[0](features)
    assert torch.equal(candidate_class, baseline_class)
    assert candidate_class.shape == (2, 6, 9, 9)


def test_adapter_parameters_receive_gradients() -> None:
    adapter = LocalBilinearAdapter(8, rank=4)
    with torch.no_grad():
        adapter.output.weight.normal_(0, 0.01)
    features = torch.randn(2, 8, 5, 5, requires_grad=True)

    adapter(features).square().mean().backward()

    assert adapter.left.weight.grad is not None
    assert adapter.right.weight.grad is not None
    assert adapter.output.weight.grad is not None

