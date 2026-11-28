import torch

from coffee_detector.auliya_shape_coeff import ShapeProjection, shape_regression, targets_from_masks


def test_overlap_teacher_follows_instance_ids_and_skips_other_classes():
    masks = torch.zeros((1, 30, 30), dtype=torch.uint8)
    masks[0, 0:10, 0:10] = 1
    masks[0, 12:22, 12:23] = 2
    masks[0, 15:20, 18:23] = 0
    masks[0, 23:28, 23:28] = 3
    b = {'masks': masks, 'cls': torch.tensor([[7.],[13.],[0.]]),
         'batch_idx': torch.tensor([0,0,0]), 'img': torch.empty((1,3,30,30))}
    target, valid = targets_from_masks(b, overlap=True)
    assert valid.tolist() == [True, True, False]
    assert target[0] > target[1]


def test_zero_initialized_residual_and_detached_mask_gradient():
    coef = torch.randn(1, 32, 6, requires_grad=True)
    p = ShapeProjection(32)
    s, delta = p(coef)
    assert torch.count_nonzero(delta) == 0
    s.sum().backward()
    assert coef.grad is None and p.projection.weight.grad is not None


def test_auxiliary_gradient_only_on_assigned_valid_masks_in_native_order():
    shape = torch.zeros((2, 3), requires_grad=True)
    fg = torch.tensor([[True,True,False],[False,True,False]])
    indices = torch.tensor([[1,0,0],[0,0,0]])
    b = {'batch_idx': torch.tensor([0,0,1]), 'shape_target': torch.tensor([.5,.8,-.7]),
         'shape_valid': torch.tensor([False,True,True])}
    shape_regression(shape,fg,indices,b).backward()
    assert (shape.grad != 0).tolist() == [[True,False,False],[False,True,False]]
    assert shape.grad[0,0] < 0 < shape.grad[1,1]
