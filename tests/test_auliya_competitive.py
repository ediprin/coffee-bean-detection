import pytest
import torch
import torch.nn.functional as F

from coffee_detector.auliya_competitive import (
    competitive_loss, positive_logits, MARGIN,
)


def test_cross_anchor_pair_is_the_only_difference():
    # Same strongest wrong-class logit (anchor 1, class 1).
    # True evidence: M1 uses 1 at anchor 1; M2 uses 5 at anchor 0.
    z=torch.tensor([[5.,2.,0.],[1.,4.,0.]],requires_grad=True)
    labels=torch.tensor([0,0])
    group=torch.tensor([10,10])
    m1=competitive_loss(z,labels,group,"M1")
    m2=competitive_loss(z,labels,group,"M2")
    assert torch.allclose(m1,F.softplus(torch.tensor(MARGIN+4-1)))
    assert torch.allclose(m2,F.softplus(torch.tensor(MARGIN+4-5)))
    g1=torch.autograd.grad(m1,z,retain_graph=True)[0]
    g2=torch.autograd.grad(m2,z)[0]
    assert g1[1,1]>0 and g1[1,0]<0 and g1[0,0]==0
    assert g2[1,1]>0 and g2[0,0]<0 and g2[1,0]==0


def test_object_balanced_reduction_and_singleton_equivalence():
    z=torch.tensor([[3.,2.],[3.,2.],[0.,2.]],requires_grad=True)
    y=torch.tensor([0,0,0])
    g=torch.tensor([1,1,7])
    expected=(F.softplus(torch.tensor(0.))+F.softplus(torch.tensor(3.)))/2
    for arm in ("M1","M2"):
        assert torch.allclose(competitive_loss(z,y,g,arm),expected)


def test_empty_positive_set_has_finite_zero_and_backward():
    z=torch.empty(0,21,requires_grad=True)
    for arm in ("M1","M2"):
        value=competitive_loss(z,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),arm)
        assert value.item()==0
        value.backward()
        assert z.grad is not None


def test_positive_mapping_respects_image_boundary():
    z=torch.arange(2*3*4,dtype=torch.float32).reshape(2,3,4)
    preds={"scores":z}
    batch={"batch_idx":torch.tensor([0,0,1]),"cls":torch.tensor([[2],[1],[0]])}
    fg=torch.tensor([[True,True,False,False],[True,False,False,False]])
    idx=torch.tensor([[1,0,0,0],[0,0,0,0]])
    zz,y,g=positive_logits(preds,batch,fg,idx)
    assert y.tolist()==[1,2,0] and g.tolist()==[1,0,2]
    assert torch.equal(zz,z.permute(0,2,1)[fg])
    idx[1,0]=1
    with pytest.raises(ValueError,match="out of bounds"):
        positive_logits(preds,batch,fg,idx)


def test_deterministic_tie_break():
    z=torch.tensor([[3.,2.],[3.,2.]],requires_grad=True)
    val=competitive_loss(z,torch.tensor([0,0]),torch.tensor([0,0]),"M2")
    grad=torch.autograd.grad(val,z)[0]
    assert grad[0,0]<0 and grad[0,1]>0 and torch.equal(grad[1],torch.zeros(2))
