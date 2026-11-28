"""S1/S2 native Segment26 contract tests, no dataset required."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from ultralytics.nn.modules.head import Segment26
from coffee_detector.auliya_semantic_transfer import install, ARMS


def head():
    torch.manual_seed(11)
    h = Segment26(nc=21,nm=8,npr=32,reg_max=1,end2end=True,ch=(16,32,64))
    h.stride = torch.tensor([8.,16.,32.])
    return h


def inputs():
    g=torch.Generator().manual_seed(12)
    return [torch.rand(2,c,n,n,generator=g) for c,n in [(16,16),(32,8),(64,4)]]


def variant(h,arm):
    m=SimpleNamespace(model=[deepcopy(h)])
    install(m,arm)
    return m.model[-1]


def test_initial_outputs_native_train_eval():
    native=head()
    candidates=[variant(native,a) for a in ARMS]
    x=inputs()
    for training in (True,False):
        native.train(training)
        result=native(x)
        for h in candidates:
            h.train(training)
            r=h(x)
            orig=result if training else result[1]
            raw=r if training else r[1]
            for branch in ("one2many","one2one"):
                for k in ("boxes","scores","mask_coefficient"):
                    assert torch.equal(orig[branch][k],raw[branch][k]), (branch,k)
                p,q=orig[branch]["proto"],raw[branch]["proto"]
                if isinstance(p,tuple):
                    assert all(torch.equal(a,b) for a,b in zip(p,q))
                else:
                    assert torch.equal(p,q)


def test_gradient_routes_and_only_p3():
    for arm in ARMS:
        h=variant(head(),arm).train()
        with torch.no_grad():
            h.o2m_residual.weight.fill_(.01)
            h.o2o_residual.weight.fill_(.01)
        x=[t.requires_grad_() for t in inputs()]
        r=h(x)
        trunk=h.independent_reference if arm=="S1" else h.proto.semseg[:2]
        p=next(trunk.parameters())
        sem=r["one2many"]["proto"][1]
        g=torch.autograd.grad(sem.square().mean(),p,allow_unused=True,retain_graph=True)[0]
        assert (g is None) if arm=="S1" else (g is not None and bool(g.abs().sum()>0))
        g=torch.autograd.grad(r["one2many"]["scores"].sum(),p,retain_graph=True)[0]
        assert g.abs().sum()>0
        grad=torch.autograd.grad(r["one2one"]["scores"].sum(),(p,*x),allow_unused=True,retain_graph=True)
        assert all(g is None for g in grad)
        with torch.no_grad():
            baseline=h.forward_head(x,**h.one2many)
        n=x[0].shape[-2]*x[0].shape[-1]
        assert torch.equal(r["one2many"]["scores"][...,n:],baseline["scores"][...,n:])
        assert not torch.equal(r["one2many"]["scores"][...,:n],baseline["scores"][...,:n])


def test_fuse_both_inference_modes_and_equal_parameters():
    for mode in (True,False):
        counts=[]
        for arm in ARMS:
            h=variant(head(),arm).eval()
            h.end2end=mode
            with torch.no_grad():
                h.o2m_residual.weight.fill_(.01)
                h.o2o_residual.weight.fill_(.01)
                x=inputs()
                raw=h(x)[1]
                selected=raw["one2one" if mode else "one2many"]
                h.fuse()
                result=h(x)[1]
                selected2=result["one2one"] if mode else result
                for k in ("boxes","scores","mask_coefficient","proto"):
                    assert torch.equal(selected[k],selected2[k]), (mode,arm,k)
            counts.append(sum(p.numel() for p in h.parameters()))
        assert counts[0]==counts[1],counts
