import numpy as np
from coffee_detector.analysis.auliya_gc_mechanism_audit import score_assign,view_transform,metrics

def test_score_assign_max_coverage_before_score():
    iou=np.array([[.8,.9,0],[.8,0,0]])
    assert score_assign(iou,np.array([.99,.1,.8]),.75).tolist()==[1,0]

def test_score_assign_prefers_confidence_not_class():
    assert score_assign(np.array([[.99,.8]]),np.array([.1,.9]),.75).tolist()==[1]
    assert score_assign(np.array([[.4,.7]]),np.array([.1,.9]),.75).tolist()==[-1]

def test_flip_preserves_object_identity_and_geometry():
    image=np.zeros((640,640,3),np.uint8)
    image[:,0]=255
    gt=np.array([[10,20,30,40]],np.float32)
    x,b=view_transform(image,gt,(0,0,640,640),"flip")
    assert b.tolist()==[[610,20,630,40]]
    assert (x[:,-1]==1).all()

def test_photometric_views_leave_padding_unchanged():
    image=np.full((640,640,3),128,np.uint8)
    gt=np.array([[10,20,30,40]],np.float32)
    for view in ("dark","light"):
        x,b=view_transform(image,gt,(10,20,40,50),view)
        assert np.array_equal(b,gt)
        assert np.isclose(x[0,0,0],128/255)
        assert np.isclose(x[20,10,0],.9*128/255+(.1 if view=="light" else 0))
        assert x.min()>=0 and x.max()<=1
