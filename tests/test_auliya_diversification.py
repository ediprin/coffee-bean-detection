import numpy as np
from types import SimpleNamespace
from coffee_detector.auliya_diversification import oa_photo, OAPhoto, regions, spectral_saliency, overlap

def sample():
    rng=np.random.RandomState(10)
    return rng.randint(0,256,(96,128,3),dtype=np.uint8)

def test_oa_photo_deterministic_bounded_and_active():
    img=sample()
    boxes=np.array([[5,5,40,50],[60,20,110,80]])
    before=img.copy()
    a=oa_photo(img,boxes,np.random.RandomState(17))
    b=oa_photo(img,boxes,np.random.RandomState(17))
    assert a.dtype==np.uint8 and a.shape==img.shape
    assert np.array_equal(a,b) and not np.array_equal(a,img)
    assert np.array_equal(before,img)

def test_regions_nonoverlapping_and_in_bounds():
    boxes=regions(96,128,(.01,.1),5,np.random.RandomState(8))
    assert boxes
    assert all(0<=x<x2<=128 and 0<=y<y2<=96 for x,y,x2,y2 in boxes)
    assert not any(overlap(a,b) for i,a in enumerate(boxes) for b in boxes[i+1:])

def test_geometry_unchanged_and_zero_prob_identity():
    img=sample()
    segments=np.array([[[.1,.1],[.6,.1],[.6,.7],[.1,.7]]],np.float32)
    inst=SimpleNamespace(segments=segments.copy(),normalized=True)
    labels={"img":img.copy(),"instances":inst,"cls":np.array([[1]])}
    assert OAPhoto(0)(labels) is labels
    assert np.array_equal(labels["img"],img)
    out=OAPhoto(1)(labels)
    assert out["instances"] is inst and np.array_equal(inst.segments,segments)
    assert out["cls"].tolist()==[[1]] and out["img"].shape==img.shape

def test_small_and_empty_boxes():
    img=sample()
    assert spectral_saliency(img[:1,:1])==-1
    assert oa_photo(img,np.empty((0,4)),np.random.RandomState(4)).shape==img.shape
