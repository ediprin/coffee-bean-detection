import numpy as np
import pytest
from coffee_detector.analysis import auliya_representation_audit as a


def test_letterbox_and_polygon_geometry():
    x=np.zeros((300,400,3),np.uint8)
    im,s,p=a.letterbox(x)
    assert im.shape==(640,640,3)
    np.testing.assert_allclose(s,[1.6,1.6])
    np.testing.assert_allclose(p,[0,80])


def test_masked_crop_excludes_background():
    rgb=np.full((128,128,3),255,np.uint8)
    rgb[32:96,32:96]=[20,40,60]
    p=np.array([[33,33],[94,33],[94,94],[33,94]],np.float32)
    crop,m=a.masked_crop(rgb,p)
    np.testing.assert_allclose(crop[m>.99].mean(0),np.array([20,40,60])/255,atol=1e-4)


def test_visual_dimensions_and_finite():
    rgb=np.full((128,128,3),127,np.uint8)
    p=np.array([[20,20],[100,20],[100,100],[20,100]],np.float32)
    small,s,pad=a.letterbox(rgb)
    v=a.visual(rgb,small,p,p*s+pad)
    assert set(v)==set(a.V_CODES)
    assert v["V0"].shape==v["V1"].shape
    assert len(v["V2"])==48 and len(v["V3"])==53 and len(v["V4"])==14
    assert all(np.isfinite(x).all() for x in v.values())


def test_masked_mean_ignores_outside():
    x=np.array([[[1,2],[3,100]]],np.float32)
    mask=np.array([[1,1],[1,0]],np.uint8)
    np.testing.assert_allclose(a.masked_mean(x,mask),[2])


def test_greedy_raw_matching_unique_and_class_agnostic():
    gt=np.array([[0,0,10,10],[11,0,21,10]],np.float32)
    pred=np.array([[11,0,21,10],[0,0,10,10],[0,0,9,9]],np.float32)
    idx,iou=a.match_raw(gt,pred)
    np.testing.assert_array_equal(idx,[1,0])
    np.testing.assert_allclose(iou,[1,1])


def test_probe_fit_on_training_only():
    rng=np.random.default_rng(42)
    train=rng.normal(size=(100,8))
    y=(train[:,0]>0).astype(int)
    val=rng.normal(size=(10,8))
    p=a.fit_probe(train,val,y,np.zeros(10),4)
    assert p.shape==(10,)


def test_bad_bundle_hash_fails_before_zip_open(tmp_path):
    from argparse import Namespace
    bundle=tmp_path/"invalid.zip"
    bundle.write_bytes(b"not a zip")
    with pytest.raises(ValueError,match="exact original"):
        a.run(Namespace(bundle=bundle,output=tmp_path/"output",stage="visual",checkpoint=None,device="cpu"))
