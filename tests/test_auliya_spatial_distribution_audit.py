import numpy as np
from coffee_detector.analysis import auliya_spatial_distribution_audit as s
from coffee_detector.analysis import auliya_representation_audit as a


def test_regional_masks_partition_support():
    mask=np.ones((11,13),np.uint8)
    p=np.array([[1.2,2.3],[8.9,2.3],[8.9,9.1],[1.2,9.1]])
    np.testing.assert_array_equal(sum(s.regional_masks(mask,p)),mask)


def test_weighted_hist_known_distribution_and_empty():
    f=np.array([[[0.,1.],[2.,3.]]],np.float32)
    w=np.array([[1.,2.],[3.,4.]],np.float32)
    e=np.array([[.5,1.5,2.5]])
    np.testing.assert_allclose(s.weighted_hist(f,w,e),[.1,.2,.3,.4])
    np.testing.assert_array_equal(s.weighted_hist(f,np.zeros_like(w),e),np.zeros(4))


def test_histogram_does_not_change_with_rearrangement_but_regional_does():
    f=np.array([[[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]],np.float32)
    g=f[:,:,::-1].copy()
    m=np.ones((4,4),np.uint8)
    p=np.array([[0,0],[4,0],[4,4],[0,4]],np.float32)
    e=np.array([[.1,.4,.9]],np.float32)
    x,y=s.operators(f,m,p,e),s.operators(g,m,p,e)
    np.testing.assert_array_equal(x[0],y[0])
    np.testing.assert_array_equal(x[1],y[1])
    assert not np.array_equal(x[2],y[2])
    assert not np.array_equal(x[3],y[3])


def test_baseline_exact_and_dimensions():
    rng=np.random.default_rng(42)
    f=rng.normal(size=(64,8,8)).astype(np.float32)
    m=np.ones((32,32),np.uint8)
    p=np.array([[0,0],[32,0],[32,32],[0,32]],np.float32)
    e=np.tile([-.5,0,.5],(64,1))
    r=s.operators(f,m,p,e)
    np.testing.assert_array_equal(r[0],a.masked_mean(f,m))
    assert [len(v) for v in r]==[64,256,256,1024]
    assert all(np.isfinite(v).all() for v in r)


def test_calibration_uses_fixed_sample_count_and_quantiles():
    f=np.arange(16,dtype=np.float32).reshape(1,4,4)
    w=np.zeros((4,4),np.float32)
    w[1:3,1:3]=1
    sample=s.sample_cells(f,w)
    assert sample.shape==(64,1)
    assert set(sample.ravel())=={5.,6.,9.,10.}
    edges=s.calibrate(sample)
    np.testing.assert_allclose(edges[:,1],7.5)
