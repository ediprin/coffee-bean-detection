import numpy as np
from coffee_detector.analysis.auliya_gc_error_audit import assign,ranks,summarize


def test_unique_matching_prioritizes_cardinality():
    x=np.array([[.95,.7],[.8,.1]])
    np.testing.assert_array_equal(assign(x,.5),[1,0])


def test_no_candidate_or_low_iou():
    np.testing.assert_array_equal(assign(np.empty((2,0)),.5),[-1,-1])
    np.testing.assert_array_equal(assign(np.array([[.2],[.8]]),.5),[-1,0])


def test_rank_margin_and_tie_break():
    assert ranks(np.array([1.,3.,2.]),0)==(3,-2.)
    assert ranks(np.array([3.,3.,2.]),1)==(2,0.)
