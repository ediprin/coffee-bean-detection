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


def test_historical_head_selection():
    from types import SimpleNamespace
    import pytest
    from coffee_detector.analysis.auliya_gc_error_audit import select_validation_head
    class FakeModel:
        def __init__(self):
            self.names = dict(enumerate(str(i) for i in range(21)))
            self.model = [SimpleNamespace(nc=21,cv2=object(),cv3=object(),end2end=True)]
        @property
        def end2end(self):
            return self.model[-1].end2end
        @end2end.setter
        def end2end(self,value):
            self.model[-1].end2end=value
    model=FakeModel()
    names=list(model.names.values())
    select_validation_head(model,{"nms":None,"iou":None},names)
    assert model.end2end is False
    with pytest.raises(ValueError,match="Class order mismatch"):
        select_validation_head(model,{"nms":None},list(reversed(names)))
    with pytest.raises(ValueError,match="Unexpected historical nms"):
        select_validation_head(model,{"nms":False},names)
