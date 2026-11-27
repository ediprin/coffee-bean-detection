import numpy as np
from coffee_detector.experiments import run_auliya_controlled_generalization as g


def test_selection_deterministic_and_balanced():
    counts={}
    for c in range(21):
        for j in range(10):
            v=np.zeros(21,dtype=int)
            v[c]=1
            counts[f"{c}-{j}"]=v
    paths=list(counts)
    a=g.select(paths,counts,42,"fixture")
    b=g.select(list(reversed(paths)),counts,42,"fixture")
    assert a==b and len(set(a))==42
    np.testing.assert_array_equal(sum(counts[p] for p in a),np.full(21,2))


def test_partition_is_disjoint_and_group_specific():
    manifest,counts={},{}
    for group,n in (("A",100),("B",375),("C",420)):
        for i in range(n):
            path=f"{group}/{i}"
            manifest[path]={"group":group}
            counts[path]=np.eye(21,dtype=int)[i%21]
    p=g.plan(manifest,counts)
    assert {k:len(v) for k,v in p.items()}=={"B_train":300,"B_eval":75,"C_train":300,"C_eval":75}
    assert len(set().union(*map(set,p.values())))==750
    assert all(x.startswith(k[0]+"/") for k,v in p.items() for x in v)


def test_fixed_endpoint_schedule():
    assert g.SCHEDULE["epochs"]==80 and g.SCHEDULE["patience"]==0
    assert g.SCHEDULE["batch"]==16 and g.SCHEDULE["seed"]==0


def test_split_digest_is_order_sensitive():
    assert g.split_digest(["a","b"])!=g.split_digest(["b","a"])
