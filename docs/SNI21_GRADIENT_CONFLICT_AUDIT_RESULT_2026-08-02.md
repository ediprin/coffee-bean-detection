# SNI-21 Gradient-Conflict Audit Result

Date: 2026-08-02

Protocol: `sni21-gradient-conflict-audit-v1`

Checkpoint: completed Faruq-v3 `D0_seed42/weights/best.pt`

Evidence: 24 deterministic train batches, 8 images per batch, 192 images total

Training executed: no

Validation accessed: no

Test accessed: no

## Decision

**FAIL. Stop the conflict-aware gradient-projection direction.**

Neither the feature extractor nor the Detect classification towers met the
frozen conflict gate. No batch had a negative leaf-versus-ontology gradient
cosine, so PCGrad, gradient projection, and the proposed conflict-aware
dual-head are not authorized.

| Parameter group | Mean cosine | Median cosine | Q25--Q75 | Negative batches | Negative fraction | Median leaf norm | Median ontology norm | Shared parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All shared | 0.630 | 0.618 | 0.499--0.782 | 0/24 | 0.00% | 17.139 | 20.187 | 2,368,350 |
| Feature extractor | 0.659 | 0.673 | 0.505--0.810 | 0/24 | 0.00% | 16.123 | 18.142 | 2,262,624 |
| Classification head | 0.473 | 0.474 | 0.396--0.583 | 0/24 | 0.00% | 5.389 | 5.900 | 105,726 |

The frozen gate required a negative median cosine and at least 50% negative
batches. The observed negative fraction is zero for every parameter group.

## Corrected mechanism interpretation

The earlier hypothesis was that S0 improved proposal accessibility while
damaging leaf classification because the ontology and flat-class gradients
directly opposed one another. This audit rejects that explanation at the D0
checkpoint state: the gradients are consistently and substantially aligned.

The result instead supports a more limited interpretation. The ontology loss
is less specific than the 21-class objective. It can reward probability mass
inside a semantic group without requiring the correct leaf to dominate. Its
gradient norm is also comparable to or larger than the leaf gradient norm.
Thus an aligned auxiliary update can still alter the subsequent optimization
trajectory and assignments while failing to preserve the final leaf boundary.
This is an under-constraint or trajectory hypothesis, not a demonstrated
gradient-conflict mechanism.

This audit measures gradients at the common D0 endpoint rather than throughout
the S0 training trajectory. It therefore does not prove that conflict can never
occur later. It does prove that the pre-registered evidence required to justify
building gradient surgery is absent.

## Research consequence

- Do not implement PCGrad, projected auxiliary gradients, or the proposed
  conflict-aware dual-head from this evidence.
- Do not tune the conflict threshold, sample extra batches after observing the
  result, train another seed, or access test.
- Preserve D0 as the valid detector baseline.
- Treat S0's failure as a specificity problem: coarse semantic supervision is
  compatible with the local leaf gradient but does not uniquely constrain the
  desired fine-grained decision.
- Any later algorithmic candidate must directly strengthen observable
  leaf-class boundaries and must not be justified by gradient conflict.

Raw report:

```text
Coffee_Bean_Detection/experiments/faruq-v3-gradient-conflict-audit-v1/
gradient_conflict_audit.json
```

