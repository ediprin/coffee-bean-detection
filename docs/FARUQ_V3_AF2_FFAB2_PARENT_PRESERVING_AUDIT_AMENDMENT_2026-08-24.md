# AF2 + FFAB2 parent-preserving static-audit amendment

Date: 2026-08-24
Status: **audit-implementation correction before any parent-preserving training**

## Trigger

The first real Kaggle/T4 static audit on the completed AF2FS seed-42 parent stopped before training. Every parent-freeze, gradient, descriptor, box-isolation, configuration, and test-lock gate passed. The only failed gate was the original cross-instance full-forward equality threshold of `1e-6`.

Observed maximum absolute differences between the separately instantiated AF2 parent and wrapped zero-initialized residual model were:

| Arm | raw one2one box max abs diff | raw one2one score max abs diff |
|---|---:|---:|
| AF2FFAPR0 | 6.4373016e-06 | 2.0980835e-05 |
| AF2FFAPR1 | 5.7220459e-06 | 2.2888184e-05 |

The failed audit also showed:

- exactly 1,344 trainable parameters out of 2,513,334, all in the three FFAB adapters;
- all parent BatchNorm modules frozen;
- no gradient on parent parameters;
- non-adapter parent state bitwise unchanged after a candidate optimizer step;
- spectral candidate adapter gradient finite and non-zero;
- active FFAB changed class scores while raw box outputs stayed bitwise unchanged;
- zero/spectral descriptor semantics correct;
- all global configuration gates passed;
- test access remained false.

No AF2FFAPR1 training epoch was started before this failure.

## Why the original audit implementation was wrong

The protocol requires the candidate to start from the completed AF2 parent with a zero-output adapter. The first audit tried to prove this solely by comparing full forward outputs from **two separately instantiated CUDA models** with an absolute `1e-6` threshold.

That check conflates the architectural identity invariant with small cross-instance FP32/CUDA numerical differences. The observed `~1e-5` differences occurred even though the parent state itself remained frozen and every structural gate passed. Therefore the `1e-6` cross-instance comparison was an implementation-level false negative, not evidence that the FFAB adapter had already changed the parent.

## Corrected audit invariant

Audit revision `2026-08-24a` keeps the experiment protocol and final metric gates unchanged. It changes only how the pre-training identity condition is proved.

Parent preservation now requires all of the following:

1. **bitwise-identical parent tensor transfer** from the AF2FS source into every non-adapter layer and the wrapped native Detect head;
2. **bitwise adapter identity at initialization**, i.e. for every P3/P4/P5 adapter `F' == F` exactly when `alpha=0`;
3. **bitwise unchanged non-adapter state after a candidate optimizer step**;
4. no parent gradients and frozen parent BatchNorm state;
5. active FFAB must leave raw box outputs bitwise unchanged;
6. the separate full-model CUDA forward comparison is retained only as a coarse fail-closed sanity check with maximum absolute difference `<=1e-3`, not as the identity proof.

The `1e-3` sanity limit is not a performance/selection threshold. Exact preservation is established by items 1--5 above. The sanity check only catches gross wiring or weight-transfer errors.

## What did not change

- parent checkpoint: seed-matched completed AF2FS `best.pt`;
- parent remains frozen;
- only FFAB adapter parameters are trainable;
- 30 continuation epochs per seed;
- seeds 42/123/2026;
- validation-only experiment;
- test remains locked;
- final upgrade gate remains Macro >= +0.50 pp, Bottom-3 >= +0.50 pp, Worst mean non-negative, with the predeclared seed-consistency requirements;
- the prior from-start REJECT and selective-refinement failure are unchanged.

This amendment is recorded before any parent-preserving training result exists and must accompany the final result report.
