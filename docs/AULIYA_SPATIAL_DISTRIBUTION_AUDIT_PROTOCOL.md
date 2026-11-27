# Auliya Spatial Distribution Audit V1

Registered before R2–R4 results. This is an exploratory development diagnostic,
not a new detector or a held-out confirmation. Previous C results informed the
design. No test A, no detector training, no ISDE implementation.

## Frozen inputs and controls

Reuse the extracted Kaggle development folder, exact RGB checkpoint, and official
Zenodo metadata from the completed F0–F6 study. Expected content SHA:
12d60838ae810a2841607fd5a09afe6b77c8a48977ea2b9c6bc3a52394d55166.
Exactly 1,241 train and 1,265 validation instances, ordered identity/label digests
from the uploaded feature cache are enforced. P2 and P3 each have 64 channels.
The same frozen model, letterbox, GT polygon mask, feature hooks, StandardScaler,
PCA rank 32 and balanced logistic regression as the earlier audit are used.
Only B fits bin edges, scaler, PCA and classifier; C is used for evaluation.

## Factorial conditions at each level

| Code | Pooling | Raw dimensionality |
|---|---|---:|
| R1 | Global weighted mean (original masked_mean) | 64 |
| R2 | Global weighted histogram | 256 |
| R3 | Four object-bounding-box-relative regional means | 256 |
| R4 | Four regional weighted histograms | 1024 |

Histogram boundaries are training-only per-channel empirical quartiles from
64 deterministically sampled mask-weighted locations per B instance. This
is an equal-instance calibration, not an exact pixel-weighted population
quantile. Four bins include both tails. Duplicate boundaries are allowed
and recorded. All bins/regions share the same edges. Regional masks partition
the image-coordinate GT mask before area downsampling. Empty regions are zero.
Histogram pooling is orderless within each region, and fixed quadrants are
rotation-sensitive. No claim to count holes or recover topology explicitly.
P2/P3 are preselected, no layer or bin-count sweep.

Primary: PCA32 matched-dimensional probe, with raw-dimensional results as
sensitivity only. Matching the final classifier dimension does not equalize
the information retained by PCA or input descriptor complexity.

## Baseline equivalence

Before fitting R2–R4, R1 must reproduce the previous F1/F2 Macro18, Hard5,
Worst18 within 0.005 absolute for both matched and raw. Fail closed if not.
This is metric equivalence, not a guarantee of identical predictions.
Expected P2 matched: .4087485968, .2328996864, .0740740741.
Expected P3 matched: .3373170799, .1732999502, .1000000000.

## Decision, fixed before screening

Compare each matched R2/R3/R4 to its own level's R1. Pass only if Macro18
and Hard5 each improve >= .02, Worst18 falls by no more than .01, and at
least 3 of 5 fixed hard classes improve F1. This final breadth condition is
an explicit refinement of the preceding screening proposal. No automatic
promotion. P2 primary, P3 depth sensitivity. Raw results cannot override
a matched failure. Report main factorial contrasts and interaction, per-class
F1, confusion, paired rescue/damage in clean18 and hard5. No iid significance
claim or independent confirmation.

If no P2 condition passes, stop this histogram/quadrant direction unless
a new independent rationale is established. P3-only success is a separate
diagnostic finding, not automatic permission to develop a P2 head.
Even a pass is GT-mask oracle evidence; predicted-instance transfer must be
tested before detector modification.

## Kaggle

Import Auliya_Spatial_Distribution_Audit_Kaggle.ipynb. Attach the same
auliya-robusta-sni dataset and trained RGB checkpoint dataset. GPU + Internet.
No prior notebook output is required. Run All, then Save Version.
Output /kaggle/working/auliya-spatial-distribution-v1/spatial contains
report.json, validation_predictions.csv, calibration, contract and caches.
The output root is distinct from visual/features. Reuse only an exact-contract
previous spatial output. No execution on actual data has been claimed by this
commit. Full synthetic tests run in the notebook before extraction.
