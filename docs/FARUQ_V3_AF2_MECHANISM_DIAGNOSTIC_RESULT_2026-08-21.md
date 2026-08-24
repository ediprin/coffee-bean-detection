# Faruq-v3 AF2 Mechanism Diagnostic Result

Date: 2026-08-21

Protocol: `docs/FARUQ_V3_AF2_MECHANISM_DIAGNOSTIC_PROTOCOL_2026-08-21.md`

Evaluation: grouped Faruq-v3 validation, paired seeds 42/123/2026

Training executed: **no**

Test accessed: **no**

## Headline result

| Metric | D0FT mean | AF2 mean | Mean delta | Improved seeds |
|---|---:|---:|---:|---:|
| Raw top-500 proposal accessibility | 99.81% | 99.75% | -0.06 | 0/3 |
| Final proposal accessibility | 77.63% | 89.54% | **+11.91** | 3/3 |
| Matched recall | 77.63% | 89.54% | **+11.91** | 3/3 |
| Localization-conditioned Top-1 accuracy | 62.46% | 70.58% | **+8.12** | 3/3 |
| Localized wrong-class rate | 37.54% | 29.42% | **-8.12** | improved 3/3 |
| Proposal-miss rate | 22.37% | 10.46% | **-11.91** | improved 3/3 |
| Correct-decision recall | 48.54% | 63.18% | **+14.64** | 3/3 |

All deltas are percentage points (`AF2 - D0FT`). For error rates, a negative
delta is favorable.

## Paired stability

| Seed | Raw accessibility delta | Final accessibility delta | Conditional Top-1 delta | Correct-decision recall delta |
|---:|---:|---:|---:|---:|
| 42 | +0.00 | +11.79 | +8.58 | +15.02 |
| 123 | -0.19 | +12.74 | +14.98 | +20.53 |
| 2026 | +0.00 | +11.22 | +0.81 | +8.37 |

The final-output and correct-decision improvements occurred in all three seeds.
The conditional classification gain was smallest at seed 2026 but remained
positive.

## Attribution

**CLASSIFICATION_DOMINANT.**

The raw decoded candidate pool was already saturated for both models:
approximately 99.8% of validation targets had a top-500 candidate with IoU at
least 0.50. AF2 did not improve this quantity and therefore did not provide
evidence of better underlying proposal generation or box accessibility.

The large gain after final one-to-one selection has a different interpretation.
Final detections depend on class confidence and ranking. AF2 increased
localization-conditioned Top-1 accuracy by 8.12 points and reduced localized
wrong-class errors by the same amount. Better scoring/discrimination allowed
geometrically available candidates to survive the final output, raising final
accessibility by 11.91 points and correct-decision recall by 14.64 points.

Thus the operational reduction in proposal misses is downstream of improved
classification/ranking, not evidence that AF2 generated more geometrically
valid raw boxes.

## Per-class limits

The mean classification benefit was not uniform. The largest
localization-conditioned gains included `biji_bertutul_tutul` (+50 points),
`kulit_tanduk_ukuran_kecil` (+40), `biji_coklat` (+33), and
`tanah_batu_ranting_kecil` (+15). The largest negative means were
`kulit_tanduk_ukuran_sedang` (-26) and `biji_berlubang_satu` (-16).
Consequently, the result supports a global three-seed classification-dominant
mechanism but not universal per-class improvement.

## Scientific conclusion

The AF2 contribution in this study is best described as a parameter-free
frequency-angular input frontend that improves fine-grained class
discrimination and candidate ranking inside an end-to-end YOLO26 detector.
It should not be claimed to improve raw localization geometry. This is
post-hoc validation association rather than causal proof, and it does not
authorize additional tuning or test access.

Machine-readable evidence:
`docs/evidence/FARUQ_V3_AF2_MECHANISM_DIAGNOSTIC_2026-08-21.json`.
