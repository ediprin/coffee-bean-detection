# Faruq-v3 AF2 and IGEM1 Paired Confirmation Result

Date: 2026-08-15

Decision: **AF2 PASS; IGEM1 PASS — validation robustness confirmed against
the seed-matched D0FT control. Test was not accessed.**

## Protocol

The frozen protocol compared each candidate independently with the existing
optimization-matched `D0FT` control on seeds 42, 123, and 2026. Seed 42 was
reused from breadth screening; only candidate seeds 123 and 2026 were newly
trained. All runs used the Faruq-v3 grouped development split and the frozen
50-epoch schedule.

- `AF2`: frequency-domain chaotic-amplitude preprocessing.
- `IGEM1`: classification-guided static/dynamic neighborhood features with
  auxiliary mask supervision.

The protocol did not define a confirmatory AF2-versus-IGEM1 superiority test.

## Per-seed results

| Seed | Model | Macro mAP50-95 | Bottom-3 | Worst class |
|---:|---|---:|---:|---:|
| 42 | D0FT | 86.69% | 74.98% | 72.02% |
| 42 | AF2 | 88.20% | 80.04% | 79.35% |
| 42 | IGEM1 | 88.01% | 82.18% | 82.08% |
| 123 | D0FT | 86.35% | 74.75% | 67.83% |
| 123 | AF2 | 88.22% | 78.22% | 76.46% |
| 123 | IGEM1 | 88.32% | 76.69% | 73.03% |
| 2026 | D0FT | 86.81% | 79.99% | 79.31% |
| 2026 | AF2 | 87.40% | 79.85% | 78.65% |
| 2026 | IGEM1 | 86.82% | 78.93% | 78.11% |

## Three-seed aggregate

| Model | Macro mean ± SD | Bottom-3 mean ± SD | Worst mean ± SD | Decision |
|---|---:|---:|---:|---|
| D0FT | 86.62 ± 0.24% | 76.58 ± 2.96% | 73.05 ± 5.81% | control |
| **AF2** | **87.94 ± 0.47%** | **79.37 ± 1.00%** | **78.15 ± 1.51%** | **PASS** |
| IGEM1 | 87.71 ± 0.79% | 79.27 ± 2.76% | 77.74 ± 4.53% | **PASS** |

### AF2 minus D0FT

- Macro: **+1.32 points**, improved in 3/3 seeds; minimum paired delta
  +0.59 point.
- Bottom-3: **+2.80 points**, improved in 2/3 seeds; minimum paired delta
  -0.15 point.
- Worst class: **+5.10 points**, improved in 2/3 seeds; minimum paired delta
  -0.66 point.

### IGEM1 minus D0FT

- Macro: **+1.10 points**, improved in 3/3 seeds; the seed-2026 delta was
  effectively zero (+0.006 point).
- Bottom-3: **+2.69 points**, improved in 2/3 seeds; minimum paired delta
  -1.06 points.
- Worst class: **+4.69 points**, improved in 2/3 seeds; minimum paired delta
  -1.20 points.

Both candidates satisfy every independently frozen acceptance criterion.

## Interpretation

The frequency-domain AF2 mechanism and the IGEM neighborhood/mask mechanism
both retain positive mean Macro and lower-tail gains across three validation
seeds relative to matched continued training. This is stronger evidence than
their original seed-42 breadth results.

AF2 has descriptively higher aggregate means than IGEM1 by 0.22 Macro, 0.10
Bottom-3, and 0.41 Worst-class point, and substantially lower across-seed
dispersion on the two lower-tail metrics. AF2 is therefore the practical lead
candidate among these two. This ranking is descriptive: the frozen protocol
did not authorize a direct AF2-versus-IGEM1 superiority claim.

Relative to the previously confirmed validation candidates, AF2 has a slightly
higher Macro mean than STB1 (87.94% versus 87.82%), while STB1 retains slightly
higher Bottom-3 and Worst means (80.50% and 78.36%). STB1 nevertheless failed
its separate spatial-causal gate against CMC0. These rows answer different
mechanism questions and must not be treated as interchangeable proof.

## Claim boundary and next action

- This result confirms **validation robustness**, not external-domain or
  locked-test superiority.
- The Faruq locked test had already been consumed by the ACMC study and was not
  reopened here.
- No post-result tuning, additional test evaluation, or AF2/IGEM fusion is
  authorized by this protocol.
- Preserve AF2 as the lead standalone candidate and IGEM1 as an independently
  validated secondary candidate when writing the model-ablation narrative.

Authoritative raw report:
`experiments/faruq-v3-af2-igem-paired-confirmation-v1/val_reports/af2_igem_paired_confirmation.json`.

Repository evidence snapshot:
`docs/evidence/FARUQ_V3_AF2_IGEM_PAIRED_CONFIRMATION_2026-08-15.json`.
