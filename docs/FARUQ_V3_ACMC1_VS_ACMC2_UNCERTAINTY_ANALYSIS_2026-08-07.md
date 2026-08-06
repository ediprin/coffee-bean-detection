# Faruq-v3 ACMC1 vs ACMC2 Uncertainty Analysis

Date: 2026-08-07

Status: **post-confirmation exploratory analysis**

This document does not replace or modify the frozen paired-confirmation decision. The predeclared selection rule remains authoritative: ACMC2 failed the criterion `macro_mean_not_lower_than_acmc1`, so ACMC1 remains the selected model. This analysis only asks how large and stable the observed ACMC2-vs-ACMC1 differences are across the three paired seeds already available.

No new training was run. No test split was opened or accessed.

## Paired seed differences

Differences are ACMC2 minus ACMC1 in percentage points.

| Seed | Macro | Bottom-3 | Worst-class |
|---:|---:|---:|---:|
| 42 | +0.185 pp | +1.537 pp | -0.156 pp |
| 123 | -0.567 pp | +0.064 pp | +2.471 pp |
| 2026 | +0.206 pp | -0.871 pp | +0.474 pp |

## Descriptive uncertainty across seeds

| Metric | Mean delta | SD across seed deltas | 95% paired t CI | Paired t p-value | Exact sign-flip p-value |
|---|---:|---:|---:|---:|---:|
| Macro mAP50-95 | -0.058 pp | 0.440 pp | [-1.152, +1.036] pp | 0.840 | 1.000 |
| Bottom-3 mAP50-95 | +0.243 pp | 1.214 pp | [-2.773, +3.259] pp | 0.762 | 0.750 |
| Worst-class mAP50-95 | +0.930 pp | 1.372 pp | [-2.477, +4.337] pp | 0.361 | 0.500 |

The paired t calculations use only three paired seeds and therefore have two degrees of freedom. The exact sign-flip permutation has only eight possible sign assignments. These results are consequently very low-power and should be treated as descriptive/exploratory rather than definitive inference.

## Macro interpretation

The observed Macro difference is extremely small relative to seed-to-seed variability:

- ACMC1 mean Macro: 87.617%
- ACMC2 mean Macro: 87.559%
- difference: -0.058 pp
- SD of paired Macro differences: 0.440 pp

The exploratory paired test provides no evidence that ACMC1 and ACMC2 differ in Macro (`p=0.840`). However, **absence of evidence is not evidence of equivalence**. With only three seeds and no predeclared smallest effect size of interest (SESOI), the current data cannot support a formal claim that the models are statistically equivalent.

For reference, the 90% CI for the paired Macro difference is approximately [-0.801, +0.684] pp. Under a TOST-style equivalence framework:

- a predeclared equivalence margin of +/-0.5 pp would **not** be established by these three seeds because the 90% CI extends beyond that interval;
- a margin of +/-1.0 pp would contain the 90% CI, but selecting that margin after seeing the result would be post hoc and should not be used to relabel the frozen confirmation as a success.

Therefore the defensible statement is:

> ACMC2 did not satisfy the frozen promotion criterion, but the observed Macro deficit versus ACMC1 is very small and the three-seed data do not provide evidence of a substantive Macro difference.

## Tail interpretation

ACMC2 shifts the observed mean toward difficult classes:

- Bottom-3 mean delta: +0.243 pp
- Worst-class mean delta: +0.930 pp

The direction is positive on two of three seeds for both tail metrics. Nevertheless, the uncertainty intervals are wide and include zero, so these three seeds do not establish a statistically reliable tail advantage either.

The strongest bounded interpretation is therefore not that ACMC2 is globally better, but that margin conditioning appears to change the trade-off:

- Macro is practically very close to ACMC1 on average;
- Bottom-3 is slightly higher on average;
- Worst-class is approximately 0.93 pp higher on average;
- the magnitude and consistency of the tail advantage remain uncertain with n=3 seeds.

## Scientific conclusion

Keep the two conclusions separate:

1. **Protocol/selection conclusion:** ACMC1 remains selected because ACMC2 failed the predeclared `macro_mean_not_lower_than_acmc1` criterion.
2. **Scientific interpretation:** ACMC2 should not be described as simply inferior. Its Macro performance is nearly identical on average, while the observed trade-off shifts toward tail performance.

Recommended thesis wording:

> Adding top1-top2 margin uncertainty to the entropy-conditioned ACMC gate did not satisfy the preregistered promotion rule because mean Macro mAP50-95 decreased by 0.06 percentage points across three paired seeds. Exploratory paired analysis showed that this difference was small relative to between-seed variation and did not provide evidence of a Macro performance difference. At the same time, ACMC2 increased Bottom-3 and Worst-class mean mAP50-95 by 0.24 and 0.93 percentage points, respectively, suggesting a tail-oriented trade-off rather than a uniform improvement. These uncertainty estimates are based on only three seeds and are not sufficient to establish formal equivalence or a statistically reliable tail advantage.

## Boundary

- Do not alter the frozen ACMC2 confirmation decision after observing this exploratory analysis.
- Do not open the locked test split for ACMC2.
- Do not retune ACMC2 solely to erase the observed -0.06 pp Macro mean difference.
- If future work wants to make a formal equivalence/non-inferiority claim, define the SESOI/equivalence margin independently and prospectively before collecting additional seeds.

Primary result record:

`docs/FARUQ_V3_ACMC2_PAIRED_RESULT_2026-08-07.md`

Machine-readable result snapshot:

`docs/evidence/FARUQ_V3_ACMC2_PAIRED_RESULT_2026-08-07.json`
