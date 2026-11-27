# E3 mechanism audit — verified result (2026-11-27)

Source: auliya-gc-e3-report.zip SHA256 efafc3afcf9b60ec08c5be6e549b41b7adff96bd968c2610e053efbff174e571.
Code pin 8fe1d733db30bde0389e5ce358171631518e7096; runner SHA b62da6ed2ee48cca4b39cd9c293393f03b5fe4e41a7fa952aaa3ecc3bd542147.
28 tests passed, one SciPy deprecation warning. T4, Torch 2.11.0+cu130, Ultralytics 8.4.210.
3752 assignment rows, 3960 evaluation rows; zero duplicate record keys.
Recomputed 10304 scalar statistics across 92 assignment, 368 ranking, 552 stability summaries, zero mismatches (1e-8 tolerance).
E1 original view geometry rank counts exact in all four cases. Exact checkpoint and split verification reported. No training or A access reported.
Report lacks original checkpoint bytes, images and full raw candidate tensors.

## Endpoint assignment (pooled source training, clean classes)
| Branch/group | GT | Mean positives | Mean target mass | Mean target score | Box IoU | True push-up mass | Competitor abs mass |
|---|---:|---:|---:|---:|---:|---:|---:|
| O2M hard5 |450|10.0000|8.6410|.8641|.9771|.6423|.2880|
| O2M other13 |1156|9.9922|8.4540|.8459|.9743|.5238|.1476|
| O2O hard5 |450|1|.9471|.9471|.9763|.1081|.0026|
| O2O other13 |1156|1|.9471|.9471|.9737|.1079|.0122|

No zero-positive or zero-effective-target row among all 3752. No evidence of endpoint assignment starvation; O2M hard5 mean target mass is not lower than other13. Unnormalized BCE logit derivative proxy only, not actual historical training or parameter gradient. Cannot exclude early-training assignment defects or mask-quality effects.

## Candidate ranking (original view, IoU .75)
Pooled instance-weighted counts, not macro AP. Conditional rank denominator is geometry/score matched count (422 clean18, 116 hard5); oracle and native denominators include all GT (424,117).
| Scope/domain | Geometry top1/top3 | Score top1/top3 | Any top1/top3 | Correct native .001/.25 |
|---|---|---|---|---|
| Clean18 internal |278/336|345/386|395/415|412/359|
| Clean18 cross |133/244|180/285|305/382|384/200|
| Hard5 internal |71/82|92/101|104/112|115/96|
| Hard5 cross |25/46|35/57|64/96|100/38|

Cross clean score-selected top1 42.65%, vs 31.52% geometry-selected; hard5 30.17% vs 21.55%. Any-anchor top3 90.09% clean and 82.05% hard. Correct-class unique NMS match at .001 90.57% clean and 85.47% hard, vs .25 47.17% and 32.48%. Paired drop .001->.25: 184 clean, 62 hard; no reverse transition in this dataset. Independent per-GT any-anchor oracle can reuse anchors and is not deployable recall. Class-correct matching uses GT and differs from E1 geometry-first matching. No precision/AP improvement inferred from low threshold; native AP ranking considers false positives, not just GT coverage.

## Stability (score-selected, joint support)
| Scope/domain | N | Flip class changes | Dark | Light |
|---|---:|---:|---:|---:|
| Clean18 internal |422|28|12|19|
| Clean18 cross |422|93|23|45|
| Hard5 internal |116|13|6|6|
| Hard5 cross |116|27|6|10|

Cross clean score-selected rescue/damage flip 20/28, dark 7/5, light 10/19. Only 208/424 correct in at least one of four views, 216 never correct. Hard5 39/117 ever correct, 78 never correct. Re-matching combines anchor-selection and prediction variation. Monotone brightness changes are not guaranteed defect-cue preserving. Stability alone cannot solve stable errors; don't infer TTA AP.

## Decision
End broad diagnosis. First intervention hypothesis: object-level competitive class ranking over native mask-aware assigned O2M positives, auxiliary to unchanged native losses, rather than adding positive samples or changing geometry. Compare native RGB, same per-anchor competitive margin control, and cross-anchor object-level competitive margin under same B/C splits, schedule and initialization; no held-out A. Need preregister implementation, scale, assignment detach, positive set, branch weighting and matched-control details before training. Native score ranking and mask/box AP must improve; raw oracle is not enough. This is a testable candidate, not established novelty or evidence of performance. Closest prior work: FreeAnchor (NeurIPS 2019), Multiple Anchor Learning (CVPR 2020), TOOD and Rank & Sort (ICCV 2021), DivAlign (CVPR 2024). Existing bag/ranking learning is not novel; distinguish cross-anchor *inter-class* competition within fixed GT assignments, preserve native quality supervision. Generic margin/CE, bag learning, and consistency all require comparisons before novelty claim. Prior unsuccessful Faruq margin/refiner work makes a matched control important. E2 HAAR remains optional, absent in E3. No more open-ended diagnosis; use screening, then ablation/multi-seed and locked A confirmation. Single seed, small correlated B/C samples, confounded specimens/acquisition, and no direct causal proof of loss defect.
