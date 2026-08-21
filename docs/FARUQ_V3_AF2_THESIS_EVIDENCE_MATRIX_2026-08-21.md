# AF2 Thesis Evidence Matrix

Date: 2026-08-21

Status: **frozen synthesis of completed repository evidence**

This document does not authorize new training, tuning, or test access. It maps
each possible thesis statement to its strongest available evidence and its
explicit boundary.

## Evidence strength convention

| Level | Meaning |
|---|---|
| A | Frozen paired multi-seed evidence on parent-grouped development data |
| B | Leakage-audited independent external-source evidence without target training |
| C | Controlled post-hoc or single-seed development diagnostic |
| D | Weak/correlated exploratory evidence; context only |
| X | Unsupported or contradicted; must not be claimed |

## Core evidence matrix

| Question | Evidence | Strength | Defensible statement | Boundary / forbidden extension | Authority |
|---|---|---:|---|---|---|
| Is the development split suitable for controlled comparison? | Faruq-v3 contains 1,665 train images/2,986 instances and 294 validation images/526 instances; all 21 classes have 24--26 validation instances; exact-hash and parent overlap are zero; test is absent from the development archive. | A | The development comparison is parent-grouped and class-complete. | This does not create an untouched final in-domain test for AF2. | `FARUQ_V3_BASELINE_PROTOCOL.md` |
| Is AF2 better than an optimization-matched control in-domain? | Across seeds 42/123/2026, D0FT to AF2: Macro 86.62% to 87.94% (+1.32 points, 3/3 seeds); Bottom-3 76.58% to 79.37% (+2.80, 2/3); Worst 73.05% to 78.15% (+5.10, 2/3). Every frozen confirmation gate passed. | A | AF2 provides a stable mean validation improvement, with its largest benefit in lower-tail classes. | Do not call this independent test superiority or cross-paper SOTA. | `FARUQ_V3_AF2_IGEM_PAIRED_CONFIRMATION_RESULT_2026-08-15.md` |
| Is the gain merely extra optimization? | Every AF2 seed starts from the same seed-matched D0 checkpoint and is compared with the corresponding D0FT schedule control. | A | AF2 gains are measured beyond matched continued optimization. | Comparing AF2 only with the earlier D0 baseline would overstate the architectural effect. | AF2/IGEM protocol and evidence JSON |
| Does AF2 improve raw localization geometry? | Raw top-500 proposal accessibility is already saturated: D0FT 99.81%, AF2 99.75%, delta -0.06 point, improved 0/3 seeds. | A diagnostic | There is no evidence that AF2 generates more geometrically accessible raw proposals. | Do not claim localization improvement from final recall alone. | `FARUQ_V3_AF2_MECHANISM_DIAGNOSTIC_RESULT_2026-08-21.md` |
| Does AF2 improve fine-grained class decision after localization? | Conditional Top-1 accuracy rises 62.46% to 70.58% (+8.12 points, 3/3); localized wrong-class rate falls by 8.12; correct-decision recall rises 48.54% to 63.18% (+14.64, 3/3). | A diagnostic | AF2's observed mechanism is classification/ranking dominant. | This is post-hoc association, not causal proof for every class. | Mechanism diagnostic result/evidence |
| Is the benefit uniform across classes? | Large conditional gains occur for several classes, but `kulit_tanduk_ukuran_sedang` and `biji_berlubang_satu` have negative mean conditional deltas. | C | The aggregate and lower-tail distribution improve despite heterogeneous class effects. | Do not claim every SNI class improves. | Mechanism diagnostic per-class evidence |
| Does AF2 transfer to an independent acquisition source without target training? | Leakage-safe Coffee Standard evaluation retains 148 independent parent identities, 3,989 instances, and 18 directly mapped classes. Across three seeds, Macro rises 11.43% to 15.51% (+4.08 points, minimum +3.36, 3/3). | B | AF2 consistently improves target-free cross-dataset Macro directionally. | Absolute performance remains poor; three unmapped SNI classes and seven excluded source labels prevent full-taxonomy deployment claims. | `COFFEE_STANDARD_V8_EXTERNAL_RESULT_2026-08-16.md` |
| Is AF2 generally robust to illumination? | At seed 42, AF2 has positive Macro advantage in only 2/9 isolated photometric conditions; Worst-class mean advantage is -2.32 points and minimum is -10.23. Frozen screen fails. | C / X for broad robustness | AF2 helps warm/cool shifts but does not demonstrate general illumination robustness. | Never summarize the Coffee Standard result as universal lighting robustness. | `FARUQ_V3_AF2_ILLUMINATION_ROBUSTNESS_RESULT_2026-08-17.md` |
| Does AF2 solve dense 220--300-object scenes? | Synthetic density diagnostic shows AF2 improves 3/4 conditions but is not the leading synthetic-density model; absolute mAP remains very low and no independent real-dense benchmark exists. | C / X for deployment | Synthetic density is a stress-test limitation, not the primary AF2 claim. | No real 300-gram, conveyor, counting, or dense deployment claim. | Coffee Standard consolidated result and Drive synthetic report |
| Does AF2 generalize on Adrian? | All candidates collapse on Adrian; only eight parent identities exist. | D | Adrian is weak corroborative context only. | Do not use Adrian to establish external robustness or rank AF2. | Coffee Standard consolidated result |
| Is AF2 parameter-free? | D0FT and AF2 each contain exactly 2,511,990 parameters and 10,124,840 serialized state-tensor bytes across all seeds. | A efficiency | AF2 is a parameter-free input frontend relative to D0FT. | Parameter-free does not mean compute-free, buffer-free, or memory-free. | `FARUQ_V3_AF2_EFFICIENCY_AUDIT_RESULT_2026-08-21.md` |
| What is AF2's deployment cost? | Tesla T4 paired audit: median latency 13.52 to 23.59 ms (1.745x), p95 19.15 to 33.78 ms (1.767x), throughput 68.93 to 39.96 image/s (0.581x), peak allocated memory 75.2 to 127.6 MB (1.696x). | A efficiency | Lower-tail accuracy is purchased with substantial FFT latency and temporary CUDA memory. | Tensor-forward throughput is not full-system FPS; standard YOLO FLOPs omit AF2. | Efficiency audit result/evidence |
| Can simple radial or orientation redesign improve AF2? | Radial fails at seed 42. Orientation improves seeds 42/123 but fails three-seed confirmation: Macro +0.32, Bottom-3 -0.12, Worst -1.50 points; original AF2 retained. | A for orientation decision | Original AF2 is more stable than the tested radial/orientation variants. | Do not promote the attractive seed-42 orientation result. | Isolated and paired orientation result documents |
| Does illumination-conditioned gating improve AF2? | AF2R1 loses to its equal-parameter zero-information control by -0.62 Macro, -1.13 Bottom-3, and -1.40 Worst at seed 42. | C | The proposed adaptive conditioning mechanism is rejected. | Its raw gain over frozen AF2 is confounded by continuation optimization. | `FARUQ_V3_AF2_ADAPTIVE_RESIDUAL_GATE_RESULT_2026-08-17.md` |
| Does channel calibration explain the continuation gain? | AF2CAL3 loses to AF2FT30 by -0.23 Macro, -0.17 Bottom-3, and -0.55 Worst. | C | Three learned channel scales do not explain or improve the AF2 continuation result. | Do not attribute AF2R0 gains to channel calibration. | `FARUQ_V3_AF2_CHANNEL_CALIBRATION_RESULT_2026-08-17.md` |
| Do extra DG/FG objectives improve AF2? | AF2DG, AF2FG, and AF2DGFG fail the frozen seed-42 factorial decision; the joint arm is below the control on all primary metrics. | C | Additional regularizers do not improve the selected AF2 under this protocol. | This does not prove all domain-generalization losses are universally ineffective. | `FARUQ_V3_DIDA_AF2_FACTORIAL_RESULT_2026-08-17.md` |
| Is the Hong transfer a stronger coffee-specific solution here? | Full Hong-derived YOLO26 transfer loses 4.92 Macro, 23.58 Bottom-3, 33.01 Worst, and 16.89 conditional Top-1 points, despite +11.79 proposal accessibility. | C | Coffee-specific prior art does not automatically transfer across YOLO version and 21-class taxonomy. | Do not claim Hong et al.'s original YOLOv10 method is invalid; only this transfer failed. | `HONG_YOLO26_TRANSFER_RESULT_2026-08-02.md` |

## Thesis claim ledger

### Claims that are supported

1. AF2 is an end-to-end, one-stage, parameter-free frequency-angular input
   frontend integrated with YOLO26; it does not use a second ROI classifier.
2. Relative to seed- and optimization-matched D0FT, AF2 improves three-seed
   grouped-validation Macro and mean lower-tail AP.
3. The observed gain is classification/ranking dominant rather than a gain in
   raw proposal geometry.
4. AF2 improves target-free cross-dataset Macro on the leakage-safe,
   directly-mapped Coffee Standard subset in all three seeds.
5. AF2 has an explicit accuracy--efficiency trade-off: unchanged parameter
   count but higher FFT latency and peak CUDA memory.
6. Several theoretically plausible AF2 extensions fail controlled gates,
   supporting retention of the simpler original operator.

### Claims that are not supported

1. AF2 is a newly invented Fourier algorithm.
2. AF2 is state of the art against papers that use different data, taxonomies,
   splits, hardware, or metrics.
3. AF2 has been confirmed on an untouched Faruq-v3 in-domain test.
4. AF2 is universally robust across illumination, acquisition domains, coffee
   varieties, or all 21 classes.
5. AF2 is ready for real 300-gram dense scenes, conveyor operation, counting,
   or SNI deployment.
6. AF2 improves raw localization geometry.
7. Approximately 40 image/s tensor-forward throughput is full-system FPS.

## Final evidence disposition

AF2 is the selected primary thesis model. D0FT is the mandatory causal control.
IGEM1 is a confirmed secondary candidate but is not the thesis center. Original
D0, Hong transfer, ROI refiners, ontology heads, multilevel heads, synthetic
fusion models, and AF2 extensions belong in the screening/negative-ablation
narrative rather than the principal comparison table.

