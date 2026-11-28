# Auliya GC Semantic Transfer v1 — screening result (2026-11-28)

Decision: **STOP_SEMANTIC_TRANSFER_V1**. Retain S0 native RGB as reference. Do not run C->B, sweep residual width/levels, or combine with MASK2 on this evidence. This is a negative one-direction screening, not a general falsification of semantic feature sharing.

Source: user-provided auliya-gc-semantic-transfer-report.zip, SHA256 f43a0380840915192c348559b3e475a69a87d9b54677e2b2e85e26ccc8f1075f. 23 members, ZIP CRC passes. Scientific pin 8b74ae0e911ca61c029950f5fbe0ad2acde456c0. Reported 33 tests pass (one SciPy deprecation warning); preflight native initial state/loss parity true. Same B/C split, schedule, pretrained hash, runtime, group and fixed epoch selection as embedded S0 reference. Both run contracts match corresponding report, checkpoint hash matches progress marker, 80 finite epoch records per arm, source B only. No test A access reported. Recomputed 80 aggregate AP values from per-class AP across the four stored reports, and 104 ranking statistics from four CSVs (990 rows, unique keys per file, identical S1/S2 instance IDs and true classes per target); all match. Raw model weights, raw images, and independent historical baseline archive not available here; cannot reproduce AP or independently verify actual checkpoint bytes or access history.

## Results (% AP50-95)
| Metric | S0 | S1 | S2 | S2-S0 pp |
|---|---:|---:|---:|---:|
| B->C box18 |44.3863|45.5630|45.0866|+0.7003|
| B->C hard5 box |36.4325|34.2167|35.0112|-1.4213|
| B->C bottom3 box |23.6296|13.6450|19.7807|-3.8489|
| B->C worst box |15.0634|6.1486|18.2288|+3.1654|
| B->C mask18 |38.9145|39.9804|40.2389|+1.3244|
| B->C box21 |49.0273|47.1293|47.1363|-1.8910|
| B->C mask21 |43.8318|41.8062|42.6495|-1.1823|
| B->B box18 |91.6088|89.5820|89.6591|-1.9497|
| B->B hard5 box |90.4073|86.1611|88.4647|-1.9425|
| B->B worst box |77.5897|75.3400|70.5739|-7.0158|
| B->B mask18 |79.5046|78.3345|78.2857|-1.2189|

S2 versus S1: cross box18 -0.4764, hard5 +0.7945, bottom3 +6.1357, worst +12.0801, mask18 +0.2585 pp. S2 rescues some lower-tail loss relative to independent control but does not beat native hard5 or bottom3. Cross hard5 class deltas S2-S0: multiple holes -14.38, single hole +18.84, partly black -6.52, immature -4.30, large parchment -0.75 pp. Black +3.17, normal +17.14, small parchment +13.21; small/medium foreign -22.60/-26.34 pp. Small per-class support (10-12 on C) limits interpretation.

## Ranking (original-view IoU .75, external-NMS O2M)
Cross clean18 score-selected top1 S0 76/196, S1 74/196, S2 74/196; hard5 S0 18/54, S1 15/54, S2 19/54 (S0 counts inferred from registered E3 rates and denominators, not independent CSV check). S1->S2 clean18 paired all-GT top1 57 both correct, 17 rescued, 17 damaged, 107 neither; hard5 12 both, 7 rescued, 3 damaged, 33 neither. Native class-correct at .25 clean18 S1 78/198, S2 84/198; hard5 16/55, 18/55. Conditional ranking is not operational detection accuracy. S2 vs S0 conditional clean18 -1.02 pp, hard5 +1.85 pp. No broad ranking improvement.

## Gates
3/7 pass: cross worst nonregression, mask18 nonregression, within box18 >= -2pp (actual -1.9497pp). Fail cross box18 +2pp, hard5 +2pp, joint S2-S1 +.5pp, joint score top1 improvement. Reverse direction absent, not failed.

## Capacity
Train params S0/S1/S2 11,452,523 / 11,753,365 / 11,457,941. Fused params 10,373,467 / 10,671,344 / 10,671,344 (both inference modes), +297,877 vs S0 (~2.87%). S1/S2 matched deployed parameters but different training capacity; no isolated causal semantic-loss claim. 80 epochs S1 1069.87s, S2 1024.07s reported CSV time, not latency. Single seed, development group cross-shift conflates acquisition and specimens.

Checkpoints (reported): S1 75be7f93c84e170196661d782a0e8363d0b0d0fc8c7af96d24d5cac0de094857; S2 f3f1047d68cada99b06145506194da42cbe165fba9d33e79bbf65070f5b0e9fc.

Next: review D1/D2 and RGB_MASK2 results before choosing any new experiment. No test A, new training or method change in this audit.
