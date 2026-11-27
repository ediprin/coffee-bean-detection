# M1/M2 competitive supervision — verified negative screen

Source: auliya-gc-competitive-report.zip SHA256
0111fcd2da9b0702497b0b0190ee95819e2d3b61fbcca0e0c56d3b251e18e899.
357 members. Code pin e53a6539c1f7d7744581fe410adf0266fa8c1fec.
33 tests passed, one SciPy deprecation warning. Preflight state equality,
11,452,523 parameters (unfused), 20 positive anchors for synthetic batch;
aux M1 2.508379, M2 1.414601, gradient L1 .814376/.712493.
All four 80-epoch runs, 19 batches/epoch, T4, Torch 2.11.0+cu130,
Ultralytics 8.4.210. 320 auxiliary epoch records. Run contracts match RGB
on split, schedule, pretrained asset, GPU/runtime and endpoint selection.
All new checkpoint references match progress JSON. No weights in archive;
byte-level checkpoint verification not repeated here. No A access reported.
Historical RGB reports exactly equal original G1/G2 report archive.
E3 original report SHA a84106845b83b73149ce187ca3b6663920cdbc44eb8f4b70a4c51dc236f6b6e6
verified against original E3 ZIP. 312 AP and rank aggregates recomputed,
all agree. Paired ranking CSV keys join one-to-one to E3 original rows.

## Equal direction weighted means (%)
| Model | Within box18 | Cross box18 | Cross hard5 | Cross bottom3 | Cross worst | Cross mask18 |
|---|---:|---:|---:|---:|---:|---:|
| M0 |90.217|46.440|34.356|19.620|13.840|40.832|
| M1 |87.416|44.770|33.441|14.161|11.043|39.420|
| M2 |89.478|45.250|33.508|16.280|10.788|40.001|

M2 vs M0: -1.190/-0.848/-3.340/-3.053/-0.831 pp on cross
box18/hard5/bottom3/worst/mask18. M2 vs M1 +.480/+ .066/+2.119/
-.256/+.582 pp. Both directions macro18 lower than M0.
B->C M0/M1/M2 box18 44.386/42.434/42.670, hard5
36.433/36.059/31.723, worst 15.063/8.614/10.809.
C->B box18 48.494/47.107/47.830, hard5 32.279/30.824/35.292,
worst 12.618/13.472/10.766. All 8 preregistered gates false.
Worst-class changes include M2 B->C multi-hole -23.94 pp, normal
-21.16; C->B small parchment -20.22, normal -11.52. Some class gains
occur but no consistent directional benefit.

## Mechanism check (equal direction weighting, original views, IoU .75)
| Scope metric | M0 | M1 | M2 |
|---|---:|---:|---:|
| Cross clean18 score Top1 |42.397|35.183|36.017|
| Cross clean18 score Top3 |67.184|62.711|54.987|
| Cross clean18 any Top3 |89.832|86.831|93.906|
| Cross hard5 score Top1 |30.376|24.612|26.792|
| Cross hard5 score Top3 |49.074|50.358|39.725|
| Cross hard5 any Top3 |81.935|83.050|89.501|
| Cross clean18 native correct .25 |47.032|38.283|44.663|
| Cross hard5 native correct .25 |32.698|22.507|29.267|

Clean18 score-selected raw coverage M0/M2 99.495%; hard5 99.091%.
Score-selected mean margin M0/M2 clean18 -2.157/-4.917,
hard5 -4.324/-7.782. Within clean18 score Top1 M0/M2 81.508/82.427.
Paired cross clean18 Top1 counts M0/M1/M2 180/149/153 (424 GT),
M1 rescue/damage 36/67, M2 34/61. Hard5 35/28/31 (117 GT),
M2 8/12 rescue/damage. Unmatched counted incorrect for paired counts.
Conditional rank metrics use matched denominator. Any-anchor oracle
uses GT and permits anchor reuse, not deployable AP/recall.
M2 increases oracle Top3 but worsens operational candidate ranking.
Auxiliary mean at epoch1->80: M1 B 4.092->.08115, C 4.061->.02616;
M2 B 3.117->.02094, C 3.001->.00978. Decreasing source loss does not
establish generalization. Same lambda does not imply matched gradient
magnitudes (M2 loss <= M1 for identical logits); no causal claim about
the reason for failure.

## Decision
STOP_OCCS_V1. Keep native RGB M0. Do not sweep margin/lambda or stack
OCCS with MASK2. No additional broad ranking/feature audit. Negative
result applies to this specific formulation, single seed and protocol,
not all competitive or contrastive methods. Next priority complete
independent RGB_MASK2 factorial experiment in Codex; compare all four
arms on matched eval grid, report Box18/hard5 and mask AP. If useful,
screen RGB_MASK2 (or HAAR_MASK2 only if incremental Haar benefit
established) on same B/C generalization split before claiming transfer.
If segmentation-only gains, treat as separate task; for classification
consider a controlled domain-generalization intervention with
augmentation-only control and native loss. Do not open held-out A or
change fixed gates retrospectively. Codex MASK2 raw artifacts have not
been audited in this result. No new method implemented here.
