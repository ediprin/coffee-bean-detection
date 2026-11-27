# Controlled generalization V1 — result (2026-11-27)

Source: user-provided auliya-controlled-generalization-report.zip, SHA256
29678183f93a8de3092e8b3291cdf41ba86da9b11990b85aae3da76538d08bc7.

## Verification
20 tests passed, one SciPy deprecation warning. Both 80-epoch histories
complete. Run contracts identical except source group; split contracts
match registered partitions, same 300 source training and 75 target eval
images per group. Training B 976 instances, C 900; evaluation B 265, C 230.
Ultralytics 8.4.210, Torch 2.11.0+cu130, Tesla T4, seed 0.
All four clean18, all21, hard5, bottom3 and worst AP aggregates recomputed
from reported classwise AP; comparison.json contrasts verified.
Reported last.pt hashes agree with progress markers:
B 38385effe9335bf5ad23fee9913f19f5373494da1c20bdb578e1138ae52719fc;
C 41ffe7f52503d64ca4a008dd477e0575aa52604c8274f5c7005f1fb0a0723642.
Checkpoint bytes, raw images, and predictions are absent from report ZIP.
No test A access reported. Original source/test path in log is an archive
directory, not held-out group A. Training logs report 0 corrupt images.

## Results (%)
| Source -> target | Box18 | Box Hard5 | Box bottom3 | Box worst | Mask18 | Mask Hard5 | Mask bottom3 | Mask worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B->B |91.61|90.41|81.60|77.59|79.50|78.89|68.07|66.81|
| C->B |48.49|32.28|15.61|12.62|42.75|27.92|13.03|10.54|
| C->C |88.82|81.79|69.66|63.75|77.97|71.95|58.80|52.98|
| B->C |44.39|36.43|23.63|15.06|38.91|30.95|19.17|11.79|

Matched target differences (internal minus cross), percentage points:
B box18 43.12, box Hard5 58.13, mask18 36.76, mask Hard5 50.97.
C box18 44.44, box Hard5 45.35, mask18 39.05, mask Hard5 41.00.
Unweighted direction averages: internal/cross box18 90.22/46.44;
hard5 86.10/34.36; bottom3 75.63/19.62; worst 70.67/13.84.
Mask18 78.74/40.83; hard5 75.42/29.44.
All 18 clean classes have lower box AND mask AP for cross-group than
internal on both matched target sets. Black bean cross box AP 13.91
(C->B) and 15.06 (B->C); partly black 20.30 and 34.60;
single hole 12.62 and 51.90. Thus prior fixed hard5 does not cover every
vulnerable class; do not retrospectively change the hard5 definition.

## Interpretation and next action
Strong evidence of group-dependent transfer difficulty, not causal
identification of acquisition vs specimen shift or classification vs
localization error. Same image count, not equal instance count.
Within-group correlated repeated specimens can inflate internal AP.
Single seed; evaluation supports 10–13 instances per class; prior B/C
development exposure, no iid significance claim or independent confirmation.

Priority: paired HAAR vs RGB on the exact G1/G2 source/target splits and
epoch-80 schedule, no operator change. RGB references frozen. Use both
directions; primary box18 and fixed hard5, mask and bottom-tail supplementary.
The official-fold-A HAAR evidence is modest and mask-only; a positive
within-group result does not establish cross-group benefit. Obtain
per-instance class-agnostic and class-ranking diagnostics to distinguish
classification from geometry. No architecture development, new wavelet
variant, or test A opening before screening. This report does not initiate
training or implement the HAAR arm.
