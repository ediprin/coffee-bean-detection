# Auliya spatial distribution audit — result and stop decision (2026-11-27)

Source: user-provided auliya-spatial-audit-results.zip, SHA256
dae26c06ebf446eba94e467b80aac2cc54ccfa78ac4b89a9ddf789b5ba177a09.
Analysis also reads the two prior user-provided visual/feature result ZIPs.
No raw images, training, or group-A evaluation were performed for this report.

## Verification
16 tests passed (one SciPy deprecation warning); 795 shards, 1241 B and
1265 C instances, same ordered IDs and labels as prior feature audit.
P2/P3 R1 arrays exactly equal prior F1/F2 train and validation arrays
(max absolute difference zero); matched and raw R1 predictions identical
to corresponding F1/F2 predictions. Recomputed all 16 per-class F1 vectors
from the CSV; agree with report. Manifest-verified extracted input fingerprint
12d60838ae810a2841607fd5a09afe6b77c8a48977ea2b9c6bc3a52394d55166;
original ZIP byte equality unverified; no test A access reported.
RGB checkpoint SHA 85b9b459fcd5079de3c27f8b897a4bd0c7160efa96bea25957c5eadd4d617ed3.
Training-only calibration reports 79424 samples per level, no repeated
adjacent quartile edges. Runtime Tesla T4, ultralytics 8.4.96, sklearn 1.6.1.

## Results (%)
| Level/operator | matched Macro18 | Hard5 | Worst18 | raw Macro18 | Hard5 | Worst18 |
|---|---:|---:|---:|---:|---:|---:|
| P2 R1 |40.87|23.29|7.41|44.69|19.55|7.77|
| P2 R2 |41.43|12.59|2.99|39.77|14.55|8.57|
| P2 R3 |38.33|13.84|3.60|44.13|15.67|5.37|
| P2 R4 |41.14|9.08|0.00|41.86|12.97|3.03|
| P3 R1 |33.73|17.33|10.00|34.58|16.83|7.23|
| P3 R2 |29.34|17.34|2.02|29.75|16.20|5.00|
| P3 R3 |31.30|12.13|6.11|33.58|18.06|7.34|
| P3 R4 |32.34|17.03|4.60|32.77|17.46|2.35|

0/6 matched candidates pass the preregistered joint threshold.
P2 matched R2/R3/R4 hard-set rescue/damage: 12/43, 11/42, 12/52.
Hard-set size 309; R1 correct 68; union of all four P2 matched operators
correct 97 (31.39%), not an attainable classifier estimate.
P2 matched Hard5 average factorial contrasts: histogram -7.73 pp,
regional -6.48 pp, interaction +5.94 pp. Positive interaction does not
imply R4 outperforms R1. R2 and R3 improve zero hard classes; R4 improves
one (+0.21 pp partly black); R4 large parchment true positives 0/60.
All raw P2 alternatives also lower Macro18 and Hard5.

## Persistent errors (exploratory, same C)
IDs and labels agree exactly across all three result archives.
Across 21 matched columns (6 visual + 7 feature + 8 spatial, includes
duplicated R1/F1,F2), 104/309 hard-set objects are never correctly
classified. Across all 42 matched/raw probe columns, 62/309 are never
correct; these are not independent classifiers. The latter by true class:
single hole 14/68, partly black 4/60, immature 15/60, large parchment
28/60, multiple holes 1/61. No claim of irreducible error or annotation
incorrectness; even the oracle is limited to recorded predictions.

## Decision
STOP_HISTOGRAM_QUADRANT_DIRECTION. No ISDE integration, no bin-count
or quadrant sweep, no new P2 head on this evidence. Keep mean pooling as
diagnostic reference. This negative result is specific to fixed quartile
histograms, axis-aligned quadrants, frozen features and linear probes;
does not falsify all distribution or spatial representations.
Next priority: blinded visual label-observability review of persistent
errors with matched correct controls, followed by a controlled
generalization analysis that separates within-acquisition evaluation
from held-out specimen/acquisition evaluation where metadata allows.
A specialist review is needed before class/annotation changes. Do not
tune on group A. B and C differ in acquisition and specimens, so the
current design does not establish causal domain-shift or overfitting.
