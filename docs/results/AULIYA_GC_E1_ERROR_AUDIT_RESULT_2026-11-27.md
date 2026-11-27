# E1 V2 frozen error diagnosis — result (2026-11-27)

Source auliya-gc-diagnosis-report.zip SHA256
972c461ad9049e04f7a7a85f067fcec59c8891fa56bd608b2035fc22fc29629e.
24 tests passed, one SciPy deprecation warning; T4, Torch 2.11.0+cu130,
Ultralytics 8.4.210. Code pin e1d6c03d929df09e97d57a8e48f62ff71af3c62b;
runner SHA 73178436cbf95ce7ed8fec47bce8c5e1727805156c9be9d1e43b0da67b2212d9.
Both exact historical checkpoint hashes verified by runner. Segment26,
one-to-many external NMS (nms=None, IoU .7) as historical validation.
3960 CSV records, four conditions, two IoU and two confidence thresholds.
All 368 summary records recomputed from CSV: counts, conditional ranks,
median margins and native conditional accuracy agree. No training or A
access reported. Report lacks checkpoint bytes, images and full predictions.

## Primary IoU .5, confidence .25 (clean18)
| Condition | n | Raw matched | Raw top1 | Raw top3 | Margin median | Native correct/wrong/miss |
|---|---:|---:|---:|---:|---:|---|
| B->B |226|226|68.14%|80.97%|3.566|187/36/3|
| C->B |226|226|35.40%|62.39%|-2.051|94/113/19|
| C->C |198|197|62.94%|77.66%|2.603|153/41/4|
| B->C |198|196|27.04%|52.55%|-2.692|70/112/16|

Pooled (instance-weighted, not macro AP): internal raw coverage 423/424;
cross 422/424; at .75 internal and cross each 422/424. Median matched
raw IoU internal .98707, cross .98377 (IoU .5). Raw conditional top1
278/423=65.72% vs 133/422=31.52%; top3 336/423=79.43% vs
244/422=57.82%. Median margin +3.268 vs -2.266. Native coverage
417/424 vs 389/424; correct 340/424 vs 164/424; wrong 77 vs 225;
miss 7 vs 35. Conditional accuracy 340/417 vs 164/389.
Hard5 internal/cross raw top1 71/116 vs 25/116, top3 82/116 vs 46/116;
native correct/wrong/miss 90/25/2 vs 30/78/9 of 117 each.
Paired GT IDs: 140 correct in both, 200 internal-correct lost, 24 rescued
by cross, 60 wrong/missed in both. Of 200 lost, 175 assigned wrong class,
25 unmatched. Not causal partition of AP gap.
Class-specific cross correct: black 2/23, partly black 2/24, single hole
6/23, multi-hole 5/23; black raw top1 1/23. Not independent specimens.
Examples cross geometry-first confusion: large parchment->parchment-
covered 10, black->brown 7, multi-hole->single-hole 7. Low confidence
(.001) substantially increases wrong-class matches because matcher
optimizes geometry only; even a correct-class alternative may exist.
Do not infer lower threshold worsens detector from this statistic.
Raw coverage is an oracle over all anchors, not deployed recall.
Raw rank evaluated at geometry-selected anchor, not highest-scoring
anchor; cannot claim a theoretical ceiling on reranking. No FP-side AP
decomposition, mask attribution or causal proof of supervision defect.

## Decision
High-priority bottleneck is generalization of class discrimination rather
than missing raw box geometry; confidence filtering also contributes.
Simple top-k reranking alone not a sufficient primary hypothesis,
especially Hard5 where only 46/116 matched true classes rank <=3.
E3 should be a bounded diagnostic: positive assignment and target-score
quality on training-only data, and object-level class stability under
label-preserving input changes on development sets; compare geometry
selected and score-selected candidates / class-correct alternatives.
No repeat of exhaustive histogram or feature-fusion screens. E2 HAAR is
optional comparator; absent in this report. Do not infer any benefit.
Select intervention based on mechanism, not proposal commitment.
Held-out A remains locked; single seed, small correlated development
sample, source specimens and acquisition confounded. No causal claim
about loss, feature representation or domain shift component.
