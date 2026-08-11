# Hong-to-YOLO26 Full Transfer — Validation Result

Date: 2026-08-02  
Protocol: `HONG-YOLO26-TRANSFER-v1.2.0`  
Dataset: Faruq-v3 grouped development split  
Candidate: `HF` (YOLO26n-P3 + three Hong DSConv sites + SPPF-Attention + PConv detection heads)  
Baseline: completed `D0_seed42` checkpoint  
Seed: 42  
Evaluation: validation only; test remained locked

## Frozen gate result

**Decision: FAIL.** The predefined stopping rule applies. Do not run component
controls, additional seeds, or the locked test for this candidate.

| Criterion | HF minus D0 | Result |
|---|---:|---|
| Macro mAP50-95 | -4.92 points | FAIL |
| Conditional top-1 accuracy | -16.89 points | FAIL |
| Bottom-3 class mAP50-95 | -23.58 points | FAIL |
| Worst-class mAP50-95 | -33.01 points | FAIL |
| Proposal accessibility | +11.79 points | PASS |
| Operational correct-decision F1 | -11.05 points | FAIL |
| Batch-1 latency ratio | 1.4287x (+42.87%) | FAIL |

Only the proposal-preservation criterion passed. All accuracy, lower-tail,
operational, and latency criteria required by the frozen protocol failed.

## Interpretation

The complete Hong-derived package made more ground-truth objects accessible to
the candidate set, but it substantially reduced the probability of assigning
the correct fine-grained class once a proposal was accessible. The failure is
therefore not explained by proposal scarcity. Its dominant observed signature
is degraded class discrimination, especially for the lower tail, accompanied
by excessive latency overhead.

Because `HF` combines DSConv, SPPF-Attention, and PConv in one arm, this result
does not causally identify which component produced the degradation. The
pre-registered protocol permits component/removal controls only after the full
candidate passes. Running those controls after this result would reverse the
declared gate and create post-hoc search, so they are not authorized.

This one-seed fail-fast result does not prove that Hong et al.'s architecture is
universally ineffective. It shows that this faithful mechanism transfer from
the paper's YOLOv10 setting to the present YOLO26n-P3/Faruq-v3 protocol is not a
qualified improvement over D0.

## Artifact status

- Training resumed successfully from the shared-Drive `last.pt` and completed
  the seed-42 run.
- Checkpoint parameter names confirmed that KDS/CDS DSConv and partial-channel
  PConv modules survived serialization and resume.
- Raw checkpoint and reports remain outside Git under
  `Coffee_Bean_Detection/experiments/hong-yolo26-transfer-v1/`.
- Test images accessed: **no**.
- Test opened: **no**.

