# E1 frozen error audit and E2 HAAR comparator (registered 2026-11-27)

The controlled G1/G2 experiment established large within-to-cross-group
gaps. The method remains open; the old proposal does not constrain the
choice. E1 v2 is a no-training diagnostic. E2 is an optional separate paired
training screen. E3 (assignment, gradient, feature stability) is deferred
until E1 results identify what to test. Do not tune on group A.

## E1: geometry access and class ranking
Reuse exact G1/G2 last.pt SHA256 values from the verified results report.
Reconstruct the same frozen B/C split from the audited dataset. Run both
models on both 75-image eval sets (300 forward passes), FP32 square 640
letterbox. Use the pinned Ultralytics 8.4.210 raw one-to-many output and native
external class-aware NMS (multi-label, IoU .7, max_det 300) at each confidence.
The original training logs show nms=None. In this version validator sets
end2end=(nms is False); thus external NMS is the baseline inference mode.
E1 v1 incorrectly assumed one-to-one, and stopped before inference. V2
preserves exact checkpoint and class order, explicitly selects historical
mode, converts decoded raw xywh to xyxy and uses the native NMS function.
NMS time limit extended to avoid partial outputs in single-image audit. No weights change.
At IoU .50 and .75, match GT to candidates one-to-one, maximizing valid
match count then total IoU, ignoring class. For raw 8400 anchors record
coverage, rank of true label, logit margin, true sigmoid score and top1.
For native output at .001 and .25 record correct-class match, wrong-class
match or unmatched GT. Report conditional accuracy, top1/top3 conditional
on raw geometry match, margins and per-class counts; preserve per-instance
CSV. Raw geometry access is an oracle, not operational recall. Native
matches are geometry-first; a correct-class alternative may exist even
when the assigned candidate has wrong class. Unmatched means no unique
assignment at the given IoU and confidence, not necessarily missing
geometry. This is a GT-side diagnostic, not full TIDE decomposition,
not AP reproduction, and does not partition all false positives.
Square inference differs from official rectangular validation batching.
No mask error attribution, causal separation or statistical significance
claimed. Compare within vs cross on the same target, and compare .50/.75
and confidence thresholds before interpreting classification vs geometry.

## E2: HAAR comparator, no new method
Same exact HaarLuminanceEnhancer as official-fold screening, no new gain
or angular operator. G1-HAAR and G2-HAAR independently initialized from
official YOLO26s-Seg; same fixed 80 epochs, native schedule, split, seed,
batch, GPU and Torch/Ultralytics versions as completed RGB. Same nominal
training budget. No target-group validation during training. Report same
box/mask metrics. RGB references frozen, not retrained. Verify RGB report
runner SHA, checkpoint reference SHA, split, runtime; no silent fallback
if runtime differs. Note other dependencies and nondeterminism may vary.
Screen on two cross directions: both Macro18 gains positive, mean Macro18
and mean Hard5 gain >=2 pp each, Worst AP loss <=1 pp in each direction.
Within-group metrics are secondary regression checks. Gate is exploratory,
not significance. Failure stops HAAR optimization by default; success is
evidence for controlled ablation, not novelty or final confirmation.

## How to run
Import Auliya_GC_Diagnosis_Kaggle.ipynb; attach existing extracted dataset
AND full output of Auliya_Controlled_Generalization_Kaggle, including both
runs/B/weights/last.pt and runs/C/weights/last.pt, and corresponding reports.
A report-only zip is insufficient for E1. Default RUN_E1=True, RUN_HAAR=False
for fast diagnosis. For independent E2 session set RUN_E1=False,
RUN_HAAR=True. Save Version and retain full checkpoint output for resume.
V2 uses a new diagnosis output directory to avoid mixing old code pins and results.
If both enabled, runs E1 first then HAAR. Report archive excludes weights.
No actual runtime execution or tests claimed in implementation.
