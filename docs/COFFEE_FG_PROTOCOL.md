# CoffeeFG-YOLO26 v2

## Screening outcome — 2026-08-01

The quick-10 validation screen is complete and the study is **stopped**.
P2 did not improve proposal accessibility, while both authorized P3 ROI
refiners substantially degraded macro and lower-tail AP. R0Q fell 8.77
percentage points and R1Q fell 10.86 points in macro AP50-95 relative to D0Q;
R1Q also lost 2.09 points against its capacity-matched first-order control.
Test remained locked. Do not run the full schedule or additional seeds.

The frozen result and artifact provenance are recorded in
[`COFFEE_FG_QUICK10_RESULT_2026-08-01.md`](COFFEE_FG_QUICK10_RESULT_2026-08-01.md).

## Research question

Can candidate-level cross-scale second-order classification improve
fine-grained coffee-defect detection after the objects are already accessible
to the detector?

CoffeeFG is a single detector at inference. It does not save crops or call an
external classifier. Internally it decodes candidates, applies ROIAlign to
feature-pyramid maps, refines class logits, and returns boxes and classes in one
model call. It is not described as fully differentiable through proposal
selection: top-K selection and proposal coordinates are detached.

The valid claim is:

> The box-head modules, box assignment, and box loss are unchanged. Refiner
> gradients may still alter the shared backbone/neck features and therefore
> may indirectly affect localization.

## Frozen factors

The experiment separates pyramid resolution, ROI capacity, and second-order
interaction:

| Code | Pyramid | Classification path | Question |
|---|---|---|---|
| D0 | P3-P5 | stock YOLO26n | baseline |
| D1 | P2-P5 | stock YOLO26n | does P2 improve proposal accessibility? |
| R0 | P3-P5 | first-order P3/P4 ROI | ROI-capacity control without P2 |
| R1 | P3-P5 | bilinear P3/P4 ROI | bilinear effect without P2 |
| R2 | P2-P5 | first-order P2/P3 ROI | ROI-capacity control with P2 |
| R3 | P2-P5 | bilinear P2/P3 ROI | bilinear effect with P2 |

R0/R1 and R2/R3 use identical rank, ROI size, candidate count, schedule, loss,
and refiner parameter count. The primary mechanism comparisons are therefore:

```text
R0 vs R1  when diagnostics select D0/P3-P5
R2 vs R3  when diagnostics select D1/P2-P5
```

`D0 vs R3` is not a valid bilinear ablation because it mixes P2, extra
capacity, ROI refinement, and bilinear interaction.

The local P3 and P2 YAML files are pinned from Ultralytics 8.4.96 rather than
resolved from a moving upstream branch.

## Mandatory diagnostic gate

### Quota-aware fail-fast screen

Before the full 50-epoch D0/D1 comparison, `D0Q` and `D1Q` may be trained for
10 epochs with an otherwise identical schedule. This quick comparison is a
screening device only: failure stops P2, while a pass authorizes the original
full schedule but is not itself final evidence. Quick and full checkpoints use
different run codes and output roots and must never be aggregated together.

Train only D0 and D1 on validation first. Then run:

```bash
python -u -m coffee_detector.analysis.coffee_fg_diagnostics \
  --p3-checkpoint /path/to/D0_seed42/weights/best.pt \
  --p2-checkpoint /path/to/D1_seed42/weights/best.pt \
  --data-root /path/to/grouped-detection-data \
  --output /path/to/coffee-fg-v2/val_reports/diagnostic_seed42.json \
  --split val \
  --candidate-counts 50 100 300 500 \
  --max-det 500 \
  --device 0
```

The audit reports, for both one-to-one and one-to-many branches:

- proposal accessibility at each pre-refinement K;
- greedy one-to-one matched recall;
- classification accuracy conditional on IoU-correct localization;
- wrong-class, missed, and duplicate candidate counts;
- matched-class confusion;
- the maximum class-accuracy headroom if localized candidates received the
  oracle class;
- exact-count accuracy, count MAE, and signed bias for one-to-one output and
  one-to-many+NMS;
- validation density and whether `max_det` covers its largest image.

Refinement is rational only when the selected foundation reaches at least 90%
proposal accessibility at the largest K and retains at least 2% localized
wrong-class headroom. These thresholds are protocol defaults, not universal
constants; change them only before running the diagnostic.

If P2 improves accessibility by at least one percentage point, the diagnostic
selects R2/R3. Otherwise it selects R0/R1. The screening runner rejects refiner
training without this report and rejects a refiner pair that contradicts it.

V2 deliberately uses one-to-one candidates for both predicted-candidate
training and inference, preserving YOLO26's NMS-free deployment path.
One-to-many+NMS is a diagnostic comparator. If it has materially higher
accessibility while one-to-one fails the gate, stop V2; do not silently switch
proposal sources. A separate NMS-based protocol would then be required.

## Refiner

For candidate box \(b_k\), two levels are aligned by ROIAlign:

\[
A_k=\phi_a(\operatorname{ROIAlign}(F_a,\operatorname{stopgrad}(b_k))),
\quad
B_k=\phi_b(\operatorname{ROIAlign}(F_b,\operatorname{stopgrad}(b_k))).
\]

First-order control:

\[
h_k^{FO}=W_f[\operatorname{mean}(A_k);
                  \operatorname{mean}(B_k)].
\]

Low-rank spatial bilinear interaction:

\[
h_k^{LRBP}=
\operatorname{norm}\left(
\operatorname{signedsqrt}
\left(\operatorname{mean}(A_k\odot B_k)\right)
\right).
\]

Residual class logits:

\[
z_k^{final}=\operatorname{stopgrad}(z_k^{YOLO})
             +\alpha z_k^{refiner}.
\]

The first-order control contains a capacity-matching matrix so both modes have
exactly equal trainable parameter counts for the same channels and rank.

## GT-to-predicted curriculum

Training no longer relies on GT ROIs for all epochs:

1. epochs 0-9: GT boxes teach a stable object descriptor;
2. epochs 10-24: linearly mix GT ROI loss with IoU-matched predicted
   candidate loss;
3. epoch 25 onward: predicted candidates only.

Candidate labels come from one-to-one GT matching at IoU >= 0.5. Ambiguous and
unmatched candidates are ignored. Proposal coordinates and base candidate
logits are detached. Refiner feature gradients use the original one-to-many
feature maps so the shared representation can still learn.

This schedule reduces GT/inference mismatch but does not make top-K selection
fully differentiable.

## Dense and conveyor contracts

YOLO26 end-to-end output defaults to 300 detections. The v2 configs use
`max_det=500` and refiner `topk=500` because a 300 g scene may contain roughly
220-340 beans. The diagnostic must still prove that this cap covers the
validation distribution.

Report two deployment regimes separately:

- conveyor: sparse images, latency with the validated smaller K;
- 300 g sample: dense images, K/max_det >= the validated object-count tail.

Do not report one FPS number as if it represented both.

## Dataset and evaluation

- Group by source identity before splitting.
- Augmented siblings and crops from one source image stay in one split.
- Screening and diagnostics use validation only.
- Test requires explicit `--evaluation-split test --open-test` after the
  architecture, K, confidence, and all thresholds are frozen.
- Every validation class must have ground-truth instances. The evaluator stops
  rather than filling absent classes with overall mAP.

Minimum final metrics:

- box AP50 and AP50-95;
- per-class, macro, bottom-three, and worst-class AP50-95;
- proposal recall/accessibility versus K;
- localized class accuracy and confusion;
- missed, duplicate, and wrong-class counts;
- exact-count accuracy, count MAE, and signed bias;
- AP by size and density stratum;
- parameters, MACs, FP32 size, detector/refiner/total latency, and throughput;
- one-to-one versus one-to-many+NMS.

## Commands

Stage 1 trains only the two stock foundations:

```bash
python -u -m coffee_detector.experiments.run_coffee_fg_screening \
  --data-root /path/to/grouped-detection-data \
  --output-root /path/to/coffee-fg-v2 \
  --models D0 D1 \
  --seeds 42 \
  --evaluation-split val \
  --device 0
```

After the diagnostic selects P3:

```bash
python -u -m coffee_detector.experiments.run_coffee_fg_screening \
  --data-root /path/to/grouped-detection-data \
  --output-root /path/to/coffee-fg-v2 \
  --models D0 R0 R1 \
  --seeds 42 \
  --evaluation-split val \
  --diagnostic-report /path/to/diagnostic_seed42.json \
  --device 0
```

If it selects P2, replace the models with `D1 R2 R3`. Do not open test during
screening.
