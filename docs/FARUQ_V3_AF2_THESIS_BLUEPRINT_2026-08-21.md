# Thesis Blueprint: AF2-YOLO26 for Fine-Grained SNI Coffee Defects

Date: 2026-08-21

Status: **recommended thesis position based on completed evidence**

## Recommended title

### Indonesian

**Adaptasi Frontend Frekuensi-Angular Tanpa Parameter pada YOLO26 untuk
Deteksi Fine-Grained Cacat dan Kontaminan Biji Kopi Berbasis SNI**

### English

**Parameter-Free Frequency-Angular Frontend Adaptation for YOLO26 in
Fine-Grained SNI Coffee-Bean Defect and Contaminant Detection**

Use `adaptation`, not `invention`, because the AFAB-2 frequency mechanism is
transferred from LFDet. The thesis-specific contribution is its controlled
one-stage YOLO26 adaptation, SNI-21 validation, mechanism attribution,
external evaluation, and efficiency characterization.

## Central problem

Published coffee detectors often report global detection metrics on coarse or
merged labels. The present 21-class SNI setting retains subtle local defects,
color/texture boundaries, material contaminants, and size families. After raw
proposal accessibility is controlled, the remaining bottleneck is the
conditional class decision rather than absence of geometrically valid boxes.

The thesis therefore does not ask whether a newer YOLO version alone is
better. It asks whether a parameter-free frequency-angular input frontend can
strengthen fine-grained classification inside a single end-to-end YOLO26
detector without adding a crop-based second stage.

## Main research question

> Does adapting the AFAB-2 frequency-angular mechanism as an end-to-end input
> frontend for YOLO26 improve fine-grained SNI-21 coffee-defect detection over
> a seed- and optimization-matched detector, and what mechanism, generalization
> boundary, and computational cost characterize the improvement?

## Sub-questions

1. Does AF2 improve Macro, Bottom-3, and Worst-class mAP50-95 consistently
   across seeds relative to D0FT?
2. Is the observed benefit attributable to raw proposal/localization access or
   classification and candidate ranking after localization?
3. Does the direction persist on a leakage-safe external coffee dataset
   without target-domain training?
4. What illumination, dense-scene, per-class, and in-domain-test limitations
   remain?
5. What parameter, latency, throughput, and CUDA-memory trade-off accompanies
   AF2?

## Contribution structure

### Primary contribution

An end-to-end YOLO26 detector with a parameter-free AF2 frequency-angular
frontend that preserves the raw image through residual gating and requires no
ROI Align, decoded-box dependency, proposal reranking module, or second
classifier.

### Empirical contributions

1. A controlled three-seed comparison against seed-matched D0FT showing
   +1.32-point Macro, +2.80-point Bottom-3, and +5.10-point Worst-class mean
   gains.
2. A conditional error decomposition showing no raw-proposal gain but an
   +8.12-point localization-conditioned Top-1 gain and +14.64-point
   correct-decision-recall gain.
3. A target-free external evaluation showing +4.08-point mean Macro on the
   leakage-safe 18-class Coffee Standard mapping, improved in 3/3 seeds.
4. A same-device deployment audit showing identical parameters but 1.745x
   median latency and 1.696x peak allocated CUDA memory.
5. Controlled negative ablations showing that radial separation, orientation
   folding, adaptive illumination gating, channel calibration, and extra
   DG/FG objectives do not supersede original AF2 under their frozen gates.

### Methodological contribution

A reproducible evaluation design combining parent-identity grouping,
optimization-matched controls, prospective gates, lower-tail metrics,
conditional localization/classification diagnosis, target-free external
evaluation, and same-runtime efficiency measurement.

This methodological package is supporting novelty. It should not replace the
AF2-YOLO26 adaptation as the architectural center of the thesis.

## Model definition

Let normalized RGB input be `x`. AF2 divides each channel into overlapping
32 x 32 patches, computes a two-dimensional FFT, estimates angular magnitude
density, applies an entropy-conditioned hard amplitude selection with
`gamma=0.1`, reconstructs the selected signal by inverse FFT, and forms the
raw-preserving input

```text
x_AF2 = x + x * minmax(recover_AF2(x)).
```

`x_AF2` is passed directly to the ordinary YOLO26n P3--P5 detector. The
frontend adds no trainable parameters. Training remains single-stage and
end-to-end in the operational sense: one input, one detector graph, one loss
system, and one detection output. The fixed FFT operator itself is not learned.

## Why this is a model modification

AF2 is not merely an offline image-processing recipe. It is instantiated in
the detector graph, executed during both training and inference, preserves
gradient flow to the downstream detector, is serialized through the custom
YOLO model class, and changes the tensor received by the backbone. It is
therefore an **input-frontend architectural modification** of YOLO26.

The boundary must remain explicit: AF2 does not change the trainable YOLO26
backbone, neck, or head, and it adds no learned weights. If a university rubric
requires a newly invented trainable internal block, this design would not meet
that narrower requirement. Under the ordinary research definition of
architecture adaptation, however, the integrated frontend plus controlled
evidence is a valid model-level contribution.

## Experimental hierarchy for the thesis

### Main table

Report only the causal comparison required for the principal claim:

| Arm | Role |
|---|---|
| D0FT | Seed- and optimization-matched YOLO26 control |
| AF2 | Proposed frequency-angular frontend adaptation |

Use seeds 42, 123, and 2026 and report mean, standard deviation, paired delta,
minimum delta, and improved-seed count for Macro, Bottom-3, and Worst-class
mAP50-95.

### Supporting tables

1. Original D0 versus D0FT, to show why optimization matching is necessary.
2. AF1/AF2/AF12 seed-42 factorization, to justify selecting AF2 rather than the
   complete transferred bundle.
3. AF2 versus IGEM1 as independently confirmed mechanisms, explicitly marked
   descriptive because no direct superiority test was frozen.
4. Mechanism diagnostic: raw accessibility, final accessibility, conditional
   Top-1, wrong-class rate, miss rate, and correct-decision recall.
5. Coffee Standard external paired results.
6. Efficiency trade-off.
7. Negative AF2 extensions and illumination stress test.

Do not place every screened architecture in the primary result table. The
full breadth belongs in an appendix or screening-flow figure.

## Recommended thesis chapters

### Chapter 1 — Introduction

- SNI-scale physical defects and contaminants as a fine-grained detection
  problem.
- Why global mAP and box count alone hide difficult classes.
- Research gap: coffee architecture papers rarely isolate proposal access from
  conditional classification using grouped identities and lower-tail metrics.
- Research questions, objectives, scope, contributions, and non-claims.

### Chapter 2 — Literature review

- SNI taxonomy and fine-grained confusion: Bahy and Rifai; Jundullah et al.
- Strong YOLO coffee baselines: Gope et al.; Chen and Widiyanto.
- Coffee architecture modifications: Hong et al.; Ji et al.; KN-YOLOv8.
- Classification/localization decomposition: Lei et al.
- Frequency-domain feature enhancement and LFDet/AFAB.
- Dense-scene and postprocessing work as adjacent, separate problems.
- Close with the gap already frozen in `RELATED_WORK_POSITIONING.md`.

### Chapter 3 — Methodology

- Faruq-v3 mask correction and parent-grouped development split.
- Leakage controls and SNI-21 class contract.
- YOLO26 D0 and optimization-matched D0FT controls.
- AF2 equations, implementation choices, and residual raw path.
- Training schedule and paired seeds.
- Macro, Bottom-3, Worst-class, and per-class metrics.
- Proposal-versus-classification diagnostic.
- Coffee Standard ontology mapping and parent deduplication.
- Illumination/synthetic stress tests and same-device efficiency protocol.
- Prospective decision gates and test-lock policy.

### Chapter 4 — Results and discussion

1. Dataset validity and baseline bottleneck.
2. Candidate screening and why AF2 is selected.
3. Three-seed D0FT--AF2 confirmation.
4. Classification-dominant mechanism attribution.
5. External cross-dataset result and severe absolute domain gap.
6. Illumination, synthetic density, and per-class limitations.
7. Accuracy--efficiency trade-off.
8. Negative AF2 extensions and why the simpler operator is retained.
9. Comparison with related work without cross-dataset SOTA claims.

### Chapter 5 — Conclusion

- Answer each sub-question directly.
- State AF2's validated benefit and computational cost together.
- State that general illumination robustness, real dense 300-gram operation,
  and untouched in-domain test confirmation remain future work.

## Recommended headline result paragraph

> Across three paired seeds on the parent-grouped Faruq-v3 development split,
> AF2 increased Macro mAP50-95 from 86.62% to 87.94%, Bottom-3 from 76.58% to
> 79.37%, and Worst-class performance from 73.05% to 78.15% relative to the
> optimization-matched D0FT control. Raw top-500 proposal accessibility did not
> improve, whereas localization-conditioned Top-1 accuracy increased by 8.12
> points, indicating a classification/ranking-dominant effect. Without target
> training, AF2 also improved Coffee Standard Macro by 4.08 points across all
> three seeds. The frontend added no detector parameters, but increased median
> T4 tensor-forward latency from 13.52 to 23.59 ms and peak allocated memory
> from 75.2 to 127.6 MB.

## Novelty wording

Use:

> The novelty lies in the controlled adaptation and validation of a
> parameter-free frequency-angular frontend within an end-to-end YOLO26
> detector for a 21-class SNI fine-grained taxonomy, together with explicit
> mechanism, external-domain, lower-tail, and efficiency evidence.

Do not use:

> A completely new frequency transform or universally robust coffee detector.

## Final scope decision

The thesis is ready to move from model search to writing. Original AF2 is the
fixed proposed model, D0FT is the fixed primary control, and no further model
tuning is justified by the completed evidence. Any future real-dense dataset,
independent in-domain test, coffee-variety evaluation, or deployment pipeline
must be presented as future work rather than silently added to the current
claim set.

Evidence authority:
`docs/FARUQ_V3_AF2_THESIS_EVIDENCE_MATRIX_2026-08-21.md`.
