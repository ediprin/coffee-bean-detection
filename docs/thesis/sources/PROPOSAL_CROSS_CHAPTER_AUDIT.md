# Proposal Cross-Chapter Consistency Audit

Date: 2026-08-25

Scope: Bab I–III after Hong-adapted AF2 optimization rewrite.

Status: **CORE LOGIC SYNCHRONIZED; final citation/artifact/format gates remain open**.

## 1. Core problem-method chain

Current chapters now agree on:

```text
coffee literature
-> detailed / visually similar defect classes can be difficult
-> aggregate metrics can hide tail classes
-> need stronger discriminative input / representation

non-coffee preprocessing + spectral literature
-> image-space transformation can affect downstream detection
-> frequency/angular processing is technically plausible

research design
-> AF2 contains multiple design choices
-> factorized AF2 analysis / limited sensitivity
-> select AF2*
-> method freeze
-> matched native YOLO26 vs AF2*-YOLO26 confirmation
-> aggregate + tail + mechanism + visualization/error + efficiency analysis
```

PASS: no chapter requires the unsupported causal claim `coffee difficulty -> frequency bottleneck`.

## 2. Working-title alignment

Current title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

The two key title terms are now operational:

```text
Optimasi
= factorized AF2 structural analysis
+ limited parameter sensitivity
+ method freeze

Analisis
= overall performance
+ lower-tail behavior
+ mechanism diagnostics
+ visualization
+ error analysis
+ efficiency
```

Status: **TITLE-METHOD ALIGNMENT PASS**.

The previous risk that `Optimasi` was decorative has been closed at methodology-design level.

## 3. Research-question alignment

### RQ1 — AF2 optimization

Question:

```text
How do AF2 design factors affect performance,
and which configuration is most defensible for selection?
```

Matched methodology:

- AF2C reference;
- AF2WIN;
- AF2ORI;
- AF2POL;
- AF2SOFT;
- AF2LUM;
- one-factor-at-a-time screening;
- Macro mAP50-95 primary selection;
- Bottom-3 / Worst-class constraints;
- latency as engineering trade-off;
- optional limited parameter sensitivity;
- method freeze before confirmatory evaluation.

Status: **ALIGNED**.

### RQ2 — confirmatory effectiveness

Question:

```text
Does selected AF2 improve matched YOLO26 detection?
```

Matched methodology:

- native YOLO26n vs selected AF2*-YOLO26n;
- same official `yolo26n.pt` source;
- matched 21-class head initialization;
- same grouped split and schedule;
- paired seeds 42 / 123 / 2026;
- Macro mAP50-95 primary.

Status: **ALIGNED**.

### RQ3 — lower-tail / difficult classes

Matched methodology:

- Bottom-3;
- Worst-class;
- per-class AP;
- confusion/error analysis;
- rescue-regression transition analysis.

Status: **ALIGNED**.

### RQ4 — discrimination vs proposal-accessibility pattern

Matched diagnostics:

- raw top-500 proposal accessibility;
- localization-conditioned Top-1;
- correct-decision recall;
- input/spectral visualization;
- activation visualization only if YOLO26 compatibility is verified.

Status: **ALIGNED WITH DIAGNOSTIC BOUNDARY**.

Raw proposal accessibility is not complete box-regression/localization quality.

## 4. Objective alignment

| Objective | Method / output | Status |
|---|---|---|
| Optimize AF2 design | factorized structural analysis + limited sensitivity | PASS |
| Confirm selected AF2 | native vs AF2* paired direct protocol | PASS |
| Analyze lower tail | Bottom-3/Worst/per-class/errors | PASS |
| Diagnose mechanism pattern | proposal + class diagnostics + visualization | PASS |
| Evaluate efficiency | params/latency/throughput/VRAM | METHOD DEFINED; FINAL ENVIRONMENT OPEN |

## 5. AF2 terminology audit

### `parameter-free`

PASS. AF2 has no learned preprocessing parameters.

### `frequency-angular`

PASS with precise meaning:

- `frequency` = local patch Fourier representation;
- `angular` = amplitude density grouped by Fourier direction;
- not bounding-box angle or oriented detection.

### radial terminology

`radius_ratio=0.05` exists in shared config but is inactive in pure `mode=af2`. Do not describe AF2 reference as simultaneously applying AF1 radial high-pass filtering.

### `content-adaptive`

PASS. Threshold depends on patch/channel entropy.

### `lightweight`

Do not use as default AF2 label. Parameter-free does not imply low latency or memory.

## 6. Optimization-provenance audit

Two evidence layers are intentionally separated.

### Historical factorization genealogy

Uses seed-matched D0 coffee checkpoint for candidate screening.

Role:

```text
development / structural-selection evidence
```

### Final confirmatory thesis protocol

Uses official YOLO26n pretrained source directly with matched target head.

Role:

```text
main native-vs-AF2 confirmatory evidence
```

PASS: proposal Bab III now states this mismatch explicitly rather than silently merging the two protocols.

## 7. Dataset consistency audit

```text
train      = 1,665 images / 2,986 annotations
validation =   294 images /   526 annotations
classes    = 21
```

PASS:

- grouped split retained;
- parent/hash leakage gates retained;
- test not used for selection;
- SNI remains standards/taxonomy context rather than full grading reconstruction.

## 8. Training-protocol consistency audit

Final direct confirmatory authority:

```text
max epochs   50
imgsz        640
batch        16
workers      2
patience     15
optimizer    auto
close_mosaic 10
max_det      500
deterministic true
```

Older conflicting 100-epoch D0 config is not authoritative for final direct confirmation.

## 9. Visualization/error-analysis audit

Current Bab III now includes:

```text
Original RGB
-> local patch
-> FFT magnitude
-> angular density
-> adaptive threshold
-> retained angular response
-> reconstructed cue
-> AF2-enhanced RGB
-> prediction / activation visualization
```

Guardrails:

- visualization is qualitative support, not causal proof;
- CAM/EigenCAM is not frozen until YOLO26 compatibility is verified;
- qualitative examples should use deterministic/fixed-seed sampling from predefined outcome groups;
- rescue/regression statistic is thesis-defined, not attributed to Hong or COCO.

Status: **METHOD DEFINED; IMPLEMENTATION AUDIT OPEN**.

## 10. Pilot-result provenance

Seed-42 direct result remains feasibility evidence only.

Exact AF2DIRECT numeric provenance still requires reconciliation into the machine evidence record before final result-table archival.

Do not describe seed 42 as final superiority evidence.

## 11. Bab II / citation-key consistency

PASS:

- normalized §2.2 and §2.9 modules retained;
- `SPEC-01/SPEC-02` remain canonical angular-theory keys;
- old FREQ key collision must not be reintroduced.

## 12. Open gates before formatted DOCX export

- [x] title-method alignment closed;
- [x] RQ1–RQ4 synchronized with Bab III;
- [x] factorized optimization design defined;
- [x] direct-vs-factorization provenance distinction explicit;
- [x] mechanism / visualization / error-analysis structure defined;
- [ ] exact page/equation citations for AFAB/LFDet-derived formulas;
- [ ] exact page/equation citations for YOLO26 details used in Bab III;
- [ ] decide/freeze exact parameter-sensitivity values if that stage is retained;
- [ ] verify YOLO26-compatible activation visualization;
- [ ] reconcile AF2DIRECT exact numeric artifact provenance;
- [ ] freeze final locked-test procedure before any test access;
- [ ] record confirmatory hardware/runtime for efficiency comparison;
- [ ] final APA citation conversion and 50-reference requirement audit;
- [ ] generate Bab III figures;
- [ ] assemble into official USU DOCX template and render visual QA.

## 13. Source-of-truth map

```text
Thesis logic
  -> foundation/00-09

Bab I
  -> proposal/02_BACKGROUND.md
  -> proposal/03_PROBLEM_FORMULATION.md

Bab II
  -> proposal/04_LITERATURE_REVIEW.md
  -> proposal/04_02_INSPECTION_QUALITY_NORMALIZED.md
  -> proposal/04_09_RELATED_WORK_TABLE.md

Bab III
  -> proposal/05_METHODOLOGY.md

Bab III design
  -> foundation/08_BAB3_HONG_ADAPTED_OPTIMIZATION_DESIGN.md

Document generation
  -> foundation/09_USU_DOCUMENT_GENERATION_CONTRACT.md

Citation namespace
  -> sources/CANONICAL_SOURCE_KEYS.md

Bab II QA
  -> sources/BAB2_CITATION_AUDIT.md
  -> sources/BAB2_NORMALIZATION_AUDIT_2026-08-25.md

Bab III QA
  -> sources/BAB3_PROTOCOL_AUDIT.md
  -> sources/BAB3_HONG_REWRITE_AUDIT_2026-08-25.md

Cross-chapter QA
  -> sources/PROPOSAL_CROSS_CHAPTER_AUDIT.md
```