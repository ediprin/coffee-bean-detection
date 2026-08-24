# Proposal Cross-Chapter Consistency Audit

Date: 2026-08-25

Scope: Bab I–III proposal foundation after Bab II normalization and first Bab III methodology draft.

Status: **CORE LOGIC CONSISTENT; several final-submission gates remain open**.

## 1. Core problem-method chain

Current chapters agree on the following logic:

```text
coffee literature
-> detailed / visually similar defect classes can be difficult
-> aggregate metrics can hide tail classes
-> need discriminative representation / input utility

non-coffee preprocessing + spectral literature
-> image-space transformation can affect downstream detection
-> frequency/angular processing is technically plausible

research hypothesis
-> test parameter-free AF2 before YOLO26

experimental design
-> same detector / source / split / schedule
-> native vs AF2 input treatment
-> aggregate + tail + diagnostic + efficiency evaluation
```

PASS: no chapter currently requires the unsupported causal claim `coffee difficulty -> frequency bottleneck`.

## 2. Research-question alignment

### RQ1 — overall effectiveness

Question:

```text
Does frequency-angular preprocessing improve YOLO26 fine-grained coffee-defect detection?
```

Matched methodology:

- D0DIRECT vs AF2DIRECT;
- same official `yolo26n.pt` source;
- matched target-head initialization;
- same training schedule;
- Macro mAP50–95 + mAP50 / per-class AP.

Status: **ALIGNED**.

### RQ2 — difficult / lower-tail classes

Question:

```text
What is the effect on low-performing / difficult classes?
```

Matched methodology:

- per-class AP;
- Bottom-3 mAP50–95;
- Worst-class mAP50–95;
- per-seed class behavior.

Status: **ALIGNED**.

### RQ3 — discrimination vs proposal-accessibility pattern

Current normalized wording:

```text
Is the observed performance pattern more consistent with improved class discrimination
than improved raw proposal/localization accessibility?
```

Matched diagnostics:

- raw top-500 proposal accessibility;
- localization-conditioned Top-1;
- correct-decision recall.

Status: **ALIGNED WITH DIAGNOSTIC BOUNDARY**.

Important: raw proposal accessibility is not a complete box-regression/localization-quality metric. The proposal must not silently return to the older wording that implies full localization causality.

## 3. Objective alignment

| Objective | Method / output | Status |
|---|---|---|
| Evaluate AF2 effectiveness | paired D0DIRECT/AF2DIRECT | PASS |
| Analyze lower tail | Bottom-3/Worst/per-class | PASS |
| Diagnose class-vs-proposal pattern | mechanism diagnostics | PASS |
| Evaluate efficiency | params/latency/throughput/VRAM | METHOD DEFINED, FINAL ENVIRONMENT OPEN |

Efficiency does not need its own main RQ unless the campus/advisor requires one; it currently functions as a secondary engineering objective.

## 4. AF2 terminology audit

### `parameter-free`

PASS. Current `operator.py` contains no trainable AF2 parameters.

### `frequency-angular`

PASS with a precise meaning:

- `frequency` = local patch Fourier representation;
- `angular` = amplitude density grouped by Fourier direction;
- it does **not** mean bounding-box angle / oriented detection.

### radial terminology

Important correction already locked:

```text
radius_ratio=0.05 is present in shared AFAB config
but is inactive in pure mode=af2.
```

Therefore the thesis must not describe AF2 as simultaneously applying AF1 radial high-pass filtering unless a different mode is explicitly tested.

### `content-adaptive`

PASS. Threshold depends on patch/channel angular entropy even though no weights are learned.

### `lightweight`

Do not use as a default AF2 label. Parameter-free does not imply low latency or low memory.

## 5. Dataset consistency audit

Current methodology and direct protocol agree:

```text
train      = 1,665 images / 2,986 annotations
validation =   294 images /   526 annotations
classes    = 21
```

PASS: Bab I/II use SNI only as standards/taxonomy context and do not claim the detector reconstructs the full SNI defect-value grading procedure.

PASS: locked test is not used for method selection in the proposal/pilot.

## 6. Training-protocol consistency audit

Direct thesis protocol:

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

Conflict found in older `configs/D0_yolo26n.yaml`:

```text
epochs   100
workers  4
patience 20
```

Resolution: **direct-from-pretrained frozen protocol wins** for Bab III.

## 7. Pilot-result consistency audit

Proposal narrative correctly treats seed 42 as feasibility evidence.

However, provenance is split:

- machine-capture result record contains D0DIRECT metrics + Boolean promotion result, but lacks exact AF2DIRECT numeric fields;
- `foundation/04_PILOT_EVIDENCE.md` contains user-verified exact AF2DIRECT metrics and paired deltas.

Current Bab I uses the user-verified pilot numbers. This is acceptable for a preliminary proposal draft if labelled as preliminary, but before final thesis/result-table archival the exact AF2DIRECT values should be imported from the saved run artifact into the machine evidence record.

Do not describe the machine JSON as containing those exact AF2 numbers until that reconciliation occurs.

## 8. Literature-routing consistency audit

Bab II now has a normalized modular assembly:

- `04_02_INSPECTION_QUALITY_NORMALIZED.md` replaces the older embedded §2.2;
- `04_09_RELATED_WORK_TABLE.md` replaces the older embedded §2.9 table.

`COF-17` García 2019 now handles manual/classical coffee inspection evidence.

Result:

```text
COF-07 Kesiman -> §2.1 taxonomy + §2.6 fine-grained difficulty
COF-08 Arwatchananukul -> §2.1 taxonomy + §2.6 unseen/fine-grained evidence
```

This closes the most obvious citation-recycling hotspot.

## 9. Citation-key consistency audit

Background has been normalized from deprecated spectral aliases to:

```text
SPEC-01 = Cao et al. radial/angular spectral analysis
SPEC-02 = Zhang & Tan orientation spectrum
```

Do not reintroduce the old collision where `FREQ-01/FREQ-02` referred to those papers.

## 10. Working-title risk

Current title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

`Analisis` is clearly supported by the design.

`Optimasi` is **not yet automatically guaranteed by the current confirmatory design**, because the thesis method currently freezes one retained AF2 operator/configuration and compares it against native YOLO26.

Two defensible routes remain:

### Route A — retain `Analisis dan Optimasi`

The final methodology must include a clearly scoped optimization/ablation question, e.g. controlled operator-factor selection or retained parameter/configuration analysis, without reopening uncontrolled module stacking.

### Route B — use a safer title if final work remains comparison-only

Example direction:

`Analisis Pengaruh Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi`.

No title change is made automatically; this is an advisor/proposal decision.

## 11. Open gates before formatted proposal export

- [x] core problem-method chain consistent;
- [x] RQ1–RQ3 map to metrics/experiments;
- [x] AF2 implementation and config audited;
- [x] Bab II citation recycling normalized;
- [x] direct protocol conflict resolved;
- [ ] decide whether title retains `Optimasi` and what experiment operationalizes it;
- [ ] pair DFT/FFT fundamentals with final textbook edition/page references;
- [ ] reconcile AF2DIRECT exact numeric artifact provenance;
- [ ] freeze final locked-test procedure before any test access;
- [ ] record actual confirmatory hardware/runtime for efficiency comparison;
- [ ] run final citation/index/numeric recertification during DOCX generation;
- [ ] create the final research-flow figure from the synchronized Bab III design.

## 12. Source-of-truth map for future generation

```text
Thesis logic
  -> foundation/00-07

Bab I
  -> proposal/02_BACKGROUND.md
  -> proposal/03_PROBLEM_FORMULATION.md

Bab II
  -> proposal/04_LITERATURE_REVIEW.md
  -> proposal/04_02_INSPECTION_QUALITY_NORMALIZED.md
  -> proposal/04_09_RELATED_WORK_TABLE.md

Bab III
  -> proposal/05_METHODOLOGY.md

Citation namespace
  -> sources/CANONICAL_SOURCE_KEYS.md

Bab II QA
  -> sources/BAB2_CITATION_AUDIT.md
  -> sources/BAB2_NORMALIZATION_AUDIT_2026-08-25.md

Bab III QA
  -> sources/BAB3_PROTOCOL_AUDIT.md

Cross-chapter QA
  -> sources/PROPOSAL_CROSS_CHAPTER_AUDIT.md
```
