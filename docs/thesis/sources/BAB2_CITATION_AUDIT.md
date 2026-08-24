# Bab II Citation Audit

Purpose: track reference breadth, citation recycling, source-key integrity, numerical verification, and promotion status for `docs/thesis/proposal/04_LITERATURE_REVIEW.md`.

Status: **CITATION-READY REWRITE IN PROGRESS**. Sections 2.1–2.8 have passed their first source-grounded rewrite. The remaining prose task is §2.9 `Penelitian Terkait`, followed by a whole-chapter reference/index audit. One foundational-source item also remains open: exact textbook edition/page verification for the DFT/FFT theory anchor.

## Current section status

| Section | Main source keys | Status | Remaining action |
|---|---|---|---|
| 2.1 Coffee beans / physical defects | STD-01, COF-07, COF-08, COF-02 | FUNCTIONALLY ADEQUATE | Do not add references only to meet a quota. |
| 2.2 Conventional inspection | COF-07, COF-08, COF-14, COF-10, REV-01 | PASS FIRST REWRITE | Final bibliography normalization only. |
| 2.3 Object detection | DET-02, DET-03, DIAG-01, DIAG-02, DIAG-03 | PASS FIRST REWRITE | Add evaluation source only where metric definition is needed. |
| 2.4 YOLO | DET-03, COF-06, COF-01, COF-02 | FUNCTIONALLY ADEQUATE | Preserve concise history; avoid detector-paper recycling. |
| 2.5 YOLO26 | DET-01 | FUNCTIONALLY ADEQUATE | Repository protocol belongs primarily to Bab III. |
| 2.6 Fine-grained detection | FG-03, FG-02, COF-07, COF-08, COF-05, COF-04, COF-03, COF-12, COF-13 | PASS FIRST REWRITE | Keep FG-01 reserved for frequency-method bridge. |
| 2.7 Preprocessing | PRE-04, PRE-05, PRE-06, PRE-01, PRE-02, PRE-07, PRE-03, PRE-08 | PASS FIRST REWRITE | No major source gap. |
| 2.8.1 DFT/FFT | PRE-08, PRE-03, FG-01 | PASS PRIMARY-PAPER REWRITE / TEXTBOOK AUDIT OPEN | Pair final thesis with exact `THEORY-01`/`THEORY-02` edition/page once available. |
| 2.8.2 amplitude/phase | PRE-03, PRE-08, FG-01 | PASS FIRST REWRITE | Do not overgeneralize FE-YOLO/FDA task-specific interpretations. |
| 2.8.3 radial/angular spectrum | SPEC-01, FG-01 | PASS FIRST REWRITE | SPEC-02 remains optional until its primary full text/page is re-opened. |
| 2.8.4 frequency-aware vision/detection | PRE-08, PRE-03, FG-01, FREQ-01, FREQ-02, FREQ-03, WAVE-01 | PASS FIRST REWRITE | Keep input-space vs feature-space mechanisms explicitly separated. |
| 2.9 Related work | Working table ~10 studies | UNDER | Replace with the verified 12–18-study balanced table + proposed-research row. |

## §2.8 source-role audit

The frequency section is deliberately built from different evidence layers rather than repeatedly citing the parent AFAB paper.

### 2.8.1 — transform foundation

- `PRE-08` FDA provides a primary-paper 2-D DFT formulation, FFT statement, amplitude/phase decomposition, inverse transform, and an example of non-learned Fourier input manipulation.
- `PRE-03` FE-YOLO independently provides complex Fourier coefficients and transform–process–reconstruct use before object detection.
- `FG-01` is used only to explain why its parent method uses **patch-wise**, rather than global, frequency responses for its own fine-grained remote-sensing setting.
- `THEORY-01/THEORY-02` remain bibliography-strengthening anchors, not an excuse to claim page verification that has not yet been done.

### 2.8.2 — amplitude and phase

- `PRE-03` directly supplies \(F=R+jI\), amplitude, and phase equations.
- `PRE-08` supplies an independent example where low-frequency amplitude is changed while source phase is retained.
- `FG-01` supplies the parent-method choice to remodel amplitude while preserving phase during patch reconstruction.

Safe conclusion: amplitude and phase can be treated differently by Fourier-based methods. Unsafe conclusion: amplitude or phase has been proven to be the coffee-defect bottleneck.

### 2.8.3 — radial and angular representation

- `SPEC-01` Cao et al. is the main independent theoretical bridge: spectrum energy is summarized through radial and angular distributions; radial behavior is used for frequency/periodicity/scale analysis and angular behavior for texture directionality.
- `FG-01` supplies the actual parent-method angular-density formulation and entropy-conditioned suppression used in AFAB-2.
- `SPEC-02` Zhang & Tan is intentionally not required for the active prose until its primary full text is re-opened. This avoids relying on an old project summary as if it were direct evidence.

The term **angular** is explicitly defined as direction in Fourier polar coordinates, not object/bounding-box rotation.

### 2.8.4 — location of frequency processing

The rewrite now distinguishes:

```text
INPUT / DATA SPACE
PRE-08 FDA       -> non-learned Fourier manipulation; segmentation/UDA
PRE-03 FE-YOLO   -> learned Fourier enhancement before YOLO
FG-01 AFAB       -> patch-wise data-space frequency processing for FGOD

FEATURE / NETWORK SPACE
FREQ-01 FFC      -> interconnected spatial/spectral feature processing
FREQ-02 FDADNet  -> spatial-frequency defect-detection representation
FREQ-03 FDConv   -> adaptive frequency modulation in dense prediction
WAVE-01 WTConv   -> wavelet feature-processing precedent
```

This separation prevents the false literature claim that all frequency-aware methods are equivalent forms of high-pass preprocessing.

## Parent-method boundary

`FG-01` Xu et al. is the closest parent source, but its scope is now constrained precisely:

- LFDet contains AFAB, CGFI, and FTIF;
- AFAB contains patch-wise DFT, AFAB-1 adaptive high-pass filtering, and AFAB-2 chaotic angular-amplitude suppression;
- AFAB-2 alone must not inherit the headline gain of full LFDet;
- the paper's aircraft-domain result does not validate coffee transfer;
- the thesis adaptation uses the local-frequency/angular principle as a standalone frontend for YOLO26, while coffee-specific residual reconstruction and integration choices belong to our Bab III implementation description.

## Citation-key integrity

The deprecated collision remains closed. Never restore:

```text
FREQ-01 = Cao 2019
FREQ-02 = Zhang & Tan 2003
FREQ-03 = WTConv
FREQ-04 = FDConv
```

Canonical namespace:

```text
SPEC-01 = Cao et al. 2019
SPEC-02 = Zhang & Tan 2003
WAVE-01 = WTConv
FREQ-01 = Fast Fourier Convolution
FREQ-02 = FDADNet
FREQ-03 = Frequency Dynamic Convolution
```

## Paper-reuse status

Source routing is now acceptably diversified:

- Hong is concentrated in §2.4 and the future §2.9 table;
- Hebert/Jundullah/Samudra are concentrated in §2.6;
- §2.7 uses a separate preprocessing corpus;
- §2.8 uses Fourier/spectral theory and method papers rather than recycling coffee papers;
- FE-YOLO legitimately appears in §2.7 and §2.8 because it plays two different roles: preprocessing precedent and Fourier-method comparator.

## Gates before `FULL CITATION-READY`

- [x] Official coffee standard + independent taxonomy sources.
- [x] Foundational object-detection literature.
- [x] Original YOLO source.
- [x] YOLO26 primary source with preprint status stated.
- [x] General FG/FGOD + multiple independent coffee difficult-class sources.
- [x] Fixed, task-driven, transform-domain, agricultural, and Fourier preprocessing precedents.
- [x] Primary-paper DFT/amplitude/phase equations available.
- [x] Radial/angular theory uses canonical `SPEC-*` namespace.
- [x] Input-space vs feature-space frequency methods separated.
- [ ] Final textbook edition/page audit for DFT/FFT foundation.
- [ ] §2.9 expanded to 12–18 evidence-diverse studies plus `Penelitian yang Diusulkan`.
- [ ] Whole-chapter unique-reference and overuse count completed.
- [ ] Every remaining numerical claim re-opened against primary full text.
- [ ] Every table index/quartile field normalized from the index-audited master map.

## Final audit metrics to record after §2.9

```text
unique_sources_total        = TBD
unique_sources_by_section   = TBD
max_empirical_section_reuse = TBD
uncited_technical_paragraph = TBD
unresolved_keys             = TBD
unverified_numeric_claims   = TBD
unverified_index_claims     = TBD
```

Planning target, not a quota:

```text
unique_sources_total        >= 35
max_empirical_section_reuse <= 2 substantive theory sections
uncited_technical_paragraph = 0
unresolved_keys             = 0
unverified_numeric_claims   = 0
unverified_index_claims     = 0
```
