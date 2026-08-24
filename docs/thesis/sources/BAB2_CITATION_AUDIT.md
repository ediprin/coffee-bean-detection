# Bab II Citation Audit

Purpose: track whether the current Bab II draft has enough reference breadth, whether the same empirical papers are being recycled, and whether every citation key resolves unambiguously.

Audit target: `docs/thesis/proposal/04_LITERATURE_REVIEW.md`.

Status: **CITATION-READY REWRITE IN PROGRESS**. Sections 2.1–2.5 have passed their first source-grounded rewrite; Sections 2.6–2.9 still require source-level promotion.

## Current draft audit

| Section | Current explicit keys / state | Current count | Target | Status | Main action |
|---|---|---:|---:|---|---|
| 2.1 Coffee beans / physical defects | STD-01, COF-07, COF-08, COF-02 | 4 | 5–8 planning target | FUNCTIONALLY ADEQUATE | Official standard + three independent operational taxonomies now present. Add another source only if it supports a needed sentence; do not pad citations. |
| 2.2 Conventional inspection | COF-07, COF-08, COF-14, COF-10, REV-01 | 5 | 4–7 | PASS FIRST REWRITE | Manual-inspection evidence, historical CV transition, review landscape, and modern edge/deep-learning example are separated. |
| 2.3 Object detection | DET-02, DET-03, DIAG-01, DIAG-02, DIAG-03 | 5 | 4–6 | PASS FIRST REWRITE | Two-stage/one-stage foundations plus classification-localization diagnosis are now separated from coffee evidence. |
| 2.4 YOLO | DET-03, COF-06, COF-01, COF-02 | 4 distinct | 5–8 planning target | FUNCTIONALLY ADEQUATE | Original YOLO is now the foundation; three coffee studies provide small-, medium-, and larger-taxonomy context without encyclopedia-style version history. |
| 2.5 YOLO26 | DET-01 | 1 primary + experiment protocol boundary | 2–4 planning target | FUNCTIONALLY ADEQUATE | Primary YOLO26 preprint now grounds architecture/status. Add implementation/repo source only when Bab III protocol is described; do not pad Bab II with duplicate docs. |
| 2.6 Fine-grained detection | COF-07, COF-02, COF-04, COF-05, COF-13 | 5 | 7–10 | UNDER | Add `FG-02`, `FG-03`, `FG-01`, plus rotate coffee sources `COF-03/08/12/13`. |
| 2.7 Preprocessing | PRE-04, PRE-05, PRE-01, PRE-02 | 4 | 7–10 | UNDER | Add PRE-06, PRE-07, PRE-08 and use PRE-03 as transition to frequency section. |
| 2.8.1 DFT/FFT | Equations but no citation key | 0 | 2–3 | FAIL | Add `THEORY-01`, optional `THEORY-02`. |
| 2.8.2 amplitude/phase | PRE-03 only | 1 | 2–4 | UNDER | Add `THEORY-01`, `THEORY-02`, optional `PRE-08`. |
| 2.8.3 radial/angular | SPEC-01, SPEC-02 | 2 | 2–4 | KEY CONFLICT RESOLVED | Canonical spectral keys are used; verify exact equation/page support during rewrite. |
| 2.8.4 frequency + detection | PRE-03, AGR-01, AGR-02, WAVE-01, FREQ-03, FG-01 | 6 | 6–10 | BREADTH OK / PROSE PENDING | Rewrite to distinguish input preprocessing from internal feature-space frequency processing. |
| 2.9 Related work | Working table ~10 studies | 10 | 12–18 | UNDER | Promote the verified balanced 14-study shortlist + proposed-research row. |

## Detector-source grounding now closed for first rewrite

### §2.3

The active draft now distinguishes:

- `DET-02` Faster R-CNN — canonical two-stage/RPN context;
- `DET-03` original YOLO — direct regression / unified one-stage context;
- `DIAG-01` TOOD — classification/localization task misalignment;
- `DIAG-02` Wu et al. — different representation/head preferences for classification and localization;
- `DIAG-03` IoU-Net — classification confidence is distinct from localization confidence.

This section no longer relies on coffee papers to define generic object detection.

### §2.4

The original YOLO paper is the conceptual source. Coffee studies are used only to show domain adoption and changing taxonomy difficulty:

- `COF-06` — compact four-class green-coffee benchmark;
- `COF-01` — modern coffee YOLO improvement with subtle-class motivation;
- `COF-02` — 20-category SNI-oriented detector with classwise heterogeneity.

Hebert and Jundullah are deliberately not used here so they remain available as stronger fine-grained evidence in §2.6.

### §2.5

`DET-01` is explicitly identified as a 2026 arXiv preprint, not a journal/Q1 source. The draft currently uses it for the paper-level claims on:

- dual-head native end-to-end / NMS-free inference design;
- DFL removal;
- MuSGD;
- Progressive Loss;
- STAL;
- backbone–neck–P3/P4/P5 detect-head architecture.

The thesis-specific claim that AF2 sits before the detector is a methodological boundary of this project, not a claim attributed to the YOLO26 paper.

## Critical citation-key issue — resolved in active draft

The structural draft and the earlier `METHOD_BRIDGE_MATRIX.md` previously had a citation-key collision:

```text
old draft meaning:
FREQ-01 = Cao et al. 2019
FREQ-02 = Zhang & Tan 2003
FREQ-03 = WTConv
FREQ-04 = FDConv

latest master-map meaning:
FREQ-01 = Fast Fourier Convolution
FREQ-02 = FDADNet
FREQ-03 = FDConv
```

The canonical namespace is:

```text
SPEC-01 = Cao et al. 2019
SPEC-02 = Zhang & Tan 2003
WAVE-01 = WTConv
FREQ-01 = Fast Fourier Convolution
FREQ-02 = FDADNet
FREQ-03 = FDConv
```

`04_LITERATURE_REVIEW.md` uses `SPEC-01/SPEC-02` in §2.8.3 and `WAVE-01/FREQ-03` in §2.8.4. The deprecated aliases must not be reintroduced.

## Paper-reuse status

The source-normalized rewrite has reduced recycling:

- §2.2 no longer uses Hong/Bahy/Hebert/Jundullah as a generic detector bundle;
- §2.3 contains only detector foundations and diagnostic papers;
- §2.4 uses Hong/Bahy for narrowly defined YOLO-domain roles;
- Hebert/Jundullah remain reserved for §2.6 fine-grained difficult-class evidence.

Remaining routing policy:

- Hong: primary `2.4`, table `2.9`, optional one sentence in `2.6` only if necessary.
- Bahy: taxonomy/large-class YOLO context (`2.1` or `2.4`), table `2.9`; fine-grained use only for a specific classwise claim.
- Hebert: primary `2.6` difficult subtle classes + table `2.9`.
- Jundullah: primary `2.6` 20-class detection + table `2.9`; avoid substantive reuse in `2.4`.
- Kesiman: `2.1` taxonomy + `2.6` granularity-difficulty evidence + table; its uses support different claims.
- Muchtar: `2.2` manual-to-edge automation transition; optional efficiency context in Bab III.

## Diversity gates before marking Bab II fully citation-ready

Bab II cannot move to **FULL CITATION-READY** until all of the following hold:

- [x] §2.1 has at least one official standard source and at least two independent taxonomy/coffee sources.
- [x] §2.3 cites foundational detector literature, not coffee papers for basic definitions.
- [x] §2.4 cites the original YOLO source.
- [x] §2.5 cites the YOLO26 primary source and clearly labels its preprint status.
- [ ] §2.6 contains at least two general fine-grained/FGOD papers plus at least four independent coffee studies.
- [ ] §2.7 contains fixed, learned task-driven, transform-domain, and agricultural preprocessing examples.
- [ ] §2.8.1–2.8.2 cite canonical Fourier/image-processing foundations.
- [x] §2.8.3 uses `SPEC-01/SPEC-02`, not the deprecated FREQ aliases.
- [ ] §2.8.4 clearly distinguishes input preprocessing from internal feature-space frequency processing in final prose.
- [ ] Table 2.1 contains 12–18 evidence-diverse studies plus `Penelitian yang Diusulkan`.
- [ ] No non-foundational empirical paper is a substantive source in 3+ theory subsections.
- [ ] Every numerical claim has been re-opened in the primary full text.
- [ ] Every index/quartile entry has been checked against the master index audit rather than inferred from publication type.

## Planned post-rewrite audit metrics

When `04_LITERATURE_REVIEW.md` is fully rewritten, record:

```text
unique_sources_total        = TBD
unique_sources_by_section   = TBD
max_empirical_section_reuse = TBD
uncited_technical_paragraph = TBD
unresolved_keys             = TBD
unverified_numeric_claims   = TBD
unverified_index_claims     = TBD
```

Desired end state:

```text
unique_sources_total        >= 35 (planning target, not quota)
max_empirical_section_reuse <= 2 substantive theory sections
uncited_technical_paragraph = 0
unresolved_keys             = 0
unverified_numeric_claims   = 0
unverified_index_claims     = 0
```

The related-work table does not count as an additional theory-section reuse for this gate; it is intentionally a synthesis table.