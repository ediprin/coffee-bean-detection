# Bab II Citation Audit

Purpose: track whether the current Bab II draft has enough reference breadth, whether the same empirical papers are being recycled, and whether every citation key resolves unambiguously.

Audit target: `docs/thesis/proposal/04_LITERATURE_REVIEW.md`.

Status: **CITATION-READY REWRITE IN PROGRESS**. Sections 2.1–2.2 have passed their first source-routing rewrite; Sections 2.3–2.9 still require source-level promotion.

## Current draft audit

| Section | Current explicit keys / state | Current count | Target | Status | Main action |
|---|---|---:|---:|---|---|
| 2.1 Coffee beans / physical defects | STD-01, COF-07, COF-08, COF-02 | 4 | 5–8 planning target | FUNCTIONALLY ADEQUATE | Official standard + three independent operational taxonomies now present. Add another source only if it supports a needed sentence; do not pad citations. |
| 2.2 Conventional inspection | COF-07, COF-08, COF-14, COF-10, REV-01 | 5 | 4–7 | PASS FIRST REWRITE | Manual-inspection evidence, historical CV transition, review landscape, and modern edge/deep-learning example are now separated. |
| 2.3 Object detection | DIAG-01, DIAG-02, DIAG-03 | 3 | 4–6 | UNDER | Add `DET-02`, `DET-03`, `EVAL-01`; coffee-specific `COF-09` only as optional bridge. |
| 2.4 YOLO | COF-01, COF-02, COF-04, COF-05, COF-06 | 5 | 5–8 | COUNT OK, FOUNDATION MISSING | Add `DET-03` and use selected coffee applications; avoid using Hebert/Jundullah heavily here if reserved for §2.6. |
| 2.5 YOLO26 | Placeholder only | 0 | 2–4 | FAIL | Add `DET-01`, with `EVAL-01/EVAL-02` only where metrics are discussed; repo protocol stays implementation evidence. |
| 2.6 Fine-grained detection | COF-07, COF-02, COF-04, COF-05, COF-13 | 5 | 7–10 | UNDER | Add `FG-02`, `FG-03`, `FG-01`, plus rotate coffee sources `COF-03/08/12/13`. |
| 2.7 Preprocessing | PRE-04, PRE-05, PRE-01, PRE-02 | 4 | 7–10 | UNDER | Add PRE-06, PRE-07, PRE-08 and use PRE-03 as transition to frequency section. |
| 2.8.1 DFT/FFT | Equations but no citation key | 0 | 2–3 | FAIL | Add `THEORY-01`, optional `THEORY-02`. |
| 2.8.2 amplitude/phase | PRE-03 only | 1 | 2–4 | UNDER | Add `THEORY-01`, `THEORY-02`, optional `PRE-08`. |
| 2.8.3 radial/angular | SPEC-01, SPEC-02 | 2 | 2–4 | KEY CONFLICT RESOLVED | Canonical spectral keys are now used in the prose; verify exact equation/page support during rewrite. |
| 2.8.4 frequency + detection | PRE-03, AGR-01, AGR-02, WAVE-01, FREQ-03, FG-01 | 6 | 6–10 | BREADTH OK / PROSE PENDING | Key normalization is complete. Rewrite to distinguish input preprocessing from internal feature-space frequency processing. |
| 2.9 Related work | Working table ~10 rows | 10 | 12–18 | UNDER | Promote the verified balanced 14-row shortlist + proposed-research row. |

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

`04_LITERATURE_REVIEW.md` now uses `SPEC-01/SPEC-02` in §2.8.3 and `WAVE-01/FREQ-03` in §2.8.4. The deprecated aliases must not be reintroduced.

## Paper-reuse status after first rewrite

The first rewrite deliberately removed the previous detector-paper cluster from §2.2. Conventional-inspection discussion now uses `COF-07`, `COF-08`, `COF-14`, `COF-10`, and `REV-01`, so Hong/Bahy/Hebert/Jundullah are no longer recycled there as a generic citation bundle.

Remaining routing policy:

- Hong: primary `2.4`, table `2.9`, optional one sentence in `2.6` only if necessary.
- Bahy: taxonomy/large-class YOLO context (`2.1` or `2.4`), table `2.9`; fine-grained use only for a specific classwise claim.
- Hebert: primary `2.6` difficult subtle classes + table `2.9`.
- Jundullah: primary `2.6` 20-class detection + table `2.9`; brief landscape mention in `2.4` at most.
- Kesiman: `2.1` taxonomy + `2.6` granularity-difficulty evidence + table; its two uses support different claims.
- Muchtar: `2.2` manual-to-edge automation transition; optional efficiency context in Bab III.

## Diversity gates before marking Bab II fully citation-ready

Bab II cannot move to **FULL CITATION-READY** until all of the following hold:

- [x] §2.1 has at least one official standard source and at least two independent taxonomy/coffee sources.
- [ ] §2.3 cites foundational detector literature, not coffee papers for basic definitions.
- [ ] §2.4 cites the original YOLO source.
- [ ] §2.5 cites the YOLO26 primary source and clearly labels its publication status.
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