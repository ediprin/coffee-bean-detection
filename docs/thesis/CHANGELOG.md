# Thesis Proposal Changelog

## v0.9.0 — 2026-08-24

Frequency-domain foundation promoted to first-pass source-grounded prose.

### Rewritten

- §2.8.1 now grounds the 2-D DFT/FFT and inverse-transform pipeline in primary Fourier-method papers, while keeping exact textbook edition/page verification open for the final bibliography;
- §2.8.2 now distinguishes complex coefficients, amplitude, and phase using FE-YOLO/FDA primary formulations rather than generic unsourced prose;
- §2.8.3 now grounds radial/angular spectral interpretation in Cao et al. and connects it to the actual angular-density/entropy mechanism of Xu et al. AFAB-2;
- §2.8.4 now explicitly separates input/data-space spectral processing from internal feature-space frequency methods (FFC, FDADNet, FDConv, WTConv) so they are not collapsed into a generic “high-frequency enhancement” category;
- Xu/AFAB is identified as the closest parent mechanism, but full LFDet gains are explicitly separated from AFAB-2 component evidence;
- AF2 is positioned as a standalone parameter-free adaptation of local-frequency + angular-amplitude selection before YOLO26, with coffee-specific residual reconstruction/integration reserved for Bab III.

### Claim boundary strengthened

The chapter now allows only the following bridge: Fourier representations can be manipulated and reconstructed; radial/angular spectra can describe frequency scale and directional structure; and frequency-aware processing has precedents in fine-grained/dense/defect vision. It **does not** allow the conclusion that coffee defects are proven frequency-separable or that AFAB-2 transfer is already validated.

### Audit update

- `BAB2_CITATION_AUDIT.md` marks §2.8.1–§2.8.4 as first-pass source-grounded, with one open item: exact textbook edition/page verification for the final DFT/FFT theory anchor;
- §2.9 related-work synthesis is now the only major Bab II prose/table section not yet promoted.

## v0.8.0 — 2026-08-24

Preprocessing section promoted to first-pass source-grounded prose.

### Rewritten

- §2.7 now distinguishes fixed/composite preprocessing, transform-domain preprocessing, learned task-driven preprocessing, downstream detection-utility cautions, and non-learned spectral input manipulation;
- `PRE-04` Syauqi and `PRE-05` Chen maize provide agricultural raw-vs-enhanced precedents; `PRE-06` WCTE provides transform-domain preprocessing; `PRE-01` IA-YOLO and `PRE-02` DENet provide task-driven learned preprocessing; `PRE-07` Retinexformer and `PRE-03` FE-YOLO reinforce downstream-utility evaluation; `PRE-08` FDA provides a non-learned Fourier input-space precedent outside detection;
- Syauqi is explicitly described as a **CLAHE-based composite pipeline** (gamma correction, CLAHE, blending, denoising, sharpening), not CLAHE alone;
- AF2 is now positioned as **parameter-free, input-space, content-adaptive spectral preprocessing**, rather than being incorrectly grouped as either a learned enhancement network or a fully fixed global filter;
- the chapter locks the distinction that improved visual appearance is not equivalent to improved detector utility.

### Audit update

- `BAB2_CITATION_AUDIT.md` marks §2.7 `PASS FIRST REWRITE` with eight distinct source keys;
- first-pass source-grounded prose is now complete through §2.7;
- next target is §2.8 Fourier/frequency theory, followed by the expanded §2.9 related-work table.

## v0.7.0 — 2026-08-24

Fine-grained object detection section promoted to first-pass source-grounded prose.

### Rewritten

- §2.6 now separates fine-grained definition/FGOD theory, coffee classification diagnostics, coffee object-detection difficult-class evidence, and model-internal discriminative-representation responses;
- `FG-03` + `FG-02` ground general theory; `COF-07/08` diagnose coffee granularity/generalization; `COF-03/04/05` provide direct detection similarity/tail evidence; `COF-12/13` show internal representation responses;
- Hong is intentionally absent from §2.6; Xu/`FG-01` is intentionally deferred to the frequency bridge so the parent AFAB paper does not dominate both problem diagnosis and solution-space justification;
- causal boundary locked: coffee literature supports fine-grained discrimination difficulty, not a proven frequency bottleneck.

### Audit update

- `BAB2_CITATION_AUDIT.md` marks §2.6 `PASS FIRST REWRITE` with nine distinct source keys;
- next source-grounded prose target is §2.7 preprocessing.

## v0.6.1 — 2026-08-24

Object-detection, YOLO, and YOLO26 theory sections have been promoted from scaffold prose to first-pass source-grounded prose.

### Rewritten

- §2.3 now uses Faster R-CNN and the original YOLO paper to distinguish two-stage and one-stage detection, then uses TOOD, Wu et al., and IoU-Net to justify classification/localization diagnosis;
- §2.4 now uses the original YOLO paper as the conceptual foundation and limits coffee-domain examples to Gope, Hong, and Bahy rather than recycling the entire coffee detector set;
- §2.5 now explicitly identifies YOLO26 as a 2026 arXiv preprint and grounds its dual-head end-to-end design, DFL removal, MuSGD, Progressive Loss, STAL, and P3/P4/P5 architecture in the primary YOLO26 source;
- AF2 is explicitly kept outside the YOLO26 backbone/neck/head and represented as an input-space treatment in the comparison equations.

### Audit update

`BAB2_CITATION_AUDIT.md` now marks the foundational-source gates for §2.3, §2.4, and §2.5 as closed for the first rewrite. Remaining major prose work begins at §2.6 fine-grained object detection, followed by preprocessing, Fourier theory, and the expanded related-work table.

## v0.6.0 — 2026-08-24

Citation-ready Bab II prose rewriting has started from the coffee-domain foundation.

### Rewritten

- `proposal/04_LITERATURE_REVIEW.md` §2.1 now derives its green-coffee definition, physical-defect vocabulary, grading context, and taxonomy boundaries from the official SNI plus verified coffee datasets/studies rather than generic prose;
- §2.2 now separates manual-inspection limitations, historical handcrafted computer vision, and the transition to deep-learning/edge systems using different source roles instead of recycling the same detector papers;
- §2.8.3 and §2.8.4 have been normalized to the canonical `SPEC-*`, `WAVE-*`, and `FREQ-*` namespaces while the file was being rewritten.

### Primary evidence used in this pass

- `STD-01` — SNI 01-2907-2008: green-coffee definition, defect definitions, quality grouping, and defect-value context;
- `COF-07` — Kesiman et al. 2023: SNI-aligned dataset construction, manual expert identification, and the final 17-defect subset;
- `COF-08` — Arwatchananukul et al. 2024: independent 17-type Thai Arabica defect taxonomy and manual-sorting context;
- `COF-02` — Bahy & Rifai 2026: 20-category SNI-oriented object-detection taxonomy;
- `COF-10` — de Oliveira et al. 2016: controlled-image/color-calibration computational-intelligence precedent;
- `COF-14` — Muchtar et al. 2025: manual-sorting fatigue/inconsistency and modern edge/deep-learning deployment context;
- `REV-01` — Motta et al. 2024: literature-landscape synthesis only, not a substitute for primary numerical evidence.

### Audit update

`BAB2_CITATION_AUDIT.md` now marks §2.2 as passing its first rewrite and §2.1 as functionally adequate with one official standard plus three independent operational taxonomy sources. The old frequency-key collision is also marked resolved in the active Bab II draft.

Sections §2.3–§2.9 are still working prose and must not yet be described as fully citation-ready.

## v0.5.0 — 2026-08-24

Concrete Bab II source routing and citation-key normalization added.

### Added

- `sources/CANONICAL_SOURCE_KEYS.md`: one stable key namespace across proposal files;
- `sources/BAB2_REFERENCE_POOL.md`: concrete 40+ source pool routed section-by-section across Bab II;
- `sources/BAB2_CITATION_AUDIT.md`: live audit of source count, citation recycling, missing foundational citations, key conflicts and promotion gates.

### Corrected

A citation-key collision was detected between the older method bridge and the latest master reference map. Older working files used `FREQ-01/FREQ-02` for Cao et al. and Zhang & Tan, while the master map already used those IDs for Fast Fourier Convolution and FDADNet. Canonical keys are now:

- `SPEC-01` — Cao et al. radial/angular Fourier energy;
- `SPEC-02` — Zhang & Tan orientation spectrum;
- `WAVE-01` — WTConv;
- `FREQ-01` — Fast Fourier Convolution;
- `FREQ-02` — FDADNet;
- `FREQ-03` — Frequency Dynamic Convolution.

`METHOD_BRIDGE_MATRIX.md` has been normalized accordingly and broadened to include canonical Fourier theory, additional preprocessing precedents, and explicit separation of input preprocessing from internal feature-space frequency methods.

### Citation-readiness finding

The current `04_LITERATURE_REVIEW.md` remains a structural draft. Its main deficiencies are now explicit rather than hidden:

- §2.1 and §2.5 have no resolved citations yet;
- §2.3 lacks foundational detector sources;
- §2.6–§2.8 are under-routed relative to the available corpus;
- several coffee papers are currently repeated across too many sections;
- §2.8 contains deprecated frequency-key aliases that must be normalized during rewrite;
- the related-work table should be promoted from the current working size to the verified 12–18-study campus-style shortlist.

The next drafting stage is therefore **source-normalized prose rewriting**, not additional literature searching by default.

## v0.4.0 — 2026-08-24

Reference-diversity and anti-recycling policy added for Bab II.

### Added

- `sources/REFERENCE_ALLOCATION_MATRIX.md`: subsection-by-subsection reference routing, diversity targets, source-role separation, and citation audit rules.

### Strengthened rules

- Hong remains the literature pivot but must not become a universal citation for unrelated concepts;
- each Bab II subsection receives its own source pool instead of recycling the same small set of papers;
- non-foundational empirical papers should normally have one primary argumentative role and at most one secondary role in Bab II;
- foundational sources may recur where technically necessary;
- final Bab II should aim for broad coverage (roughly 35–50 distinct authoritative/primary references as a planning target), without padding citations;
- final export requires an audit for unique-reference count, overused papers, unsupported technical paragraphs, source-group domination, numerical claim verification, and index/quartile verification.

## v0.3.0 — 2026-08-24

Bab II aligned to the uploaded USU/campus proposal convention.

### Added

- `foundation/06_USU_BAB2_PATTERN.md`: structural rules extracted from the uploaded campus proposal;
- `proposal/04_LITERATURE_REVIEW.md`: first Bab II scaffold using the campus pattern;
- campus-style `Penelitian Terkait` table with columns `Penulis & Tahun`, `Indeks`, `Fokus Penelitian`, `Metode / Model`, and `Kontribusi dan Pengisian Gap Penelitian`;
- final table row `Penelitian yang Diusulkan` positioning AF2 + YOLO26.

### Updated

- `proposal/01_PROPOSAL_SKELETON.md`: replaces the former free-form Bab II outline with the campus-aligned sequence.

### Frozen Bab II sequence

1. Biji Kopi Hijau dan Cacat Fisik Biji Kopi;
2. Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya;
3. Object Detection;
4. YOLO;
5. YOLO26;
6. Fine-Grained Object Detection;
7. Preprocessing Citra untuk Object Detection;
8. Representasi Citra pada Domain Frekuensi, including FFT, magnitude/phase, radial/angular spectrum, and frequency processing for detection;
9. Penelitian Terkait with a final proposed-research row.

### Important rule

We imitate the **campus structure and presentation pattern**, not unsupported claims or citation weaknesses in the example. Index/quartile values remain `TBD - verify` until checked against the appropriate bibliometric source and year.

## v0.2.0 — 2026-08-24

Evidence-to-draft layer added.

### Added

- `sources/COFFEE_EVIDENCE_MATRIX.md`: core coffee-domain problem evidence and overclaim boundaries;
- `sources/METHOD_BRIDGE_MATRIX.md`: preprocessing, frequency, angular, fine-grained, agricultural, and classification/localization bridge literature;
- `sources/BACKGROUND_CLAIM_LEDGER.md`: claim-by-claim traceability and safe wording for Bab I;
- `proposal/01_PROPOSAL_SKELETON.md`: Bab I–III structure, RQs, objectives, boundaries, metrics and fair-comparison rules;
- `proposal/02_BACKGROUND.md`: first evidence-grounded working draft of the thesis background.

### Strengthened rules

- keep coffee-domain evidence and non-coffee mechanism evidence as separate layers;
- every numerical claim must be re-opened in the primary PDF before final proposal prose;
- cross-paper synthesis must be labeled as synthesis rather than a fact from one source;
- terms such as `terbukti`, `menyebabkan`, `pertama`, `SOTA`, `optimal`, and `signifikan` require explicit evidence review;
- Bottom-3/Worst-class are our analysis choices motivated by class-wise heterogeneity, not metrics claimed as standard by the coffee papers;
- AF2-direct seed-42 remains pilot evidence only.

## v0.1.0 — 2026-08-24

Initial proposal-foundation freeze.

### Added

- dedicated branch `proposal/thesis-foundation` based on `codex/af2-direct-from-pretrained`;
- frozen thesis reasoning chain;
- Hong-centered literature-treatment method;
- evidence/claim guardrails;
- three-layer research gap;
- research questions, objectives, and scope boundaries;
- AF2-direct seed-42 pilot evidence with preliminary-only wording;
- document-generation guide;
- proposal workspace scaffold;
- source-registry guidance.

### Frozen decisions

- proposal narrative is problem-driven, not experiment-genealogy-driven;
- AF2 is framed as parameter-free input-space frequency-angular preprocessing;
- fine-grained discrimination is the primary problem framing supported by coffee literature;
- frequency is a candidate solution space, not a proven coffee-domain bottleneck;
- classification and localization effects must be diagnosed separately where possible;
- tail metrics must accompany aggregate performance;
- no further exploratory module stacking is required before proposal drafting begins.

### Not frozen as final thesis conclusions

- AF2 superiority across seeds;
- global novelty/"first" claims;
- final choice of optional CLAHE or operator ablations;
- final wording of "Analisis dan Optimasi" in the title.

Future conceptual changes should be recorded here and committed before dependent proposal chapters are revised.