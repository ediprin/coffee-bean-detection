# Proposal Draft Workspace

This directory contains proposal chapters generated from `../foundation/` and source/audit files in `../sources/`.

Do **not** treat prose drafts as the only source of truth. The thesis direction is controlled by the versioned foundation; numerical/methodological details must also agree with repository protocols and primary-paper evidence.

## Active files

```text
01_PROPOSAL_SKELETON.md
    synchronized Bab I–III structure and scope

02_BACKGROUND.md
    evidence-grounded Bab I background draft

03_PROBLEM_FORMULATION.md
    identification, RQ1–RQ4, objectives, scope, contribution

04_LITERATURE_REVIEW.md
    campus-style Bab II main draft

04_02_INSPECTION_QUALITY_NORMALIZED.md
    normalized replacement for §2.2 to remove citation recycling

04_09_RELATED_WORK_TABLE.md
    campus-style related-work table from the verified shortlist

05_METHODOLOGY.md
    full Bab III methodology draft with factorized AF2 optimization,
    direct confirmation, diagnostics, visualization, error and efficiency analysis

05_05_AF2_PRIMARY_SOURCE_HARDENED.md
    authoritative assembly replacement for §3.5;
    separates Xu et al. parent-method facts from repository transfer choices

06_RESEARCH_FLOW.md
    synchronized specifications for Figures 3.1–3.4:
    research framework, native-vs-AF2 architecture, AF2 operator,
    and factorized optimization/method-freeze genealogy
```

## Source/audit files that control Bab III

```text
../sources/BAB3_PRIMARY_SOURCE_HARDENING_2026-08-25.md
    exact primary-source locator/provenance audit for AF2 and YOLO26

../../FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_PROTOCOL_2026-08-24.md
    frozen direct confirmatory fairness/training/test-lock protocol

../../FARUQ_V3_AF2_SPECTRAL_FACTORIZATION_PROTOCOL.md
    frozen historical factorization/optimization genealogy

../../../src/coffee_detector/afab/operator.py
    executable AF2 operator and implementation-level transfer decisions

../../../configs/afab/AF2_yolo26n_chaotic_amplitude.yaml
    frozen AF2 reference configuration
```

## Assembly rule

When generating the final proposal document:

1. use `04_02_INSPECTION_QUALITY_NORMALIZED.md` in place of the older §2.2 block embedded in `04_LITERATURE_REVIEW.md`;
2. use `04_09_RELATED_WORK_TABLE.md` as the authoritative related-work table;
3. use `05_METHODOLOGY.md` as the Bab III base;
4. replace the embedded §3.5 of `05_METHODOLOGY.md` with `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`;
5. redraw Figures 3.1–3.4 from `06_RESEARCH_FLOW.md` into the campus document style;
6. resolve citation keys through `../sources/CANONICAL_SOURCE_KEYS.md`;
7. apply `../sources/BAB3_PRIMARY_SOURCE_HARDENING_2026-08-25.md` before locking AF2/YOLO26 captions, equations, or page locators;
8. run the Bab II, Bab III, and cross-chapter audits before export.

## Drafting rule

Before creating or changing a chapter:

1. read the relevant foundation files;
2. retrieve the primary papers or repository protocols needed for the chapter;
3. verify numerical and methodological claims against full text / frozen artifacts;
4. distinguish source facts, cross-paper synthesis, research hypotheses, and repository evidence;
5. preserve temporal status: historical factorization, completed pilot, and planned confirmation are different evidence layers;
6. preserve experiment genealogy: historical D0-parent factorization must not be drawn as the final direct-from-pretrained training path;
7. commit meaningful argument changes separately so the proposal remains versionable.

## Current state

```text
Bab I   = evidence-grounded working draft; RQ1–RQ4 synchronized with Optimasi scope
Bab II  = source-grounded campus-style draft with modular normalization
Bab III = optimization-centered methodology draft completed;
          AF2/YOLO26 primary-source provenance hardened
Flow    = Figures 3.1–3.4 specification synchronized with Bab III
Title   = 'Analisis dan Optimasi ...' methodologically supported by factorized optimization
```

One citation-location task remains before page-perfect formal assembly: directly re-capture the printed Xu et al. AFAB-2 Eq. (10)–(13) page block. The algorithm definition is already grounded in the parent section and repository mapping; no page number should be invented until that recertification is complete.

Next major task: formal proposal assembly into the campus DOCX template, after the remaining page-locator recertification and final chapter-level audit.