# Faruq-v3 Label Visual Audit Protocol

Version: 1.0.0

Status: executed; AI-assisted visual review complete, domain-expert confirmation pending

Scope: development train and validation only

## Purpose

Resolve the `DATA_OR_SCALE_LIMITED` result before another model experiment.
The audit visually checks the weak `kulit_tanduk` size family and the dominant
local-defect confusion pairs. It does not train a model, run inference, or
access test images.

## Selection and views

- Samples are selected deterministically at normalized-area quantiles, rather
  than manually choosing favorable examples.
- Every tile shows both the full-frame context with the target box and a zoomed
  target crop. The context view preserves the apparent scale; the crop view
  exposes local defect evidence.
- Size-family sheets are generated separately for train and validation for
  `kulit_kopi`, `kulit_tanduk`, and `tanah_batu_ranting`.
- Pair sheets are generated separately for train and validation for the six
  highest-count local-defect confusion directions from the frozen D0
  diagnostic.

## Human review gate

Reviewers must answer:

1. Are small, medium, and large labels visually ordered within each split?
2. Does the scene contain a stable physical scale reference?
3. Are local defect cues visible and consistent within each class?
4. Do train and validation apply the same label interpretation?

The script intentionally returns `PENDING_HUMAN_VISUAL_REVIEW`. It does not
automatically relabel or merge classes. A model change remains blocked until
the contact sheets are reviewed and the resulting label decision is recorded.

## Recorded outcome — 2026-08-02

All six size-family sheets and twelve local-pair sheets were generated without
training, inference, or test access. The review found mixed label
observability: `kulit_tanduk` size labels overlap strongly without a stable
physical scale reference, while several local defects retain subtle but small
visual cues. Architecture search remains blocked pending an ontology and label
observability review by an SNI domain expert. See
`docs/FARUQ_V3_LABEL_VISUAL_AUDIT_RESULT_2026-08-02.md`.
