# 05 — Document Generation Guide

## 1. Purpose

This file is the instruction layer for generating future thesis-proposal documents from the frozen reasoning base in this branch.

Before drafting any proposal section, read:

1. `00_THESIS_CONCEPT.md`
2. `01_HONG_PIVOT_LITERATURE_METHOD.md`
3. `02_EVIDENCE_AND_CLAIM_RULES.md`
4. `03_RESEARCH_GAP_RQ_SCOPE.md`
5. `04_PILOT_EVIDENCE.md`

If a future draft contradicts these files, the contradiction must be explicitly resolved by updating the foundation first. Do not silently drift the thesis direction inside a chapter draft.

## 2. Required argumentative order for the Background

The Background should normally follow this chain:

```text
A. coffee quality-control importance
B. manual/visual inspection limitations
C. transition to CV/deep learning
D. YOLO viability in coffee tasks
E. success in few/coarse-class settings does not close the fine-grained problem
F. multi-class coffee evidence: visually similar classes, class-wise disparity, tail difficulty
G. current coffee responses are mostly model-internal representation improvements
H. adjacent literature: task-oriented preprocessing can improve downstream detection
I. adjacent literature: frequency-domain processing provides a technically plausible representation space
J. candidate gap: parameter-free frequency-angular input preprocessing in coffee fine-grained detection
K. thesis hypothesis and controlled evaluation
L. optional pilot feasibility result
```

Do not start the Background with AF2 mathematics. The reader must first understand why the coffee problem requires the investigation.

## 3. Paragraph construction rule

Each literature paragraph should have three parts when possible:

```text
source fact -> relevance to the argument -> limitation/boundary
```

Example structure:

> Paper A reports X on Y classes. This indicates that model family Z is viable for coffee-defect recognition. However, because the taxonomy contains only Y classes, the result does not establish performance for a more granular 17–20-class setting.

This prevents literature review from becoming a list of unrelated accuracy values.

## 4. Hong-centered synthesis rule

Use Hong as a **pivot**, not as the sole authority.

Recommended pattern:

```text
Hong identifies/frames problem P
    ↓
parallel/earlier/later coffee papers are checked for whether P recurs
    ↓
if recurring, formulate cross-paper coffee-domain synthesis
    ↓
then introduce non-coffee mechanism literature
```

This makes the thesis argument independent of any single paper.

## 5. Numerical claims

For every numerical claim planned for the final proposal:

- verify the original full-text table/figure/section;
- verify metric definition (mAP50 vs mAP50-95 vs accuracy);
- verify number of classes;
- verify whether data are original or augmented;
- verify split protocol where relevant;
- avoid comparing incompatible metrics as if they were directly equivalent.

If a number is not re-verified during final drafting, omit it or mark it for verification rather than guessing.

## 6. Literature table format

Use a state-of-the-art matrix with fields such as:

| Field | Purpose |
|---|---|
| Paper/year | identity |
| Task | detection/classification/grading |
| Dataset | domain and scale |
| Classes | granularity |
| Model/baseline | technical context |
| Main modification | what changed |
| Main result | verified metric |
| Difficult classes | tail/fine-grained evidence |
| Author explanation | source-derived reason |
| Limitation | what remains open |
| Role in thesis | problem evidence / method bridge / baseline |

Do not create a table whose only columns are `paper`, `model`, and `accuracy`; that format loses the problem/gap logic.

## 7. Methodology generation

The minimum methodology narrative should define:

```text
Input image I
 -> AF2 operator A_FA(I)
 -> processed image I'
 -> YOLO26 detector D_theta(I')
 -> predictions
```

Then separately define:

- matched baseline;
- data/split protocol;
- training protocol;
- primary metrics;
- tail metrics;
- classification/localization diagnostics;
- efficiency metrics.

Do not mix implementation genealogy (old D0FT/experimental branches) into the core mathematical method unless needed for reproducibility/history.

## 8. Mathematics rule

Equations in the proposal must fall into one of these categories:

- equation explicitly defined by a cited paper;
- equation implemented by AF2 in this repository;
- transparent mathematical definition of an evaluation metric;
- clearly labeled derivation by the thesis author.

Never invent a paper equation merely to make the document look mathematical.

## 9. Pilot-result placement

Pilot evidence belongs near the end of Background, preliminary study, feasibility, or methodology justification depending on campus format.

It must be labeled:

- preliminary;
- one-seed screening;
- not final thesis conclusion.

Its purpose is to establish feasibility, not superiority.

## 10. Proposal files to generate next

Recommended order:

```text
proposal/
├── 01_TITLE_AND_SCOPE.md
├── 02_BACKGROUND.md
├── 03_PROBLEM_IDENTIFICATION.md
├── 04_RESEARCH_QUESTIONS.md
├── 05_OBJECTIVES_AND_CONTRIBUTIONS.md
├── 06_LITERATURE_REVIEW.md
├── 07_METHODOLOGY.md
├── 08_EVALUATION_PLAN.md
└── 09_RESEARCH_FLOW.md
```

Generate these incrementally. Commit each meaningful revision with a message that states what changed in the argument, not merely "update docs".

## 11. Versioning rule

When the conceptual foundation changes, update the relevant `foundation/` file first and commit it separately. Then update dependent proposal chapters.

Suggested commit prefixes:

```text
docs(thesis): ...
lit(thesis): ...
method(thesis): ...
eval(thesis): ...
ref(thesis): ...
```

This preserves a traceable history of why the thesis narrative changed.
