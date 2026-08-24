# Proposal Draft Workspace

This directory contains proposal chapters generated from `../foundation/` and source/audit files in `../sources/`.

Do **not** treat prose drafts as the only source of truth. The thesis direction is controlled by the versioned foundation; numerical/methodological details must also agree with repository protocols and primary-paper evidence.

## Active files

```text
01_PROPOSAL_SKELETON.md
    overall Bab I–III structure and scope

02_BACKGROUND.md
    evidence-grounded Bab I background draft

03_PROBLEM_FORMULATION.md
    identification, research questions, objectives, scope, contribution

04_LITERATURE_REVIEW.md
    campus-style Bab II main draft

04_02_INSPECTION_QUALITY_NORMALIZED.md
    normalized replacement for §2.2 to remove citation recycling

04_09_RELATED_WORK_TABLE.md
    18-study + proposed-research campus-style related-work table

05_METHODOLOGY.md
    campus-style Bab III source-grounded from frozen direct-AF2 protocol
```

## Assembly rule

When generating the final proposal document:

1. use `04_02_INSPECTION_QUALITY_NORMALIZED.md` in place of the older §2.2 block embedded in `04_LITERATURE_REVIEW.md`;
2. use `04_09_RELATED_WORK_TABLE.md` as the authoritative §2.9 table;
3. use `05_METHODOLOGY.md` as the Bab III base;
4. resolve all citation keys through `../sources/CANONICAL_SOURCE_KEYS.md`;
5. apply the Bab II and Bab III audit files before export.

## Drafting rule

Before creating or changing a chapter:

1. read the relevant foundation files;
2. retrieve the primary papers or repository protocols needed for the chapter;
3. verify numerical and methodological claims against full text / frozen artifacts;
4. distinguish source facts, cross-paper synthesis, research hypotheses, and repository evidence;
5. preserve temporal status: completed pilot versus planned confirmation;
6. commit meaningful argument changes separately so the proposal remains versionable.

## Current state

```text
Bab I   = working evidence-grounded draft
Bab II  = first-pass source-grounded; final assembly/normalization pending
Bab III = first source-grounded methodology draft completed
```

The next major drafting task after normalization is the proposal-level research-flow diagram and final consistency audit across Bab I–III.
