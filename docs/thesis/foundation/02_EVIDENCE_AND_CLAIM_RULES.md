# 02 — Evidence and Claim Rules

## 1. Claim classes

Every substantive statement in future proposal/thesis writing should belong to one of four classes.

### PAPER FACT

A statement explicitly supported by a primary paper.

Examples:

- a dataset contains N classes;
- a model reports a specific mAP value;
- an author states that two defect classes are visually similar;
- a module is inserted at a specific architecture location.

Requirement: cite the original paper, preferably full text with page/section/table verification during drafting.

### CROSS-PAPER SYNTHESIS

A pattern inferred by comparing several papers.

Example:

> More granular coffee-defect taxonomies tend to expose stronger class-wise heterogeneity than coarse/few-class settings in the reviewed literature.

Requirement: cite multiple relevant coffee papers. Do not attribute the synthesis to one paper unless it explicitly says so.

### RESEARCH HYPOTHESIS

A proposition this thesis will test.

Example:

> Frequency-angular preprocessing may improve fine-grained class discrimination without materially changing raw localization accessibility.

Requirement: label it as a hypothesis, motivation, or research question. Never present it as established literature fact.

### REPOSITORY EVIDENCE

A result produced by this repository under a defined experiment protocol.

Example:

> In the seed-42 direct screening, AF2 improved Bottom-3 mAP50-95 relative to the matched native arm.

Requirement: identify the experiment arm, seed, budget, dataset/split, and screening/final status.

## 2. Forbidden reasoning shortcuts

Do not write claims with the following logic:

```text
frequency helps aircraft detection
therefore frequency solves coffee defects
```

or

```text
processed images look sharper
therefore detection must improve
```

or

```text
overall mAP is high
therefore fine-grained classes are solved
```

or

```text
AF2 has zero learned parameters
therefore AF2 is lightweight
```

or

```text
one seed is positive
therefore AF2 is superior
```

## 3. Required cautious formulations

Prefer:

> In the reviewed coffee literature, visually similar defect categories and class-wise performance disparity remain recurring issues.

instead of:

> All coffee-defect studies suffer from the same bottleneck.

Prefer:

> Frequency-domain methods in adjacent detection tasks motivate spectral preprocessing as a candidate solution space.

instead of:

> Frequency processing is the established solution for coffee defects.

Prefer:

> AF2 introduces no additional learned preprocessing parameters.

instead of:

> AF2 is cost-free/lightweight.

Prefer:

> The seed-42 pilot provides preliminary feasibility evidence.

instead of:

> AF2 is proven to outperform YOLO26.

## 4. Coffee-domain versus adjacent-domain evidence

Tag literature mentally as one of:

- `DIRECT_COFFEE`: coffee bean defect/grading/recognition/detection;
- `ADJACENT_SEED_GRAIN`: pepper, maize seed, cocoa, wheat, rice, etc.;
- `GENERAL_DETECTION`: general object detection or adverse-weather detection;
- `MECHANISM_PARENT`: paper that defines the underlying frequency/angular/operator mechanism.

Direct coffee evidence has priority for the **problem statement**.

Adjacent/general/mechanism papers are mainly used for **solution motivation** and **technical design precedent**.

## 5. Classification versus localization

Do not infer classification and localization effects from aggregate mAP alone.

Where the repository provides diagnostics, separate:

- raw proposal accessibility;
- final proposal accessibility/ranking;
- localization-conditioned Top-1 classification;
- correct-decision recall;
- class-wise AP.

The thesis may use detector literature such as TOOD, Rethinking Classification and Localization, and IoU-Net to justify analyzing the two subtasks separately, but coffee-specific conclusions must still come from coffee experiments.

## 6. Tail metrics

Primary detection reporting should not rely only on aggregate mAP.

Track at minimum:

- Macro mAP50-95;
- Bottom-3 class mAP50-95;
- Worst-class mAP50-95;
- per-class AP;
- standard detector metrics where needed (mAP50, precision, recall).

Rationale: coffee literature repeatedly shows class-wise heterogeneity; tail metrics make difficult classes visible rather than allowing aggregate performance to hide them.

## 7. Experiment-control language

For a comparison to be described as matched, verify:

- same dataset and split;
- same pretrained initialization;
- same image size;
- same epoch/training budget;
- same optimizer/settings;
- same augmentation policy;
- paired seed where applicable;
- same test-access policy.

If any item differs, document it explicitly.

## 8. Source handling for generated documents

When generating a proposal chapter:

1. retrieve/read the relevant primary sources;
2. extract only claims actually supported by those sources;
3. keep author explanations separate from our interpretation;
4. mark unresolved bibliographic or numerical details instead of guessing;
5. do not invent equations that a cited paper does not contain;
6. do not convert repository implementation details into claims about the parent paper;
7. do not cite a review/snippet when the original paper is available.

## 9. Novelty wording

Until a systematic novelty search is complete, use:

> "dalam literatur yang ditinjau" / "pada corpus yang dikaji"

rather than:

> "belum pernah dilakukan" / "the first study".

A safe novelty position is methodological and empirical:

> The thesis evaluates parameter-free frequency-angular input preprocessing as an alternative to the predominantly model-internal representation improvements found in the reviewed coffee-defect literature.
