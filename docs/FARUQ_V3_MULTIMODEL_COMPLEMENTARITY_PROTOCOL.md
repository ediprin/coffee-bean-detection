# Faruq-v3 Multi-Model Complementarity Audit — Seed 42

Status: **post-training validation diagnostic only**. No retraining and no test access.

## Purpose

Breadth screening showed that several individually useful classification-oriented mechanisms do not combine additively. The STB1–CMC0 three-seed audit also showed substantial shared error (mean Jaccard about 0.67) but non-zero oracle headroom. This audit asks which retained seed-42 model contributes the most **decision-level complementary information**, rather than which model has the highest standalone AP.

## Models

- CMC0 — capacity-near-matched non-spatial classification control
- STB1 — shifted-window classification interaction
- AF2 — frequency-input candidate
- IGEM1 — class-aware/local-context candidate
- SAF1 — scale-aware classification alignment candidate
- ACMC1 — ambiguity/confusion-oriented classification candidate

All arms use existing seed-42 checkpoints. No checkpoint is retrained.

## Branch-native checkpoint loading

Checkpoint wrappers originate from different research branches. To avoid false failures or silent class-definition mismatches, each checkpoint is loaded from its native branch and exported into a neutral validation-event JSON. The exported schema contains only final-detection events and no model-specific tensors.

## Frozen validation matching

- validation split only
- test split must not be exposed
- image size 640
- confidence threshold 0.25
- prediction NMS IoU 0.70
- max detections 500
- prediction-to-GT matching is class-agnostic at IoU >= 0.50 in descending confidence order
- a GT is correct only when matched at IoU >= 0.50 and assigned the correct class

These diagnostics are **not mAP** and do not replace Macro / Bottom-3 / Worst AP.

## Pairwise metrics

For each unordered model pair A/B:

- object accuracy at IoU50 for A and B
- directional rescue `P(B correct | A wrong)` and `P(A correct | B wrong)`
- shared-error Jaccard
- oracle accuracy if either A or B is correct
- oracle gain over the better single model
- classification-only rescue among jointly matched GT targets
- top directional confusion-pair rescues

Pairs are ranked primarily by **oracle gain over best**, then lower error Jaccard.

## Six-model oracle

The audit also reports an all-model oracle: the fraction of GT objects correctly handled by at least one of the six models. This is an upper bound on the information available to any selector; it is not an achievable fusion result.

## Interpretation

- high Jaccard + low oracle gain: operational redundancy; do not pursue fusion
- lower Jaccard + larger oracle gain: stronger evidence of complementary decision information
- concentrated rescue in specific confusion pairs: prefer class/confusion-conditional routing
- high all-model oracle but weak pairwise gains: complementary information may be fragmented across specialists

No new architecture is authorized by this descriptive audit. Any selector, gate, routing mechanism, or new loss must receive a separate frozen protocol before training.
