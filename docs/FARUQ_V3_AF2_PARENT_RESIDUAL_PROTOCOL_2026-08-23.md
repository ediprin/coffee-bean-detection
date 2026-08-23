# Faruq-v3 AF2 Parent-Preserving Residual Protocol

Status: frozen before training

## Research question

The completed direct combinations `AF2STB1`, `AF2IGEM1`, and `AF2SAF1` were
trained jointly from D0. They therefore tested whether two mechanisms can be
relearned together, but did not test whether a retained standalone mechanism
can improve an already trained AF2 without erasing AF2's representation.

This study asks a narrower question: can a classification-only SAF or IGEM
residual improve a **frozen completed AF2 parent** under a matched
zero-information optimization/capacity control?

## Arms

| Family | Matched control | Candidate | Information entering residual |
|---|---|---|---|
| SAF alignment | `AF2SAF0` | `AF2SAF1` | zero vs frozen AF2 P3/P4/P5 |
| IGEM guidance | `AF2IGEM0` | `AF2IGEM1` | zero vs frozen AF2 P3/P4/P5 |

All four arms load the completed AF2 seed-42 checkpoint. The AF2 frontend,
backbone, neck, native box path, and native classification path are frozen.
Only the newly attached residual is trainable. The residual is zero-output
initialized, so every arm begins at AF2. Boxes remain native and cannot be
modified by the residual.

The IGEM pair retains the original coarse bbox-derived mask objective with
weight 0.05 in both candidate and control. The SAF pair uses the native
detection objective in both arms.

## Fixed development protocol

- Dataset: leakage-safe Faruq-v3 grouped development data.
- Split: train for fitting, validation for the frozen screening decision.
- Test: unavailable and locked.
- Source: AF2 seed 42, never D0 and never a standalone SAF/IGEM checkpoint.
- Seed: 42 only during screening.
- Schedule: 20 epochs, image size 640, batch 16, patience 7, identical inside
  each candidate/control family.
- Resume: `last.pt` is updated every epoch; a completion marker is bound to the
  source AF2 SHA, arm, seed, and requested epoch count.

## Static gates

Before training, the audit must establish for each family:

1. identical model YAML, AF2 configuration, training schedule, parameter
   count, trainable count, and state schema between control and candidate;
2. exact zero-information control and real-feature candidate routing;
3. numerical AF2 identity at zero initialization;
4. active candidate changes class scores while boxes stay bitwise unchanged;
5. finite residual gradients while the AF2 parent receives no gradients;
6. no ROI, decoded-box dependency, or test access.

## Seed-42 decision

The candidate must first remain close to original AF2: Macro may not fall more
than 0.2 point and Bottom-3/Worst may not fall more than 1.0 point. It must then
beat its matched control through at least one preregistered route:

- **superiority route:** Macro +0.2 point or more, Bottom-3 loss at most 0.5
  point, and Worst loss at most 1.0 point; or
- **tail-Pareto route:** Macro loss at most 0.1 point, Bottom-3 +0.5 point or
  more, and Worst +1.0 point or more.

All 21 validation classes must be present. Only a retained family may proceed
to paired seeds 123 and 2026. A rejected family stops without additional seeds.

## Interpretation boundary

A positive result supports parent-preserving residual adaptation, not a generic
claim that stacking retained modules is always beneficial. A negative result
closes only this staged SAF/IGEM route; it does not invalidate standalone AF2,
SAF, or IGEM evidence.
