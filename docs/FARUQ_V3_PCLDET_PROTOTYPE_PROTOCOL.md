# Faruq-v3 PCLDet Learned-Prototype Screening Protocol

Date frozen: 2026-08-07
Stage: broad candidate search / predecessor prototype control
Candidate: `PCL1`
Branch: `agent/pcldet-prototype-baseline-screening`

## Purpose

PCL1 is the predecessor control for APCL. It tests whether the original PCLDet
idea of a **learnable class-prototype bank + ProtoCL loss** improves the
fine-grained coffee detector before attributing any benefit to APCL's newer
EMA prototype update.

This is required for a defensible PCL -> APCL comparison. It is not a claim that
PCLDet as a whole has been literally reproduced on YOLO26.

## Full-paper basis

Ouyang et al., IEEE TGRS 2023, define:

- one prototype `w_k` per fine-grained class;
- cosine similarity between sample feature `x_i` and prototype `w_k` (Eq. 1);
- ProtoCL loss with own-class attraction and other-class repulsion (Eq. 3);
- temperature `tau = 1/32`;
- prototypes optimized by SGD/gradient descent (Eq. 4);
- a separate class-balanced sampler (CBS) for long-tail positive proposals.

PCL1 transfers Eqs. 1, 3, and 4. CBS is deliberately excluded from this
screen because:

1. the controlled coffee benchmark is approximately class-balanced;
2. CBS is formulated for positive proposal sampling in an RPN;
3. YOLO26 is a one-stage/end-to-end dense detector and has no RPN;
4. adding a new sampling intervention would confound the prototype-mechanism
   comparison against APCL.

Therefore a negative PCL1 result is evidence about this learned-prototype
transfer, not about the complete PCLDet system.

## YOLO26 adaptation

- Native YOLO26 one-to-many P3/P4/P5 dense classification features are projected
  to a 128-D embedding space.
- Only positively assigned one-to-many samples enter ProtoCL.
- One-to-one companion branch remains native YOLO26 loss.
- One learnable prototype parameter exists per one of the 21 coffee classes.
- Prototype bank is initialized from a normal distribution.
- The paper does not specify the standard deviation in its method text; PCL1
  explicitly freezes `std=1.0` as an implementation choice, not a paper fact.
- `tau=1/32` is taken directly from the paper.
- ProtoCL has weight 1.0 in the classification-loss component.
- Projection/prototype branch is training-only and skipped at inference.

For predecessor fairness, PCL1 uses the same 128-D P3/P4/P5 projection topology
and the same screening schedule as the existing APCL1 adaptation. The substantive
prototype difference is:

- PCL1: prototypes are `nn.Parameter` objects optimized by gradients;
- APCL1: prototypes are non-gradient EMA state derived from class instance means.

## Frozen data and controls

- Faruq-v3 grouped development train/val only.
- Locked holdout/test must not exist in the extracted development root.
- seed = 42 for discovery screen.
- initial weights = same D0 checkpoint used by APCL1.
- primary control = D0FT.
- ACMC1 is reported as selected-model reference, not the primary optimization
  control.
- no validation confusion pairs or hard-class identities are supplied to PCL1.

## Frozen training configuration

Matched to APCL1:

- 50 epochs;
- image size 640;
- batch 16;
- workers 2;
- patience 15;
- optimizer `auto` as used by the paired APCL implementation;
- pretrained/native D0 initialization;
- `close_mosaic=10`;
- `max_det=500`;
- deterministic seed 42.

PCL-specific:

- embedding dimension 128;
- temperature 1/32 = 0.03125;
- ProtoCL weight 1.0;
- normal prototype initialization std 1.0.

## Frozen discovery gate

Same broad-search gate family used by APCL1.

Discovery signal: at least one of:

- Macro mAP50-95 vs D0FT >= +0.20 pp;
- Bottom-3 class mAP50-95 vs D0FT >= +0.50 pp;
- Worst-class mAP50-95 vs D0FT >= +0.50 pp.

Safeguards, all required:

- Macro drop vs D0FT no worse than -1.00 pp;
- Bottom-3 drop no worse than -2.00 pp;
- Worst drop no worse than -2.00 pp.

If discovery signal + safeguards pass: `RETAIN`.
Otherwise: `REJECT` for candidate selection, while keeping PCL1 as a method
control for interpreting APCL.

## Scientific boundaries

- No test access.
- No CBS claim.
- No ReDet/RPN reproduction claim.
- No claim that Eq. 3 proposal semantics are literally identical after transfer
  to YOLO26 dense positive assignments.
- No inference overhead from the PCL auxiliary branch.
- APCL superiority, if observed later, must be measured empirically; it must not
  be assumed from the newer paper alone.
