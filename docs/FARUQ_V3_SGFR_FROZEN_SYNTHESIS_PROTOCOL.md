# Faruq-v3 SGFR Frozen Residual Synthesis Protocol

Status: completed with a negative geometry-stage result.
Protocol: `faruq-v3-sgfr-frozen-synthesis-v1`  
Date frozen: 2026-08-12

SGI1 did not beat its optimization-matched SGC0 control, so SGF2 was not
authorized and test remained closed. See
`FARUQ_V3_SGFR_FROZEN_SYNTHESIS_RESULT_2026-08-13.md`.

## Question

Can complementary geometry and frequency residuals improve the retained STB1
classification pathway without changing its learned localization function?

This is a staged optimization study, not a two-stage detector. Inference remains
one YOLO26 forward pass. No ROI Align, crop classifier, decoded-box routing, or
top-k proposal classifier is permitted.

## Evidence used to select the synthesis

Seed-42 validation results already completed on the same Faruq-v3 split:

| Model | Macro mAP50-95 | Bottom-3 | Worst |
|---|---:|---:|---:|
| STB1 | 88.67% | 83.64% | 80.81% |
| IGEM1 | 88.01% | 82.18% | 82.08% |
| AF2 | 88.20% | 80.04% | 79.35% |

The per-class diagnostic oracle is **not a realizable result or performance
claim**. It only motivates complementarity: STB1+IGEM1+AF2 has an oracle Macro
90.12%, Bottom-3 85.96%, and Worst 85.48%. Validation class winners must not be
hard-coded into the model or used as routing labels.

## Frozen arms

All arms instantiate the same 7,331,021-parameter schema so checkpoints are
stage-compatible. Only the authorized parameter subset receives gradients.

- `SGC0`: STB1 continued-training control. Backbone, neck, STB blocks, and box
  heads are frozen; only native classification heads train.
- `SGI1`: STB1 + IGEM geometry residual. The complete STB1 model is frozen;
  only geometry residual and its train-time mask supervision branch train.
- `SGF2`: completed SGI1 + AF2 frequency residual. STB1 and the learned SGI1
  residual are frozen; only the lightweight frequency residual trains.

The deployed scores and boxes are

\[
z = z_{STB1} + \Delta z_{IGEM} + \Delta z_{AF2},
\qquad b = b_{STB1}.
\]

Each residual classification projection starts at exactly zero. The AF2 branch
encodes the deterministic AF2 input residual into class corrections; it does
not run a second detector backbone.

## Frozen-state contract

Freezing includes parameters **and** BatchNorm running buffers. At every stage:

- frozen backbone/neck/STB tensors must remain identical to their source;
- one-to-many and one-to-one box heads must remain identical;
- SGI1 may change only geometry modules;
- SGF2 may change only frequency modules;
- raw boxes must be bitwise identical at the zero-residual static gate;
- completed checkpoints undergo a state-dict invariance audit.

Classification scores may change post-processing selection even when raw box
coordinates are unchanged. Therefore proposal accessibility is reported later
if an arm survives the validation gate; it is not described as box-regression
improvement.

## Data and evaluation

- Dataset: `faruq-development-v3-grouped` train and validation only.
- Seed: 42 screening.
- Image size: 640.
- Batch: 16.
- Each optimization-control/residual stage: at most 20 epochs, patience 8.
- Metrics: Macro, Bottom-3, and Worst class mAP50-95.
- Test must not be extracted, mounted, or evaluated.

## Stage gates

### Geometry stage

Train `SGC0` and `SGI1` from the exact same STB1 checkpoint and schedule. SGI1
must pass against both STB1 and SGC0:

- Macro gain at least +0.5 percentage point;
- Bottom-3 not lower;
- Worst drop no more than 1 percentage point;
- frozen-state checkpoint audit PASS.

Failure stops SGFR. AF2 is not trained.

### Frequency stage

Only after geometry PASS, train SGF2 from the completed SGI1 checkpoint. SGF2
must pass the same gate against STB1, SGC0, and SGI1. If SGF2 fails, SGI1 is
retained as the surviving candidate and no test is opened.

## Capacity and claim boundary

SGFR has 7,331,021 total parameters versus 4,589,201 for STB1. A gain is not
capacity-free. Any surviving model must report parameter count, FP32 size, and
same-device latency. A later slim/capacity control is required before claiming
that geometry-frequency specialization, rather than capacity alone, caused a
marginal gain.

One seed is screening only. No robustness, superiority, external-domain, or
test claim is authorized by this protocol.
