# Faruq-v3 STB-SR1 Selective Spatial Residual Protocol

Date frozen: 2026-08-19  
Branch: `agent/stb-sr1-selective-residual`  
Status: **FROZEN BEFORE TRAINING**  
Evaluation: Faruq-v3 grouped development validation only  
Locked test: **closed**

## Evidence trigger

The completed STB capacity-control study showed two facts that must be kept separate:

1. `STB1` is a strong validation model: three-seed mean `87.82 / 80.50 / 78.36` for Macro / Bottom-3 / Worst.
2. Its shifted-window spatial mechanism was not confirmed as a robust causal gain over the capacity-near-matched non-spatial `CMC0`: mean Macro difference was only about `+0.07` point, below the frozen `+0.50` criterion.

Therefore the next experiment does **not** claim that shifted-window attention is already validated. It asks whether a new architecture can use the strong CMC-style representation as the main classification transform and restrict spatial interaction to an optional residual correction.

## Important implementation correction

The existing `STB1` already has one learnable zero-initialized scalar gate at each P3/P4/P5 classification level. Therefore merely adding per-level scalars would reproduce the existing STB1 design and is not a new experiment.

`STB-SR1` instead changes the internal factorization.

## STB-SR1 architecture

For each classification pyramid feature `X_l`, `l in {P3,P4,P5}`:

### 1. CMC-style base

Two pointwise channel-mixing blocks, exactly the same block family used by `CMC0`, produce a gated base representation:

\[
B_l = X_l + g^{(c)}_l\big(C_l(X_l)-X_l\big),
\qquad g^{(c)}_l(0)=0.
\]

### 2. Attention-only spatial correction

Starting from `B_l`, apply one non-shifted window-attention residual followed by one shifted-window-attention residual. These residual blocks contain only `LayerNorm -> ShiftedWindowAttention -> residual`; **no Swin MLP is duplicated in the spatial branch**.

\[
S_l = A^{SW}_l\big(A^{W}_l(B_l)\big).
\]

The final classification feature is:

\[
\boxed{
Y_l = B_l + g^{(s)}_l\big(S_l-B_l\big)
}
\]

with an independent scalar spatial gate:

\[
g^{(s)}_l(0)=0.
\]

Thus at initialization:

\[
Y_l=X_l.
\]

The native YOLO26 localization/box path continues to consume `X_l` and is unchanged. Only classification logits consume `Y_l`.

## Frozen hyperparameters

- model: `configs/coffee_fg/models/yolo26n-p3.yaml`
- window size: `4`
- attention heads: `4`
- CMC MLP ratio: `4.0`
- epochs: `50`
- image size: `640`
- batch: `16`
- workers: `2`
- patience: `15`
- optimizer: `auto`
- close mosaic: `10`
- max detections: `500`
- seed: **42 only for Stage A**
- deterministic training: enabled
- initialization: same native seed-42 D0 checkpoint used by the historical STB/CMC family

No STB1 or CMC0 trained checkpoint is used to initialize STB-SR1.

## Static authorization gate

Training is forbidden until all checks pass:

1. zero-gate STB-SR1 boxes are bitwise identical to native D0;
2. zero-gate STB-SR1 scores are bitwise identical to native D0;
3. activating only the CMC gate changes scores while preserving boxes;
4. activating only the spatial gate changes scores while preserving boxes;
5. all three P3/P4/P5 levels are modified on classification only;
6. every level contains two CMC blocks plus W-MSA and SW-MSA attention-only residuals;
7. the spatial branch contains no MLP module;
8. parameter counts and overhead versus CMC0/STB1 are recorded;
9. no test data are exposed.

## Frozen Stage-A references

Use the already frozen seed-42 values from `docs/evidence/FARUQ_V3_STB_CAPACITY_PAIRED_CONFIRMATION_2026-08-14.json`:

| Model | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|
| CMC0 | 87.1006% | 81.8664% | 81.3081% |
| STB1 | 88.6694% | 83.6394% | 80.8137% |

These references are not retrained during Stage A.

## Stage-A decision gate

STB-SR1 proceeds to paired multiseed confirmation only if **all** of the following hold on seed 42.

### A. Improvement beyond the CMC base

\[
\Delta Macro_{SR1-CMC0}\ge +0.50\text{ pp}
\]

\[
\Delta Bottom3_{SR1-CMC0}\ge 0
\]

\[
\Delta Worst_{SR1-CMC0}\ge -1.00\text{ pp}
\]

### B. Retention versus the strong STB1 parent

\[
\Delta Macro_{SR1-STB1}\ge -0.50\text{ pp}
\]

\[
\Delta Bottom3_{SR1-STB1}\ge -1.00\text{ pp}
\]

\[
\Delta Worst_{SR1-STB1}\ge -1.00\text{ pp}
\]

### C. At least one advancement signal versus STB1

At least one must hold:

\[
\Delta Macro_{SR1-STB1}\ge +0.20\text{ pp}
\]

or

\[
\Delta Bottom3_{SR1-STB1}\ge +0.50\text{ pp}
\]

or

\[
\Delta Worst_{SR1-STB1}\ge +0.50\text{ pp}.
\]

If any retention criterion fails, or if there is no advancement signal, the experiment stops after seed 42 without test access.

## Claim boundary

A Stage-A PASS would support only:

> On Faruq-v3 development validation seed 42, the CMC-plus-attention-only residual architecture improves or preserves the strong STB-family reference under the frozen screening gate.

It would **not** prove that shifted-window attention is the causal reason for the gain, would not establish cross-seed stability, and would not authorize a locked-test claim.

Only after a Stage-A PASS may a new paired seed-42/123/2026 confirmation protocol be executed. No fourth seed, post-result retuning, AF2/WAV fusion, or locked-test access is permitted under this Stage-A protocol.
