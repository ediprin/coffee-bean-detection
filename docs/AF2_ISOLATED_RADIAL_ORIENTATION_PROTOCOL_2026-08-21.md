# AF2 isolated radial/orientation protocol — 2026-08-21

## Decision context

The cumulative CAFR seed42 screen did not beat the retained AF2 parent. Because C2/C4 were evaluated on top of earlier CAFR changes, their radial/orientation effects were not causally isolated against the original AF2 operator.

This follow-up tests only two hypotheses directly against AF2:

1. `AF2_RADIAL`: preserve original AF2 and add three fixed radial bands.
2. `AF2_ORIENT`: preserve original AF2 and fold signed directions into unsigned 180-degree orientations.

No combined arm is authorized yet. `AF2_RADIAL + AF2_ORIENT` is only allowed if the isolated seed42 evidence justifies it.

## Parent operator retained exactly

The AF2 parent is the current AFAB-2 transfer:

- patch size: 32
- overlap: 0.50
- gamma: 0.10
- independent RGB channel FFT/filtering
- hard entropy-conditioned suppression
- original Fourier phase retained
- fold/overlap averaging reconstruction
- per-channel min-max recovered gate
- residual `X' = X + X * N(R)`
- native YOLO26n-p3 architecture and loss unchanged

`AF2_BASE` exists as a code-level parity control and is not scheduled for training. Unit tests require its geometry/output to match the legacy AF2 implementation.

## AF2_RADIAL

Only the density aggregation changes from:

`D(theta) = sum_r A(r, theta)`

to three independently normalized radial groups:

`D_b(theta) = sum_{r in B_b} A(r, theta)`

with frozen normalized-radius boundaries:

- low: `[0, 1/3)`
- mid: `[1/3, 2/3)`
- high: `[2/3, 1]`

Direction remains 360 degrees / 360 bins. Hard AF2 selection is applied independently within each radial band. No luminance/shared gate, soft threshold, windowing, or patch-size change is introduced.

## AF2_ORIENT

Only the angular domain changes:

- parent: 360-degree signed direction, 360 bins
- candidate: 180-degree unsigned orientation, 180 bins

Both therefore keep approximately one degree per bin. This avoids the earlier confound where an orientation experiment also reduced angular resolution to 16 bins and used a Hann-window patch pipeline.

No radial split, luminance/shared gate, soft threshold, windowing, or patch-size change is introduced.

## Development protocol

- seed: 42 only
- start checkpoint: same D0 seed42 checkpoint used by the development protocol
- epochs: 50
- image size: 640
- batch: 16
- validation only
- locked test forbidden
- both isolated arms may run in parallel because neither depends on the other

Primary screening metrics:

1. Macro mAP50-95
2. Bottom-3 class mAP50-95
3. Worst-class mAP50-95

The retained AF2 seed42 reference is approximately:

- Macro: 0.8819734
- Bottom-3: 0.800428
- Worst: 0.793470

These reference values are descriptive; the result artifacts from the isolated arms are the authoritative outputs for the new runs.

## Decision rule

An isolated candidate is not promoted merely because one tail metric improves. Promotion requires a defensible tradeoff against AF2, with Macro not materially degraded and a meaningful Bottom-3/Worst improvement. Exact promotion thresholds should be frozen before multi-seed confirmation, after inspecting the two seed42 isolated results together.

If neither isolated arm is better than AF2, stop this optimization branch and retain AF2 unchanged. If one isolated arm is clearly better, promote only that arm to seeds 123 and 2026. If both are independently useful, then and only then construct a combined radial+orientation arm and screen it at seed42 before multi-seed confirmation.
