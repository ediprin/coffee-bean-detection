# Faruq-v3 AF2 + CPE0 seed-42 frozen protocol

## Question and scope

Does the already-retained AF2 input operator benefit from the already-defined CPE0 all-positive supervised-contrastive objective? This is a single seed-42 validation screen. It is not hyperparameter tuning, does not authorize locked-test access, and does not start GDS+STB. GDS+STB remains blocked until standalone GDSC1 is empirically viable.

## Frozen matched pair

Both arms start from the same AF2 seed-42 `best.pt`, use the unchanged AF2 config from `configs/afab/AF2_yolo26n_chaotic_amplitude.yaml`, and use the unchanged CPE0 projection dimension 128, temperature 0.2, and all-positive IoU threshold 0.0 from `configs/fsce_cpe/CPE0_all_positive.yaml`. Both contain the same AF2+CPE wrapper and projection parameters.

| Arm | Only intentional difference |
|---|---|
| `AF2CPE0` matched control | `cpe.loss_weight: 0.0` |
| `AF2CPE5` candidate | `cpe.loss_weight: 0.5` |

Training is frozen at seed 42, 50 epochs, image size 640, batch 16, workers 2, patience 15, optimizer `auto`, deterministic mode, no cache, close-mosaic 10, and max-det 500. No parameter may be tuned after seeing results.

## Mandatory static gate

The static audit follows the established AF2-FFA precedent. Because separate full AF2 CUDA forwards include FFT kernels, bitwise equality is not required across separate end-to-end forwards. Instead, before training the audit must establish:

- the AF2 native Detect-head state is transferred bitwise into each CPE wrapper;
- wrapper-versus-native Detect output is bitwise identical when the same head receives the same feature tensors;
- separate full-model AF2 versus AF2+CPE evaluation is numerically consistent with maximum box/score absolute difference no larger than `1e-4`;
- the CPE projection is not called during evaluation/inference;
- projection gradient is exactly zero for `AF2CPE0` (`loss_weight=0`) and finite/nonzero for `AF2CPE5` (`loss_weight=0.5`);
- model YAML, AF2 config, training schedule, parameter count, and state schema are matched, with only CPE loss weight intentionally different.

Training is unauthorized unless the audit records `PASS` and the exact AF2 checkpoint hash.

## Execution pattern

The Colab notebook follows the established AF2-FFA/STB experiment pattern: frozen setup, static audit as a separate cell, then each arm is trained/resumed separately with `run_faruq_v3_af2_cpe_arm`, logging directly to Drive. A completed result JSON is reused; a partial run resumes from `last.pt`. The notebook does not use an additional all-in-one worker layer.

## Validation decision gates

All deltas are candidate minus matched control in absolute mAP50-95 units (0.002 = 0.20 percentage points).

- Safety: Macro >= -0.002; Bottom-3 >= -0.010; Worst >= -0.010.
- Superiority route: Macro >= +0.002 and both Bottom-3 and Worst are preserved (delta >= 0).
- Tail-Pareto route: Macro >= -0.001, Bottom-3 >= +0.005, Worst >= +0.010.
- `RETAIN` requires every safety gate and either superiority route or tail-Pareto route. Otherwise `REJECT`.

Only validation is evaluated. A retained seed-42 screen is evidence to plan confirmation, not locked-test evidence.
