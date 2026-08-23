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

Before training, the audit must establish: identical native outputs for wrappers with shared weights; zero projection gradient at weight 0 and finite nonzero projection gradient at weight 0.5; and zero CPE-projection calls during evaluation/inference. Training is unauthorized unless the audit records `PASS` and the exact AF2 checkpoint hash.

## Validation decision gates

All deltas are candidate minus matched control in absolute mAP50-95 units (0.002 = 0.20 percentage points).

- Safety: Macro >= -0.002; Bottom-3 >= -0.010; Worst >= -0.010.
- Superiority route: Macro >= +0.002 and both Bottom-3 and Worst are preserved (delta >= 0).
- Tail-Pareto route: Macro >= -0.001, Bottom-3 >= +0.005, Worst >= +0.010.
- `RETAIN` requires every safety gate and either superiority route or tail-Pareto route. Otherwise `REJECT`.

Only validation is evaluated. A retained seed-42 screen is evidence to plan confirmation, not locked-test evidence.
