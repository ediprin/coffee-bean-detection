# AF2 × D-FINE-N single-GPU execution profile — 2026-08-29

## Status

**FROZEN BEFORE TRAINING.** This execution profile supplements `FARUQ_V3_AF2_DFINE_N_TRANSFER_PROTOCOL_2026-08-29.md` for a single 16-GB-class Kaggle GPU. No scientific result existed when this profile was defined.

## Why an execution profile is needed

The official D-FINE-N custom configuration sets `total_batch_size: 128`. The official D-FINE training example launches custom training on four GPUs. On a single Kaggle GPU, using a global batch of 128 would also make AF2's patchwise FFT memory footprint unnecessarily risky.

D-FINE's official README explicitly documents a linear-scaling procedure when batch size changes: learning rates scale with batch size, while EMA decay and warm-up durations are adjusted according to the corresponding batch ratio.

This experiment therefore freezes one hardware-adapted recipe **before either arm is trained**. It is not claimed to reproduce the official COCO/custom training recipe exactly. Its only scientific purpose is a paired within-D-FINE AF2 transfer test.

## Frozen single-GPU profile

Reference custom D-FINE-N batch:

```text
B_ref = 128
```

Screen batch:

```text
B_run = 16
s = B_run / B_ref = 1/8
```

Both `DFN0` and `DFN_AF2` use:

```text
train total_batch_size = 16
val total_batch_size   = 16
num_workers            = 2
maximum epochs         = 220
AMP                    = enabled
seed                   = 42
```

The official D-FINE-N custom optimizer values are scaled by `s=1/8`:

```text
default AdamW lr        : 0.0008  -> 0.0001
backbone lr             : 0.0004  -> 0.00005
backbone norm/bn lr     : 0.0004  -> 0.00005
weight decay            : unchanged
betas                    : unchanged
```

To approximately preserve warm-up exposure in samples when the number of optimizer steps per epoch rises by eight times:

```text
lr warmup_duration : 500  -> 4000 steps
EMA warmups        : 1000 -> 8000 steps
```

Following the inverse application of the D-FINE README EMA adjustment rule:

```text
1 - decay_run = (1 - decay_ref) * s

decay_ref = 0.9999
s         = 1/8

decay_run = 0.9999875
```

Thus:

```text
EMA decay = 0.9999875
```

No parameter in this execution profile may differ between the native and AF2 arms.

## Important interpretation boundary

Because this is a hardware-adapted fine-tuning recipe, absolute Faruq-v3 performance of `DFN0` must not be presented as an official D-FINE-N benchmark. The valid comparison is the paired delta:

```text
DFN_AF2 - DFN0
```

If a later environment can reproduce the official global batch schedule, that is a separate confirmation experiment and must not silently replace this frozen seed-42 screen.
