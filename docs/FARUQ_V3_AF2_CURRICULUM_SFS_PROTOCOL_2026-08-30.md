# Faruq-v3 AF2 Curriculum-SFS Seed-42 Protocol — 2026-08-30

## Status

**FROZEN BEFORE TRAINING.** Validation-only seed-42 screen. Test remains
closed. Only `AF2CURR1` is trained; the completed matched `AF2CTRL` result is
reused after its parent-checkpoint SHA is verified.

## Motivation

The direct-from-pretrained `AF2SFSCUE1` arm preserved Macro but reduced
Bottom-3 by 4.02 points and Worst-class AP by 8.36 points. In contrast, the
same mechanism families were useful only after coffee-domain adaptation:
`AF2SFS1` improved matched-continuation Macro by 0.95 point, while `AF2CUE1`
improved the matched base by 0.42/3.12/6.05 points for
Macro/Bottom-3/Worst.

This motivates an optimization intervention rather than another spectral
frontend. Curriculum learning and progressive learning establish that the
difficulty/regularization presented to a model can be scheduled rather than
held fixed. Auxiliary-learning work further shows that an auxiliary objective
can cause negative transfer and should remain subordinate to the target task.

Primary references:

- Bengio et al., *Curriculum Learning*, ICML 2009;
- Tan and Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021;
- Song et al., *Learning With Privileged Tasks*, ICCV 2021;
- Chen et al., *Auxiliary Learning with Joint Task and Data Scheduling*, ICML
  2022;
- Du et al., *Adapting Auxiliary Losses Using Gradient Similarity*, 2018.

## Research question

Starting from the exact completed AF2 seed-42 parent, can a scheduled shared
P3 space-frequency selector plus target-prioritized AF2-gate supervision beat
an equal-budget AF2 continuation control without damaging the lower tail?

## Parent and control

- Parent: completed historical `AF2_seed42/weights/best.pt`.
- Control: completed `AF2CTRL_seed42_result.json` from the frozen AF2
  complementary-mechanisms study.
- The runner requires the control's `initial_af2_checkpoint_sha256` to equal
  the candidate parent's SHA-256.
- Both use seed 42, 30 maximum continuation epochs, image size 640, batch 16,
  patience 10, optimizer `auto`, and a fresh optimizer state.

This is a matched continuation comparison. The claim is `AF2CURR1 - AF2CTRL`,
never candidate versus the pre-continuation AF2 endpoint.

## Candidate

`AF2CURR1` retains the parameter-free AF2 input frontend. A shared P3
space-frequency selector is identity initialized and affects both native box
and classification branches. Training-only 1×1 decoders observe pre-selector
P3/P4/P5 features and predict only the normalized AF2 recovery gate.

The auxiliary gate loss is privileged: for each batch, its gradient is
compared with the native detection-loss gradient at shared P3. Its detached
scale is `max(cosine, 0)`. A negatively aligned auxiliary update therefore
contributes exactly zero, while the native detection objective is never
modified or projected.

The auxiliary decoders are inactive at inference. SFS remains active and adds
770 parameters (about 0.031% of the detector).

## Frozen curriculum

The zero-based 30-epoch schedule is:

| Epochs | Phase | SFS strength | Maximum auxiliary gain |
|---|---|---:|---:|
| 0–4 | coffee warm-up | 0 | 0 |
| 5–14 | cosine spectral ramp | 0 → 1 | 0 → 0.10 |
| 15–19 | joint hold | 1 | 0.10 |
| 20–29 | auxiliary release | 1 | cosine 0.10 → 0 |

At inference SFS strength is exactly one. No validation value selected any
boundary or gain.

## Static gates

Training is blocked unless the audit verifies:

1. parent is native-head AF2 and its AF2 configuration matches the candidate;
2. initial detector output equals the AF2 parent within the frozen CPU/GPU
   numerical tolerance;
3. schedule boundaries and zero/full states are exact;
4. positive gradient alignment passes and negative alignment is blocked;
5. SFS is active when enabled and auxiliary gradients are finite;
6. inference/training-only parameter counts are exact;
7. no ROI, decoded-box dependency, or test access exists.

## Seed-42 decision

Relative to matched `AF2CTRL`, retain through either prospectively frozen
route:

**Macro route:** Macro gain at least +0.50 point, Bottom-3 not lower, and
Worst-class drop no more than 1.00 point.

**Lower-tail Pareto route:** Macro drop no more than 0.10 point, Bottom-3 gain
at least +1.00 point, and Worst-class gain at least +1.00 point.

If neither route passes, stop without extra seed or test. A pass only permits a
separately frozen paired confirmation; it is not a final thesis or test claim.
