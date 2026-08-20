# Faruq-v3 CAFR seed42 screening result — 2026-08-21

## Scope

This record freezes the first development-screen result for the CAFR causal ladder on Faruq-v3 grouped development.

- Detector: native YOLO26n-p3.
- Initial checkpoint: the same D0 seed42 checkpoint for every arm.
- Seed: 42.
- Evaluation split: validation only.
- Locked test: not accessed.
- Arms: C1, C2, C3, C4, CAFR.
- These are development-screen results, not final thesis confirmation.

## Final seed42 results

| Arm | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 | Median latency (ms) |
|---|---:|---:|---:|---:|
| C1 | 0.8773064843 | 0.7850641429 | 0.7546112128 | 22.480357 |
| C2 | 0.8725375321 | 0.7953242063 | 0.7323446759 | 23.503039 |
| C3 | 0.8695792803 | 0.7768336594 | 0.7398509179 | 23.401217 |
| C4 | 0.8668495481 | 0.7933015723 | 0.7621216583 | 22.269550 |
| CAFR | 0.8668495481 | 0.7933015723 | 0.7621216583 | 30.596670 |

Reference parent AF2 seed42 from the prior confirmation evidence: Macro 0.8819734, Bottom-3 0.800428, Worst 0.793470 (rounded to the precision available in the existing confirmation record).

## Incremental observations

### C1 — luminance-guided shared RGB gate

C1 is below the AF2 parent on Macro, Bottom-3, and Worst-class. The hypothesis that replacing AF2's independent RGB processing with a luminance-derived shared chromaticity-preserving gate is beneficial is **not supported** by this seed42 screen.

Decision: **drop C1 as an optimization candidate**.

### C2 — fixed radial × directional decomposition

Relative to C1:

- Macro: -0.4769 pp
- Bottom-3: +1.0260 pp
- Worst: -2.2267 pp

There is a descriptive Bottom-3 signal, but the cumulative C2 arm is not better overall. Because C2 is built on top of C1, this screen does **not** isolate the causal effect of radial information against the original AF2 parent.

Decision: **not confirmed; only worth an isolated AF2+radial control if pursued further**.

### C3 — soft entropy-conditioned selection

Relative to C2:

- Macro: -0.2958 pp
- Bottom-3: -1.8491 pp
- Worst: +0.7506 pp

The soft-selection modification does not provide a convincing aggregate benefit in this cumulative ladder.

Decision: **provisionally drop soft selection**.

### C4 — unsigned 180-degree orientation representation

Relative to C3:

- Macro: -0.2730 pp
- Bottom-3: +1.6468 pp
- Worst: +2.2271 pp

This is the clearest tail-class signal in the cumulative ladder, but C4 inherits C1+C2+C3. It therefore cannot yet establish that orientation symmetry itself improves the original AF2 parent.

Decision: **candidate for an isolated AF2+orientation experiment**.

### CAFR — patch calibration

Training-label object-scale calibration produced:

- Q25 equivalent bbox side: 37.7279 px
- Median equivalent bbox side: 45.1642 px
- Q75 equivalent bbox side: 54.6017 px
- Candidate patch sizes: {16, 32, 64}
- Selected patch size: 32

C4 already used patch size 32, so final CAFR and C4 produced identical detection metrics. Under the frozen calibration rule, patch size 32 is therefore consistent with the coffee training-object scale, but this is **not evidence that 32 is globally optimal**.

Decision: **no further patch-size change is justified by this screen**.

## Main decision

The full CAFR cumulative redesign does **not** improve the original AF2 parent on seed42.

Relative to the prior AF2 seed42 reference, CAFR is approximately:

- Macro: -1.5124 pp
- Bottom-3: -0.7126 pp
- Worst: -3.1348 pp

Therefore:

1. **Do not promote full CAFR to seed123/2026.**
2. **Retain AF2 as the current frequency parent/best candidate.**
3. Drop the luminance/shared-gate modification.
4. Provisionally drop the soft-selection modification.
5. Treat radial information and especially 180-degree orientation symmetry only as isolated AF2-centered hypotheses, not as confirmed improvements.
6. Do not access the locked test.

## Latency caveat

The five arms were trained/evaluated in separate Colab runtimes. Their recorded median latency values are not a matched hardware/runtime benchmark. In particular, the 30.60 ms CAFR value must not be interpreted as a causal slowdown versus C4 because C4 and CAFR are functionally identical at inference once calibration selected patch size 32. Any efficiency claim requires re-benchmarking checkpoints on the same runtime, device, warmup, and iteration protocol.

## Scientific status

- AF2 parent: **retained**.
- Full CAFR: **rejected at seed42 development screening**.
- C1 shared luminance gate: **rejected**.
- C2 radial decomposition: **unresolved as an isolated effect**.
- C3 soft selection: **provisionally rejected**.
- C4 180-degree orientation symmetry: **unresolved but highest-priority isolated follow-up**.
- Patch calibration: **selected 32; no metric change versus C4**.
