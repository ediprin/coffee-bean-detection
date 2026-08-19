# Faruq-v3 WAV_L1 Paired Multiseed Confirmation Protocol

Date frozen: 2026-08-19  
Status: **frozen before seed-123/2026 WAV_L1 training**

Branch: `agent/wav1-mechanism-factorization`  
Dataset: Faruq-v3 grouped development split only  
Seeds: **42, 123, 2026**  
Locked test: **closed / not accessed**

## Confirmatory question

Stage-1 mechanism factorization selected `WAV_L1` because level-1 Haar detail alone reproduced essentially the full seed-42 WAV1 lower-tail gain. The new confirmatory question is:

> Does the frozen single-level `WAV_L1` detail gate remain validation-robust versus seed-matched D0FT across seeds 42/123/2026?

This protocol confirms the selected mechanism. It does not reopen the Stage-1 search, retune the operator, or test a new family of alternatives.

## Frozen seed-42 observation

Seed 42 is reused and **must not be retrained**. Machine-readable evidence is frozen at:

`docs/evidence/FARUQ_V3_WAV_L1_SEED42_RESULT_2026-08-19.json`

Headline validation metrics:

- Macro mAP50-95: **0.885720537714217**
- Bottom-3 mAP50-95: **0.8399334705085897**
- Worst-class mAP50-95: **0.8209474694929713**

Seed-42 D0FT reference:

- Macro: **0.8668870418312263**
- Bottom-3: **0.7498085237045921**
- Worst: **0.7202242739437643**

The Stage-1 selection record remains `docs/FARUQ_V3_WAV1_MECHANISM_FACTORIZATION_STAGE1_RESULT_2026-08-19.md`.

## Frozen implementation

`WAV_L1` remains exactly the Stage-1 implementation:

1. RGB is converted to Rec.709 luminance
   \[
   Y=0.2126R+0.7152G+0.0722B.
   \]
2. One orthonormal Haar DWT is applied.
3. The three level-1 detail bands are collapsed into
   \[
   D_1=\sqrt{LH_1^2+HL_1^2+HH_1^2+\epsilon}.
   \]
4. `D1` is bilinearly resized to input resolution with the already-frozen interpolation rule.
5. The new-control numerical contract `stable_minmax_spatial` maps numerically-flat cues to zero and otherwise applies ordinary spatial min-max normalization.
6. The cue is expanded identically across RGB and applied through the common residual gate
   \[
   x'=x+x\odot N(c(x)).
   \]
7. The frontend contains no learned parameters; native YOLO26n-P3 is otherwise unchanged.

Configuration is frozen at `configs/wav1_factorization/WAV_L1_yolo26n.yaml`.

No level, subband, kernel, normalization rule, gate, color conversion, architecture, augmentation, optimizer, or training hyperparameter may be changed after this protocol is frozen.

## Frozen training schedule

Only seeds **123** and **2026** are newly trained. Each starts from the existing seed-matched D0 checkpoint.

The schedule remains the Stage-1 WAV_L1 schedule:

- epochs: 50
- imgsz: 640
- batch: 16
- workers: 2
- patience: 15
- optimizer: auto
- pretrained: false
- cache: false
- close_mosaic: 10
- max_det: 500
- save_period: 1
- deterministic seed-specific training

D0FT controls are reused from the existing AF2/IGEM paired-confirmation evidence; they are not retrained.

## Primary confirmation gate

For comparability with the already-completed WAV1, AF2, and IGEM1 paired validation confirmations, the primary gate is frozen to the same five criteria against seed-matched D0FT:

1. Mean paired Macro gain >= **+0.5 percentage point**.
2. Macro improves versus D0FT in at least **2/3 seeds**.
3. Mean Bottom-3 versus D0FT is **not lower**.
4. Bottom-3 improves versus D0FT in at least **2/3 seeds**.
5. Mean Worst-class decline versus D0FT is no greater than **1.0 percentage point**.

All five criteria must pass.

This gate tests validation robustness of `WAV_L1`; it is not a superiority test against WAV1.

## Required reporting

Report for D0FT and WAV_L1:

- each seed's Macro, Bottom-3, and Worst-class mAP50-95;
- paired delta for each seed;
- three-seed mean and sample standard deviation;
- mean paired delta and its sample standard deviation;
- number of improved seeds for Macro and Bottom-3;
- the five frozen gate criteria and final PASS/FAIL.

After the primary decision, the completed two-level WAV1 three-seed aggregate may be shown **descriptively** as context. No post-hoc claim that WAV_L1 is generally superior to WAV1 is allowed merely from aggregate ordering.

## Stopping rule

- **PASS:** `WAV_L1` may be described as validation-robust versus D0FT across the frozen three seeds. Proceed only through a newly frozen next-step protocol, such as placement/error-decomposition work.
- **FAIL:** stop `WAV_L1` as the simplified confirmed mechanism. Do not retune, add a fourth seed, restore level-2 fusion, or open the locked test under this protocol.

Regardless of PASS/FAIL, preserve all per-seed results. Seed-42 selection evidence is not rewritten.

## Test lock

The Faruq locked test is not restored, read, evaluated, or used for selection in this protocol.
