# Faruq-v3 STB-guided WAV-L1 protocol

Date: 2026-08-20

Status: **FROZEN BEFORE S2 TRAINING**

Locked test: **CLOSED**

## Research question

Can the already-strong STB1 detector transfer fine-grained class knowledge into
an efficient WAV-L1 + native YOLO26n student **without carrying STB1 into
inference**?

The deployed S2 student remains exactly:

```text
RGB -> parameter-free WAV-L1 input modulation -> native YOLO26n -> detection
```

STB1 exists only during training. S3 AF2 robustness code is implemented in the
same framework for completeness, but S3 is **not authorized** by this protocol.

## Evidence trigger

Three completed results motivate this test:

1. WAV-L1 is already a validation-robust, parameter-free deployment candidate.
2. STB1 has the strongest aggregate lower-tail validation profile among the
   retained architectural models across the frozen three seeds. This does **not**
   mean STB1 dominates WAV-L1 on every individual seed; seed-42 is therefore a
   genuine transfer test rather than a guaranteed stronger-teacher case.
3. The earlier FC-STB experiment (`FARUQ_V3_FCSTB_DISTILLATION_RESULT_2026-08-13.md`)
   showed that **direct GT-bounded AF2 -> STB logit distillation failed**, even
   though the teacher had complementary decisions. Therefore this protocol does
   not repeat direct student-head KL.

The new hypothesis reverses the transfer direction (STB1 -> lightweight
WAV-L1 student) and changes the optimization path to a CrossKD-inspired
cross-head route so annotation supervision and teacher supervision do not hit
one student prediction head directly.

This is an adaptation inspired by Wang et al., CrossKD (CVPR 2024), not a claim
of reproducing their detector/configuration exactly.

## Frozen S2 architecture

Student initialization: seed-matched native D0 checkpoint.

Student frontend: frozen `WAV_L1` operator from the completed WAV1 mechanism
factorization study.

Student detector: native YOLO26n P3/P4/P5 end-to-end detector. All normal model
parameters are trainable, exactly as in the WAV-L1 parent experiment.

Teacher: completed STB1 seed-matched checkpoint, fully frozen and kept outside
the serialized student model.

### Cross-head path

YOLO26 one2one features are detached by design, so the gradient-bearing path is
explicitly attached to the student's non-detached one2many P3/P4/P5 features:

```text
student non-detached P3/P4/P5
    -> frozen STB1 W-MSA/SW-MSA classification blocks
    -> frozen STB1 one2one classification head
    -> cross-head class logits
```

Those cross-head logits mimic the frozen STB1 clean-image one2one class logits.
The student's own native classification heads receive the native YOLO ground
truth objective only.

Only positive anchors for which the teacher predicts the assigned GT class and
has at least the frozen minimum GT probability are distilled.

Frozen S2 values:

- temperature: `2.0`
- branch-local distillation coefficient: `0.50`
- minimum teacher GT probability: `0.10`
- epochs: `50`
- image size: `640`
- batch: `16`
- optimizer: `auto`
- close mosaic: `10`
- max detections: `500`

**Implementation clarification frozen before training:** because YOLO26 detaches
its one2one features, CrossKD is attached to the non-detached one2many branch.
The `0.50` coefficient is applied inside that classification branch and the
result then follows YOLO26's existing one2many/one2one weighting schedule. This
is part of the frozen implementation, not a post-result tuning choice.

No box/localization distillation is added.

## Deployment contract

STB1 must not appear in the student state dict/checkpoint. AF2 must not appear
in the S2 student. S2 parameter count and inference graph must equal WAV-L1.

Before training, the static audit must prove:

- STB1 teacher checkpoint type is valid;
- student uses WAV-L1;
- student parameter/state schema equals WAV-L1;
- clean inference is bitwise equal to a WAV-L1 control at D0 initialization;
- cross-head gradients reach synthetic student-side features while frozen
  teacher parameters receive no gradient;
- AF2 robustness view is parameter-free and shape preserving (code readiness
  only; not S2 execution);
- no test access.

## Stage A: seed 42 only

Reference is the frozen WAV-L1 seed-42 validation result:

- Macro mAP50-95: `0.885720537714217`
- Bottom-3 mAP50-95: `0.8399334705085897`
- Worst-class mAP50-95: `0.8209474694929713`

The S2 seed-42 candidate must satisfy **all retention criteria**:

- Macro delta vs WAV-L1 >= `-0.005`;
- Bottom-3 delta vs WAV-L1 >= `-0.010`;
- Worst delta vs WAV-L1 >= `-0.010`.

And at least **one frozen advancement signal**:

- Macro delta >= `+0.002`; or
- Bottom-3 delta >= `+0.005`; or
- Worst delta >= `+0.005`.

This is the gate already agreed before S2 training; it is not changed after the
result is observed.

### PASS

If Stage A passes, stop and freeze a new paired S2 confirmation protocol for
seeds 42/123/2026 before running additional training.

### FAIL

If Stage A fails, stop this S2 mechanism. Do not sweep temperature, KD weight,
teacher threshold, attachment point, or a fourth seed on the same validation
result. Do not open test.

## S3 robustness stage — implemented but blocked

The codebase also contains `crosskd_af2` mode for the later final framework:

```text
clean RGB -> WAV-L1 -> student
AF2(clean RGB) -> WAV-L1 -> same student
```

The shifted AF2 view receives the same detection labels plus clean-to-AF2
positive-anchor class consistency. STB1 cross-head guidance remains training
only.

However **no S3 training is authorized here**. S3 requires S2 to pass Stage A,
then pass the separately frozen multiseed confirmation, and then receive its
own robustness protocol. AF2 is not yet claimed to be a domain-generalization
mechanism.

## Prohibited actions under this protocol

- locked-test evaluation;
- S3 training before S2 confirmation;
- post-result KD hyperparameter tuning;
- STB/WAV/AF2 inference fusion;
- claiming CrossKD reproduction rather than a CrossKD-inspired adaptation;
- claiming robustness from seed stability alone.
