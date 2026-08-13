# Faruq-v3 FC-STB Frequency Distillation Result

Date: 2026-08-13

Protocol: `faruq-v3-fcstb-frequency-distillation-v1`

Decision: **FAIL — stop without test or additional seeds**

The experiment tested whether a frozen AF2 teacher could transfer complementary
frequency-sensitive class decisions into a single STB detector. FCT0 and FCD1
used the same STB1 initialization, parameter schema, trainable parameters,
optimizer, schedule, and 20-epoch seed-42 validation protocol. The only intended
difference was the ground-truth-bounded AF2 distillation loss in FCD1.

## Completed gates

The static audit passed for both candidates:

- strict STB checkpoint transfer;
- identical 4,589,201-parameter student schema and 2,182,937 trainable
  parameters;
- raw boxes and scores identical to STB1 at zero initialization;
- backbone, neck, and box heads frozen as declared;
- AF2 teacher absent from the serialized student;
- single-forward student inference with no teacher-time dependency.

The validation-only teacher-headroom diagnostic also passed. On the same 526
validation targets:

| Diagnostic | Count | Fraction |
|---|---:|---:|
| STB1 correct | 304 | 57.79% |
| AF2 correct | 351 | 66.73% |
| AF2 exclusively rescued STB1 | 86 | 16.35% |
| AF2 regressed a target correct under STB1 | 39 | 7.41% |
| Both correct | 265 | 50.38% |
| Both wrong or missed | 136 | 25.86% |

AF2 rescues covered 19 classes, comfortably exceeding the frozen minimum of
1% of targets across at least three classes. Teacher complementarity therefore
existed before training; the final negative result is not explained by an
empty teacher-headroom gate.

## Seed-42 validation result

| Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| STB1 frozen reference | 88.67% | 83.64% | 80.81% |
| FCT0 optimization control | **89.40%** | **84.83%** | **84.15%** |
| FCD1 AF2 distillation | 89.19% | 83.24% | 79.16% |

### FCT0 minus STB1

- Macro: **+0.73 points**
- Bottom-3: **+1.19 points**
- Worst class: **+3.34 points**

### FCD1 minus STB1

- Macro: **+0.52 points**
- Bottom-3: **-0.40 points**
- Worst class: **-1.65 points**

FCD1 met only the Macro-gain criterion. It failed bottom-3 preservation and
exceeded the permitted one-point worst-class decline.

### FCD1 minus FCT0

- Macro: **-0.21 points**
- Bottom-3: **-1.59 points**
- Worst class: **-4.99 points**

FCD1 failed every criterion against its optimization-matched control.

## Interpretation

The positive change over the frozen STB1 reference is attributable to the
controlled continuation/fine-tuning procedure represented by FCT0, not to AF2
distillation. Although AF2 made complementary validation decisions, copying
its softened class distribution under the frozen GT-bounded KL rule degraded
the lower tail. This result rejects the specific FC-STB transfer mechanism; it
does not establish that all teacher-student transfer or all frequency features
are ineffective.

FCT0 is a useful optimization-control observation, not an FC-STB architectural
contribution and not a confirmed multi-seed result. It must not be promoted to
a final model from this screening alone.

## Research boundary

- Validation only; test images were not accessed.
- Seed 42 only.
- The frozen protocol requires stopping FC-STB after this failure.
- Do not run seeds 123/2026 or open test for FCD1.
- Preserve FCT0 and FCD1 checkpoints as audit evidence of the controlled
  comparison.
- `training_executed_this_call: false` in the final decision report means the
  reporting invocation reused completed artifacts; both candidates had already
  completed all 20 epochs.

## Raw artifacts

- `experiments/faruq-v3-fcstb-distillation-v1/static_audit.json`
- `experiments/faruq-v3-fcstb-distillation-v1/val_reports/frequency_teacher_headroom_seed42.json`
- `experiments/faruq-v3-fcstb-distillation-v1/val_reports/FCT0_seed42_val.json`
- `experiments/faruq-v3-fcstb-distillation-v1/val_reports/FCD1_seed42_val.json`
- `experiments/faruq-v3-fcstb-distillation-v1/val_reports/fcstb_seed42_decision.json`
- `experiments/faruq-v3-fcstb-distillation-v1/FCT0_seed42/results.csv`
- `experiments/faruq-v3-fcstb-distillation-v1/FCD1_seed42/results.csv`
