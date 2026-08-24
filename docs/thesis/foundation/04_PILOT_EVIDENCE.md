# 04 — AF2-Direct Pilot Evidence

## 1. Status

This file records preliminary feasibility evidence for proposal preparation.

**Do not use this file to make a final superiority claim.**

Current status:

```text
AF2-direct seed 42 = strong screening result / pilot evidence
final multi-seed validation = not yet established here
```

## 2. Direct-training question

The pilot tests whether AF2 can be active from the beginning of coffee fine-tuning rather than only after a native coffee-trained checkpoint.

Conceptual arms:

```text
D0DIRECT:
official yolo26n.pt -> native coffee training, 50 epochs

AF2DIRECT:
official yolo26n.pt -> AF2 active from epoch 1 -> coffee training, 50 epochs
```

The conceptual importance is that a positive direct result weakens the interpretation that AF2 requires a previously adapted D0 checkpoint before it can be useful.

## 3. Seed-42 screening result

User-verified completed run (50/50 epochs):

| Metric | D0DIRECT | AF2DIRECT | Delta AF2-D0 |
|---|---:|---:|---:|
| Macro mAP50-95 | 79.902% | 80.789% | +0.886 pp |
| Bottom-3 mAP50-95 | 64.569% | 69.582% | +5.013 pp |
| Worst-class mAP50-95 | 61.769% | 66.953% | +5.184 pp |
| Raw top-500 proposal accessibility | 99.810% | 99.810% | +0.000 pp |
| Localization-conditioned Top-1 | 57.029% | 67.188% | +10.158 pp |
| Correct-decision recall | 40.875% | 57.224% | +16.350 pp |

Reported final AF2DIRECT seed-42 summary:

```text
50 epochs completed in 1.133 hours.

DONE: AF2DIRECT 42
macro_map50_95             = 0.8078862414
bottom3_class_map50_95     = 0.6958183582
worst_class_map50_95       = 0.6695284016
```

Screen decision reported by the experiment artifact:

```text
localization_safe           = true
route_a_direct_overall_gain = true
route_b_lower_tail_pareto   = true
decision                    = PROMOTE_TO_3_SEED
test_images_accessed        = false
```

## 4. Pilot interpretation

The strongest pattern is not the +0.886 pp Macro gain by itself.

The lower tail improves by roughly +5 pp while raw proposal accessibility is unchanged:

```text
Raw proposal accessibility:
99.810% -> 99.810%

Localization-conditioned Top-1:
57.029% -> 67.188%

Correct-decision recall:
40.875% -> 57.224%
```

Therefore the seed-42 pilot is **consistent with** the working hypothesis:

```text
raw localization accessibility remains stable
while class discrimination / decision quality improves
```

The wording "consistent with" is mandatory. The diagnostic is evidence about the observed run; it is not causal proof of the internal mechanism.

## 5. Why this matters for the proposal

The result supports feasibility of a simpler thesis formulation:

```text
Input -> AF2 -> YOLO26
```

rather than making staged `D0 -> AF2` training the centerpiece of the thesis.

This is methodologically useful because the preprocessing can be defined as part of the input pipeline from the start and evaluated against a matched native YOLO26 control.

## 6. What may be stated in the proposal

Safe wording:

> A preliminary seed-42 study with matched 50-epoch direct training indicated a +0.886 percentage-point improvement in Macro mAP50-95 and approximately +5 percentage points in both Bottom-3 and Worst-class mAP50-95. Raw top-500 proposal accessibility remained unchanged at 99.810%, while localization-conditioned classification improved. These findings are treated as preliminary feasibility evidence and not as the final conclusion of the thesis.

Unsafe wording:

> AF2 has been proven to outperform YOLO26.

or

> AF2 definitively improves classification without affecting localization.

## 7. Relationship to earlier staged evidence

Earlier repository experiments showed positive AF2 results when the AF2 stage started from a native D0 checkpoint. Those results remain useful as experiment genealogy and secondary evidence, but they should not determine the main thesis narrative if direct AF2 remains viable.

Do not present the thesis as a chronology of repository experiments.

## 8. Proposal-phase freeze

No additional experiment is required **for the purpose of beginning proposal writing**.

Future experiments may still be needed for the final thesis, including multi-seed confirmation and selected controls/ablations, but proposal drafting should proceed from the current evidence base rather than delaying the document for exploratory module stacking.
