# Faruq-v3 FMH1 Focal-Modulation Result

Status: completed with a negative seed-42 validation result. Test remained
locked. Date: 2026-08-13.

## Result

| Model | Macro mAP50-95 | Bottom-3 | Worst-class |
|---|---:|---:|---:|
| STB1 frozen reference | 88.67% | 83.64% | 80.81% |
| FCT0 optimization control | 89.40% | 84.83% | 84.15% |
| FMH1 | 87.60% | 79.40% | 78.99% |

### FMH1 minus STB1

- Macro: -1.07 points
- Bottom-3: -4.24 points
- Worst-class: -1.82 points

### FMH1 minus FCT0

- Macro: -1.80 points
- Bottom-3: -5.43 points
- Worst-class: -5.16 points

Both frozen comparisons failed every acceptance criterion. FMH1 therefore
does not authorize a capacity control, additional seeds, or test evaluation.

## Interpretation

The official-style local-to-global focal-modulation operator was wired only to
the P3/P4/P5 classification paths and began bitwise-identically to D0; active
modulation preserved raw box outputs in the static audit. The negative result
therefore cannot be dismissed as accidental localization-path rewiring.

On this dataset and schedule, focal modulation did not improve the retained
STB representation. The disproportionate Bottom-3 and Worst-class losses show
that its contextual aggregation was especially unhelpful for the weakest fine-
grained classes. This rejects FMH1 as a candidate; it does not establish a
universal claim against FocalNet or focal modulation on other datasets.

FMH1 had 4,621,195 total parameters, versus 4,589,201 reported for STB1. It
therefore also provides no accuracy justification for its slightly higher
capacity. No latency claim was tested.

## Frozen decision

`FAIL — STOP_FMH1_WITHOUT_TEST_OR_EXTRA_SEEDS`

Authoritative raw report:

`experiments/faruq-v3-focal-modulation-v1/val_reports/fmh1_seed42_decision.json`
