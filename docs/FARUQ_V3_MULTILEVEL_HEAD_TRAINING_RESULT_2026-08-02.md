# Faruq-v3 Capacity-Matched Multilevel Head Training Result

Date: 2026-08-02

Decision: **FAIL — stop without test or additional seeds**

The experiment evaluated seed 42 on Faruq-v3 validation only. Test remained
unavailable and locked.

## Result

| Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| D0 | 79.97% | 68.72% | 65.09% |
| MHC0, P5 capacity control | 73.01% | 59.55% | 58.10% |
| MHF1, P3+P4+P5 fusion | 78.31% | 64.02% | 61.81% |

MHF1 versus MHC0:

- Macro: **+5.30 points**;
- bottom-3: **+4.47 points**;
- worst-class: **+3.71 points**.

MHF1 versus D0:

- Macro: **-1.65 points**;
- bottom-3: **-4.70 points**;
- worst-class: **-3.28 points**.

MHF1 passed every capacity-control criterion and failed every D0 criterion.
Therefore multilevel fusion contains useful signal relative to the matched
P5-only branch, but the integrated end-to-end training configuration did not
preserve the stronger native detector.

## Resume deviation

The first MHF1 process terminated with `SIGSEGV` after recording 45 epochs.
Resume produced one further epoch and then wrote a completion manifest; the
final evaluated checkpoint therefore came from a run with 46 recorded epochs,
not the frozen 50. This deviation is disclosed rather than treated as a
protocol-complete positive result. It does not reverse the FAIL decision,
because MHF1 missed D0 on all three frozen metrics.

## Interpretation boundary

This result supports only the narrow statement that P3+P4+P5 is better than
the capacity-matched P5 control under this head. It does not show improvement
over YOLO26 D0. No extra seed, test access, or continuation of this exact
training protocol is authorized.

Raw artifact:
`experiments/faruq-v3-multilevel-head-v1/val_reports/multilevel_head_seed42_decision.json`

## Post-result initialization audit

After this result was frozen, the FRM1 preservation audit exposed a loader
namespace issue in the original multilevel trainer. Wrapping native `Detect`
changes state keys such as `model.N.cv*` to `model.N.base_head.cv*`.
Ultralytics' generic partial loader did not explicitly remap those native head
keys when constructing MHC0/MHF1. Therefore the reported D0-versus-MHC0/MHF1
deltas are **not a strict identical-initialization causal comparison** and
must not be cited as one.

The MHF1-versus-MHC0 capacity-controlled comparison remains informative:
both candidates used the same wrapper, loading path, schedule, and parameter
schema, and fusion improved all three metrics relative to P5 control. The
formal FAIL against D0 is unchanged; confidence in attributing that gap solely
to multilevel fusion is reduced. The loader is now fixed for future runs by a
strict native-head transfer and a regression test. Historical artifacts are
not rewritten or silently rerun.
