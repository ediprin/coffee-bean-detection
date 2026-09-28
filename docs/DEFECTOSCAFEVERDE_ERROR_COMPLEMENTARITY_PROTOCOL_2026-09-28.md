# DefectosCafeVerde Object-Level Error Complementarity Protocol

Frozen: **2026-09-28**

## Why this audit precedes another model

The paper-backed, physical-grouped DefectosCafeVerde seed-42 screens do not
support another guessed frequency architecture. The matched native detector is
the current validation leader. AF2DIRECT, DCWCF1, LIFRPF1, and RAFC1 all finish
below their relevant native control on at least Macro, Bottom-3, and Worst-class
mAP50-95.

Classwise AP nevertheless shows specialization. For example, `agrio` improves
under every candidate and `triangulo` improves modestly, while `negro` is the
native worst class and declines under every candidate. Classwise AP deltas do
not establish that two models correct different physical objects. This audit
therefore measures object-level overlap before any fusion, router, or new
training is proposed.

## Frozen endpoints

All endpoints are completed seed-42, 50-epoch validation checkpoints using
`defectoscafeverde-grouped-physical-v1` and the same frozen dataset-audit SHA.

| Endpoint | Mechanism |
|---|---|
| `D0DIRECT` | native RGB YOLO26n; reference |
| `AF2DIRECT` | AF2 input frontend |
| `DCWCF1` | chromatic-wavelet classification conditioning |
| `LIFRPF1` | luminance residual classification conditioning |
| `RAFC1` | training-only raw/Fourier classification consistency |

The exact result JSON and checkpoint SHA for every endpoint must agree. Test
must be absent from the extracted runtime root.

## Unit of analysis

The unit is one annotated validation object, identified by image path and GT
index. Each checkpoint is evaluated independently on the same 808 validation
images. Two diagnostic stages are recorded:

1. `raw_top500`: top 500 one-to-one candidates, matched class-agnostically at
   IoU 0.50;
2. `final_conf0001`: native final predictions with confidence at least 0.001,
   matched class-agnostically in confidence order at IoU 0.50.

For every object and endpoint the audit records accessibility, unique match,
predicted class, correctness, confidence, and IoU. It reports:

- correct-set intersections, unions, and Jaccard overlap;
- candidate-only rescues of D0 errors;
- D0-correct objects lost by each candidate;
- union correct-decision recall and remaining shared errors;
- the same quantities per class, including native worst-class `negro`.

The oracle union uses the ground-truth label to ask whether *any* completed
endpoint knows the correct answer. It is not a deployable ensemble and is not
reported as mAP.

## Frozen interpretation gate

- Material information complementarity requires at least **+1 percentage
  point** final-stage union correct-decision recall over `D0DIRECT`.
- General-repair relevance additionally requires at least one candidate-only
  rescue in the native worst class.

Possible decisions:

1. `NO_MATERIAL_COMPLEMENTARY_HEADROOM`: stop fusion and retain D0DIRECT.
2. `CLASS_SELECTIVE_COMPLEMENTARITY_WITHOUT_WORST_CLASS_REPAIR`: do not claim a
   general repair; inspect only the rescued class subset.
3. `COMPLEMENTARY_INFORMATION_PRESENT_ROUTER_AUDIT_ONLY`: freeze a separate
   train-only routability audit. This does **not** authorize model training.

No threshold is changed after viewing the output.

## Locks and claim boundary

- Training: forbidden.
- Test extraction or evaluation: forbidden.
- Weight averaging, NMS fusion, score calibration, or router fitting: forbidden.
- Validation predictions are descriptive diagnostic evidence, not a final
  superiority test.
- A positive oracle only demonstrates available complementary information. A
  later train-only analysis must show that the correct expert can be selected
  without GT before an architecture is justified.

