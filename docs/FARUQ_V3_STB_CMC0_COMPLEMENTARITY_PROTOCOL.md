# Faruq-v3 STB1–CMC0 Object-Level Complementarity Audit

Status: **post-training diagnostic only**. No retraining, no hyperparameter selection, no test access.

## Question

The paired three-seed capacity control rejected the causal claim that STB's shifted-window spatial interaction provides a sufficiently large and stable Macro advantage over the capacity-near-matched non-spatial CMC0 control. The next question is therefore not whether to stack another module, but whether STB and CMC0 make materially different errors on the same validation objects.

This audit asks:

1. When CMC0 is wrong, how often does STB1 rescue the same GT object?
2. When STB1 is wrong, how often does CMC0 rescue it?
3. How much of their error set overlaps?
4. What is the oracle upper bound if a hypothetical selector always chose the correct arm?
5. Which directional confusion pairs are rescued by the other arm?
6. Are the seed-dependent AP changes associated with different object-level rescue patterns?

## Frozen data boundary

- Faruq-v3 grouped development data.
- Validation split only.
- A data root exposing `test` is rejected.
- The existing `stb_capacity_paired_confirmation.json` is required and must state `evaluation_split=val`, `test_images_accessed=false`, and `test_opened=false`.
- Seeds are frozen to 42, 123, and 2026.

## Object matching

Final detections are produced with the same fixed inference settings for both arms:

- image size: 640
- confidence threshold: 0.25
- prediction NMS IoU: 0.70
- maximum detections: 500

Predictions are matched to GT boxes **class-agnostically** at IoU >= 0.50, in descending prediction-confidence order. Each GT can be matched once.

A GT target is counted as `correct` only if a final detection is matched at IoU >= 0.50 **and** its predicted class equals the GT class.

These object-level quantities are diagnostics. They are **not mAP** and do not replace Macro / Bottom-3 / Worst AP from the frozen evaluation protocol.

## Primary outputs

For every seed and every GT target, the audit writes:

- image and GT identifier;
- GT class;
- CMC0 accessibility, matched state, predicted class, confidence, IoU, correctness;
- STB1 equivalent fields;
- directional rescue flags.

The summary reports:

### Directional rescue

For models A and B,

`R(A->B) = P(B correct | A wrong)`.

Both overall rescue and classification-only rescue are reported. Classification-only rescue requires both models to have matched the same GT at IoU >= 0.50; one then classifies it incorrectly while the other classifies it correctly.

### Shared-error Jaccard

Let `E_A` and `E_B` be GT targets not correctly handled by A and B. The audit reports

`J(E_A,E_B) = |E_A intersect E_B| / |E_A union E_B|`.

High Jaccard suggests similar failure sets, but it is **not** a feature-space similarity measure.

### Oracle headroom

The oracle counts a GT as correct if either CMC0 or STB1 is correct. The reported `oracle_gain_over_best_model` is an **upper bound** on selection/fusion headroom, not the result of an implemented fusion method.

### Confusion-pair rescue

For targets where one model is matched but assigns the wrong class and the other model is correct, the audit ranks directional `(GT class -> wrong predicted class)` rescue counts.

## Interpretation gates

This audit does not select a new model automatically. Use the following interpretation rules:

- **High error overlap + low oracle gain:** STB1 and CMC0 are operationally redundant on validation errors. Do not pursue naive fusion.
- **Low/moderate error overlap + material oracle gain:** there is measurable complementary decision information; a selector/gate may be worth studying.
- **Complementarity concentrated in specific classes/confusion pairs:** prefer class-conditional or confusion-conditional routing over global fusion.
- **Large seed-to-seed changes in rescue direction:** treat the apparent complementarity as optimization-trajectory dependent; do not make a stable mechanistic claim yet.

No threshold for 'material' oracle gain is frozen in this first diagnostic because the audit is descriptive rather than a model-selection gate. Any later architectural gate must be frozen in a separate protocol before training.

## Explicit non-goals

Version 1 does **not** compute entropy, top-1/top-2 margin, feature CKA, or representation similarity. Final Ultralytics detections expose the selected class confidence, not the full class distribution needed for faithful entropy/margin analysis. CKA requires a separate feature-hook protocol so feature extraction choices are frozen before looking at results.

## Run

```bash
python -m coffee_detector.analysis.stb_cmc0_complementarity \
  --data-root /path/to/faruq-development-v3-grouped \
  --paired-summary /path/to/stb_capacity_paired_confirmation.json \
  --output-root /path/to/faruq-v3-stb-cmc0-complementarity-v1 \
  --cmc0-seed42 /path/to/CMC0_seed42/weights/best.pt \
  --cmc0-seed123 /path/to/CMC0_seed123/weights/best.pt \
  --cmc0-seed2026 /path/to/CMC0_seed2026/weights/best.pt \
  --stb-seed42 /path/to/STB1_seed42/weights/best.pt \
  --stb-seed123 /path/to/STB1_seed123/weights/best.pt \
  --stb-seed2026 /path/to/STB1_seed2026/weights/best.pt \
  --device 0
```
