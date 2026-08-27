# Faruq-v3 AF2 Box-Score Factorial Result

Date: 27 August 2026.

Decision: **AF2_BOX_SCORE_INTERACTION_NECESSARY**.

The validation-only factorial completed without training or test access. Both
pure endpoints reproduced their frozen historical metrics exactly; all static,
alignment, ontology, and evaluation gates passed.

| Arm | Box source | Score source | Macro | Bottom-3 | Worst |
|---|---|---|---:|---:|---:|
| DD | D0FT | D0FT | 86.69% | 74.98% | 72.02% |
| DA | D0FT | AF2 | 88.08% | 79.61% | 78.73% |
| AD | AF2 | D0FT | 86.60% | 75.00% | 72.47% |
| AA | AF2 | AF2 | **88.20%** | **80.04%** | **79.35%** |

AF2 scores supplied the dominant improvement: DA exceeded DD by 1.39 Macro,
4.63 Bottom-3, and 6.71 Worst-class points. AF2 regression was nevertheless
not disposable. Relative to AA, DA lost 0.12 Macro, 0.44 Bottom-3, and 0.62
Worst-class points. Conversely, AF2 boxes with D0FT scores did not improve
Macro and only modestly changed the tail.

The supported interpretation is therefore classification-dominant but
box-score-interaction-dependent. Naively replacing AF2 regression with D0FT
regression is rejected. Full AF2 remains the lead model; any later mechanism
must preserve its joint box-score path.

Raw Drive artifact:
`experiments/faruq-v3-af2-box-score-factorial-v1/af2_box_score_factorial.json`.

