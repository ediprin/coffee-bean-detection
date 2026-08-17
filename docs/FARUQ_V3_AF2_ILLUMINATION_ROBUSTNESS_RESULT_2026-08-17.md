# Faruq-v3 AF2 Controlled Illumination Robustness Result

Date: 2026-08-17

Decision: **FAIL — stop after seed 42; no confirmation seeds and no test
access.**

AF2 retained its ordinary clean-image advantage, but the advantage did not
generalize across isolated photometric stresses. Positive mean Macro was
driven by warm and especially cool color-temperature shifts rather than broad
illumination stability.

| Metric | Mean robustness advantage | Minimum | Positive conditions |
|---|---:|---:|---:|
| Macro mAP50-95 | +1.74 points | -2.82 | 2/9 |
| Bottom-3 | +1.24 points | -6.70 | 3/9 |
| Worst class | -2.32 points | -10.23 | 2/9 |

| Condition | Macro advantage | Bottom-3 advantage | Worst advantage |
|---|---:|---:|---:|
| dark -0.5 EV | -1.36 | -1.75 | -5.41 |
| dark -1.0 EV | -1.62 | -0.35 | -5.20 |
| bright +0.5 EV | -0.01 | -6.52 | -10.23 |
| bright +1.0 EV | -1.10 | -3.79 | -6.38 |
| contrast 0.75 | -1.55 | +0.79 | -3.92 |
| contrast 1.25 | -0.35 | -6.70 | -10.07 |
| warm | +5.66 | +7.38 | +5.38 |
| cool | +18.79 | +27.31 | +23.49 |
| localized shadow | -2.82 | -5.21 | -8.50 |

Only four of six frozen criteria passed. Macro was positive in only 2/9
conditions and mean Worst robustness was below the -1 point tolerance. The
result rejects a general illumination-robustness claim for fixed AF2; it does
not change AF2's previously confirmed in-domain and Coffee Standard results.

Authoritative Drive report:
`experiments/faruq-v3-af2-illumination-v1/illumination_screen_seed42.json`.
