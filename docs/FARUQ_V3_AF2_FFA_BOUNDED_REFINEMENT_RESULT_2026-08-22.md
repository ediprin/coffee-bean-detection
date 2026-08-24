# AF2-FFA bounded refinement result

Status: completed but causally invalid because of an optimization confound.

## Result

| Arm | Macro mAP50-95 | Bottom-3 | Worst |
|---|---:|---:|---:|
| AF2FFA0 | 88.89% | 80.84% | 77.53% |
| AF2FFA1 | 88.56% | 81.66% | 80.40% |
| AF2FFAB1 | 85.09% | 73.33% | 66.61% |

AF2FFAB1 lost 3.80/7.51/10.92 percentage points against AF2FFA0 and
3.47/8.33/13.80 points against AF2FFA1 for Macro/Bottom-3/Worst. The frozen
decision consequently returned `REJECT`; test was not accessed.

## Confound discovered after completion

AF2FFA1 uses `amplitude = alpha`, whose derivative at initialization is one.
AF2FFAB1 used `amplitude = 0.10*tanh(alpha)`, whose derivative at
initialization is 0.10. The study therefore changed both the amplitude bound
and the effective initial adapter learning rate by a factor of ten.

The result is retained as an auditable negative run, but its causal status is
`INVALID_OPTIMIZATION_CONFOUND`. It is not evidence that a bounded adapter is
inferior. AF2FFAB1 will not be retrained or used as a scientific comparator.

The corrective arm AF2FFAB2 uses `0.10*tanh(alpha/0.10)`. It has the same
unit derivative as AF2FFA1 at identity while remaining bounded to ±10%.

