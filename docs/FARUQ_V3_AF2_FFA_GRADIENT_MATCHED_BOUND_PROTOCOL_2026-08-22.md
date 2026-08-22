# Faruq-v3 AF2-FFA gradient-matched bound protocol

Status: frozen before AF2FFAB2 training.

## Question

Can a ±10% spectral residual preserve AF2FFA1's lower-tail improvement while
recovering its small Macro loss when the bound is isolated from the initial
gradient scale?

## Controlled arms

| Arm | Residual amplitude | Status |
|---|---|---|
| AF2FFA0 | `alpha`, zero descriptor | completed historical control |
| AF2FFA1 | `alpha`, spectral descriptor | completed historical candidate |
| AF2FFAB1 | `0.10*tanh(alpha)` | completed, invalid confounded evidence |
| AF2FFAB2 | `0.10*tanh(alpha/0.10)` | only newly authorized arm |

AF2FFAB2 and AF2FFA1 have identical derivative one at `alpha=0`. AF2FFAB2
changes no detector parameter count, state schema, AF2 frontend, dataset,
schedule, seed, or classification-only wiring. It starts from the same AF2
seed-42 checkpoint, trains for 30 epochs, and uses Faruq-v3 grouped validation.

## Static authorization gates

Training is prohibited unless the audit proves:

1. exact identity output at initialization;
2. a ±10% deployed residual bound;
3. initial amplitude derivative equal to AF2FFA1 within `1e-6`;
4. the historical AF2FFAB1 derivative is 0.10, documenting the confound;
5. identical detector parameter count and state schema;
6. classification-only score changes with boxes preserved;
7. finite nonzero adapter gradients; and
8. no test access.

## Seed-42 decision

AF2FFAB2 is retained as a Pareto candidate only if all gates hold:

- Macro is no more than 0.1 point below AF2FFA0;
- Bottom-3 gains at least 0.5 point over AF2FFA0;
- Worst gains at least 1.0 point over AF2FFA0;
- Macro is higher than AF2FFA1; and
- Bottom-3 and Worst fall no more than 0.5 point below AF2FFA1.

Only a retained result may be considered for additional seeds. Test remains
locked. A rejection returns the study to original AF2 without another adapter
variant.

