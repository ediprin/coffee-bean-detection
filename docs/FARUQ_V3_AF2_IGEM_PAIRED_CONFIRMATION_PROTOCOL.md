# Faruq-v3 AF2 and IGEM1 Paired Validation Confirmation

Status: **frozen before seed 123/2026 training**. Test remains closed for this
protocol.

## Research question

Do the two strongest non-STB breadth candidates remain useful across random
seeds when each is compared with the already completed, seed-matched D0FT
optimization control?

- `AF2` tests frequency-domain chaotic-amplitude preprocessing.
- `IGEM1` tests classification-guided static/dynamic neighborhood features and
  auxiliary mask supervision.

This protocol confirms each standalone mechanism. It does not combine AF2 and
IGEM1 and does not tune either mechanism after seeing seed-42 validation.

## Frozen evidence and arms

Seed 42 is reused from the completed breadth screen:

| Arm | Macro | Bottom-3 | Worst | Seed-42 status |
|---|---:|---:|---:|---|
| D0FT | 86.69% | 74.98% | 72.02% | optimization control |
| AF2 | 88.20% | 80.04% | 79.35% | RETAIN |
| IGEM1 | 88.01% | 82.18% | 82.08% | RETAIN |

Only AF2 and IGEM1 seed 123/2026 are newly trained. Existing D0 seed 123/2026
checkpoints and existing D0FT reports are reused. Every candidate starts from
the same seed-matched D0 checkpoint used by its D0FT control.

The configurations, 50-epoch schedule, input size, batch size, seed, and
validation split are unchanged from breadth screening.

## Per-candidate acceptance gate

Across seeds 42/123/2026, a candidate passes only if it satisfies all of:

1. mean Macro mAP50-95 gain over D0FT is at least 0.5 point;
2. Macro improves in at least 2/3 paired seeds;
3. mean Bottom-3 mAP50-95 is not lower than D0FT;
4. Bottom-3 improves in at least 2/3 paired seeds;
5. mean Worst-class decline is no greater than 1 point.

AF2 and IGEM1 receive independent PASS/FAIL decisions. A PASS is validation
confirmation, not authorization to reopen or retune on the already consumed
Faruq locked test.

## Engineering and leakage safeguards

- Development archive must not expose a `test` directory.
- Dataset grouping audit must pass before training.
- Seed 123/2026 D0 checkpoints must record the requested seed.
- Completed 50-epoch runs are skipped.
- Partial runs resume from `last.pt`.
- A Drive-visible exclusive lock prevents two Colab runtimes from writing the
  same run simultaneously.
- Non-monotonic resume CSVs are rejected/recovered before continuation.
- Each arm and seed has an independent output directory and validation report.
- Browser output is reduced to a one-minute compact status; full logs are
  written to local Colab disk.

## Frozen output

The authoritative report is:

`experiments/faruq-v3-af2-igem-paired-confirmation-v1/val_reports/af2_igem_paired_confirmation.json`

No test evaluation is part of this runner or notebook.
