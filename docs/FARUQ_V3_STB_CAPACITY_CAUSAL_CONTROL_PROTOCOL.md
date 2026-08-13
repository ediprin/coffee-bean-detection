# Faruq-v3 STB Capacity-Causal Control Protocol

Status: seed-42 gate **PASS**. Paired seeds 123/2026 are authorized; test
remains locked.

## Question

Does STB1 improve fine-grained coffee-defect detection because shifted-window
spatial token interaction is useful, or merely because STB adds parameters and
optimization capacity to the native YOLO26 classification path?

## Frozen arms

- `D0FT`: existing native-head fine-tuning reference.
- `CMC0`: new non-spatial capacity control. At each P3/P4/P5 level it has two
  residual channel-mixing blocks, exactly like STB1's two-block depth. Four
  C-to-C linear projections replace each attention projection group, followed
  by the same-size MLP and LayerNorm structure. All operations remain
  independent at each spatial location.
- `STB1`: completed W-MSA then SW-MSA classification-only candidate.

Both CMC0 and STB1 start bitwise-identically from the same D0 checkpoint, use
the same scalar identity gate, modify classification only, and use the same
50-epoch training schedule. STB1 has 4,589,201 parameters and CMC0 4,588,025:
a difference of 1,176 parameters or 0.0256%.

This is a capacity-near-matched causal control, not a claim of exact functional
equivalence. It isolates spatial token mixing while preserving depth, channel
capacity, schedule, initialization, and detector wiring.

## Static gate

Training is forbidden unless:

1. both models reproduce D0 boxes and scores bitwise at gate zero;
2. opening either gate changes scores but preserves raw boxes;
3. both process P3/P4/P5 with two blocks per level;
4. parameter-count difference is below 0.05%;
5. CMC0 contains no spatial attention or spatial convolution.

## Seed-42 gate

CMC0 must first be a viable control relative to D0FT: no more than 1 Macro,
2 Bottom-3, or 3 Worst-class points lower. This prevents a deliberately weak
control from manufacturing an STB advantage.

Given a viable control, the STB causal comparison passes only when STB1:

- gains at least 0.5 Macro point over CMC0;
- does not lower Bottom-3;
- loses no more than 1 Worst-class point.

Both gates must pass. Passing authorizes paired STB/CMC0 confirmation on seeds
123 and 2026. Failure stops the causal claim without test access. Existing
STB1 seed-42 artifacts are reused; only CMC0 is trained in this stage.

## Boundaries

- Faruq-v3 grouped development data and validation only.
- Seed 42 is screening, not final confirmation.
- No test extraction or evaluation.
- No hyperparameter change after observing CMC0.

## Resume incident amendment

The first seed-42 execution was interrupted after a clean best checkpoint at
epoch 11. The original resume adapter failed to load the custom CMC0 weights,
and overlapping Colab resumes produced non-monotonic CSV rows. All rows after
the clean prefix are invalid. The runner now (1) strictly loads the complete
CMC0 checkpoint on resume, (2) rejects non-monotonic epoch sequences, and (3)
uses a Drive-visible heartbeat lock to prevent concurrent writers.

The explicit `--recover-from-best` action archives the corrupt CSV/last
checkpoint, restores `last.pt` from the resumable clean `best.pt`, truncates
the CSV to its matching 11-epoch prefix, and resumes unchanged training. It is
idempotent when the run is already clean. This is engineering recovery, not
model selection or hyperparameter tuning; the architecture, seed, schedule,
validation gate, and test lock remain unchanged.
