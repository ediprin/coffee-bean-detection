# Faruq-v3 AF2 Multilevel Training Scaffold Protocol — 2026-08-28

Branch: `codex/af2-multilevel-training-scaffold`
Status: **FROZEN — SEED-42 KILL-GATE AUTHORIZED; TEST LOCKED**

## Research question

Can the optimization-mediated gain discovered in AF2SFS1 be amplified by a
temporary P3/P4/P5 spatial-frequency scaffold, while the validation and
deployed detector remain the native AF2 architecture?

This is one deliberately decisive seed-42 screen. It is not a multi-seed
confirmation and cannot by itself support a stability claim.

## Causal basis

The completed full-mAP intervention showed that AF2SFS1's +0.95-point Macro
gain survived adapter bypass, whereas the active selector contributed
approximately zero. The supported mechanism is therefore a changed training
trajectory. The new candidate tests that mechanism directly rather than adding
another inference module.

## Candidate

`AF2MTS1` wraps the native AF2 Detect head with identity-initialized
space/frequency residual scaffolds on all three pyramid inputs:

- P3: fine spatial detail;
- P4: intermediate context;
- P5: coarse semantic context.

The scaffold is active only in `model.train()`. Every validation and inference
forward bypasses it exactly. Training strength is 1.0 through epoch 18,
cosine-decays during epochs 19–27, and is exactly zero for epochs 28–30. The
completed checkpoint is exported with the wrapper and all scaffold parameters
removed. The exported model must reproduce bypass output and validation metrics.

## Fixed comparison

- Parent and initialization: completed AF2 seed-42 checkpoint.
- Control: completed matched `AF2CTRL` 30-epoch continuation.
- Candidate: `AF2MTS1`, seed 42, 30 epochs.
- Data: leakage-safe Faruq-v3 grouped development train/validation.
- YOLO model, AF2 configuration, optimizer, augmentation, image size, batch,
  patience, and close-mosaic schedule are identical to AF2CTRL.
- Test is unavailable.

## Static gates

Training is forbidden unless all gates pass:

1. initial train-mode output is exactly the AF2 parent;
2. eval output is exactly AF2 before and after artificial scaffold activation;
3. all P3/P4/P5 adapters receive finite gradients and can alter train output;
4. schedule endpoints are exactly 1.0 and 0.0 as frozen above;
5. native head schema is preserved and the stripped export contains no scaffold
   parameters;
6. no ROI, decoded-box dependency, test path, or test authorization exists.

## Seed-42 kill gate

`AF2MTS1` advances only if all conditions hold:

- Macro mAP50–95 is at least **90.50%** and at least **+1.50 points** over the
  matched AF2CTRL value;
- Bottom-3 AP is at least **84.50%** and is not below AF2CTRL;
- Worst-class AP is not below AF2CTRL (83.54748589196677% raw reference);
- all 21 validation classes are present;
- stripped and wrapped-bypass validation metrics are equal within `1e-6`;
- test was not accessed.

Failure stops this multilevel-scaffold direction. Passing the screen only
creates a high-priority candidate; it does not silently authorize test access.

## Artifacts

- output root: `experiments/faruq-v3-af2-multilevel-scaffold-v1`;
- static audit: `static_audit.json`;
- candidate result: `val_reports/AF2MTS1_seed42_result.json`;
- decision: `val_reports/af2mts1_seed42_decision.json`;
- exported native checkpoint: `AF2MTS1/AF2MTS1_seed42/weights/best_native.pt`.
