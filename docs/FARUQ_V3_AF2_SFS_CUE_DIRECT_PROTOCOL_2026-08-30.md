# Faruq-v3 AF2-SFS-CUE direct single-arm protocol — 2026-08-30

Status: **FROZEN BEFORE TRAINING**
Test: **locked**

## Question and economical sequence

Does one final model that combines AF2, the retained P3 space-frequency
selector, and pure-gate multilevel cue supervision produce a large enough
seed-42 signal to justify a later matched control and ablation?

Only `AF2SFSCUE1` is trained in this first screen.  Four-arm factorial training
is deliberately deferred.  The completed historical `AF2DIRECT_seed42` run is
used only as a same-protocol descriptive screen.  It is not represented as a
same-runtime causal control.

If the single arm fails the frozen screen, the study stops.  If it passes, the
next study must reproduce a same-runtime `AF2DIRECT` control before claiming
that the combined architecture caused the improvement; component ablation is
also deferred until that point.

## Final candidate

`AF2SFSCUE1` starts from the official 80-class `yolo26n.pt` artifact with
SHA-256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
It does not load any coffee-trained checkpoint.

1. The frozen AF2 frontend is active from the first training batch.
2. Training-only 1×1 decoders observe the untouched P3/P4/P5 features and
   reconstruct the detached, pure normalized AF2 recovery gate with mean
   Smooth-L1 loss and gain 0.10.
3. After the CUE observation point, an identity-initialized P3 selector mixes a
   learnable local spatial path and a fixed local high-frequency residual.
4. Both native box and class branches consume the same adapted P3.
5. CUE decoders are inactive at inference; SFS remains active.

The pre-SFS observation ordering is frozen to prevent CUE from directly
forcing the selector output to reconstruct the frontend cue.

## Training contract

- Faruq-v3 grouped development train/validation only;
- seed 42;
- maximum 50 epochs, patience 15;
- image size 640, batch 16, workers 2;
- optimizer auto, deterministic true;
- close mosaic 10, max detections 500;
- same official pretrained source and target-head seed as AF2DIRECT;
- no continuation and no test access.

## Economical screening gate

Historical AF2DIRECT seed-42 values are 80.79% Macro, 69.58% Bottom-3, and
66.95% Worst.  The historical summary/checkpoint metadata must be supplied by
the preserved direct-run artifact; the runner does not hard-code reconstructed
evidence.

Authorize a same-runtime control plus ablation only through either route:

- Macro route: Macro +0.50 point, Bottom-3 nonnegative, Worst no worse than
  -0.50 point; or
- lower-tail route: Macro no worse than -0.20 point, Bottom-3 +1.00 point, and
  Worst +1.00 point.

A PASS is only permission for the later causal comparison.  It is not a final
superiority, multiseed, external-domain, or locked-test claim.
