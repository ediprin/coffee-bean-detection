# Faruq-v3 CAFR protocol — 2026-08-20

## Status

Development-only protocol for **Coffee-Adaptive Frequency Representation (CAFR)** on YOLO26n.
The locked test split remains forbidden until the development protocol selects and freezes a final candidate.

## Scientific lineage

CAFR is **not** presented as a new Fourier transform. It is a coffee-domain redesign of the AFAB-2
signal-selection idea from Xu et al. (Neural Networks 187, 2025, 107402).

Parent AF2 limitation being tested:

1. AFAB-2 aggregates amplitude over radius and selects by direction, so radial frequency-band identity is lost.
2. The transferred implementation processed RGB channels independently, which can alter chromatic ratios.
3. Hard suppression can erase weak-but-discriminative signals.
4. The aircraft-domain patch size m=32 is not assumed to be optimal for coffee objects.

These are **design/transfer limitations**, not yet proven causes of error. The experiment is designed to test them.

## Proposed CAFR operator

For RGB input `X`, use Rec.709 luminance only as the spectral guide:

`Y = 0.2126 R + 0.7152 G + 0.0722 B`.

For each local patch `P_k`:

1. `F_k = DFT(P_k) = A_k exp(j phi_k)`.
2. Normalize Fourier radius `rho` to `[0,1]`.
3. Divide radius into three frozen bands: `[0,1/3)`, `[1/3,2/3)`, `[2/3,1]`.
4. For every radial band `b` and orientation bin `theta`, compute
   `D_{k,b}(theta) = sum_{r in B_b} A_k(r,theta)`.
5. Compute normalized angular probability and entropy per radial band.
6. Reuse the Xu AFAB-2 entropy threshold form `t = gamma/(1+exp(-H))`.
7. Replace hard suppression in the final method with
   `W = sigmoid((D_hat - t)/tau)`.
8. Keep the original Fourier phase and reconstruct the luminance cue with iDFT.
9. Reconstruct overlapping patches by fold/overlap averaging.
10. Normalize the recovered one-channel cue to `G` and apply the same gain to R/G/B:
    `X' = X + X * G`.

Because the same `G` multiplies all RGB channels, chromatic ratios are preserved by construction.
The native YOLO26n backbone, neck, box branch, classification branch, and native loss remain unchanged.
CAFR has no learned parameters.

## Frozen causal ladder

- `AF2-Xu`: existing transferred parent/reference; not retrained by this branch unless required.
- `C1`: luminance spectral guide + shared RGB gate; original directional hard selection.
- `C2`: C1 + explicit radial x directional decomposition.
- `C3`: C2 + entropy-conditioned soft selection.
- `C4`: C3 + unsigned 180-degree orientation representation with 16 bins, exploiting conjugate symmetry of real-image Fourier magnitude.
- `CAFR`: C4 + coffee-scale-calibrated patch size.

No additional attention, transformer, auxiliary classifier, altered box head, or new detection loss is permitted in this study.

## Patch calibration

The final CAFR patch is selected before training from **training labels only**.
For normalized YOLO bbox `(w,h)` at training size `S=640`, define equivalent side

`s = S * sqrt(w*h)`.

Report Q25 / median / Q75 over all training boxes. Frozen candidates are `{16,32,64}`.
Choose the largest candidate not exceeding the median equivalent bbox side; if none qualifies, choose the smallest.

This rule is a pre-registered domain-calibration heuristic, not a claim that the selected size is globally optimal.
The generated calibration JSON is part of the run contract.

## Development execution

First stage: seed 42 only, development validation only.

Run sequence:

`C1 -> C2 -> C3 -> C4 -> CAFR`.

All arms must start from the same D0 seed-42 checkpoint and the same dataset split/training budget.
The runner rejects a development root that exposes a `test/` directory.

CLI:

```bash
python -m coffee_detector.experiments.run_faruq_v3_cafr_arm \
  --arm C1 \
  --data-root <DEV_ROOT> \
  --grouped-summary <GROUPED_SUMMARY_JSON> \
  --d0-checkpoint <D0_SEED42_BEST_PT> \
  --output-root <OUTPUT_ROOT> \
  --device 0 \
  --authorize-training
```

For `CAFR`, `--labels-root` is optional; default is `<DEV_ROOT>/train/labels`.

## Evaluation

Primary development evidence:

- Macro mAP50-95
- Bottom-3 class mAP50-95
- Worst-class mAP50-95
- per-class AP and confusion diagnostics
- matched localization accessibility / recall
- conditional class accuracy among localized detections (`A_cls|loc`)
- parameter count
- batch-1 latency
- FLOPs / memory when the matched benchmark is run

Interpretation rule:

- If localization is materially unchanged while `A_cls|loc` and lower-tail AP improve, evidence supports improved fine-grained discrimination.
- If localization improves substantially too, claim a shared input-representation benefit rather than classification-specific improvement.
- If only aggregate mAP rises while lower-tail and matched classification do not, do not force a fine-grained mechanism claim.

## Promotion

After the seed-42 causal ladder, select at most the best two non-baseline variants for seeds 123 and 2026.
The exact promotion gate must be frozen in the decision document before the multi-seed runs.

No locked-test evaluation is authorized by this protocol.
