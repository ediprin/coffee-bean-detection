# Faruq-v3 DC2 Raw-Crop Resolution Search Protocol

Date frozen: 2026-08-07
Stage: broad candidate search / mechanistic screen
Candidate family: `DC2 raw RGB local stream`
Branch: `agent/dc2-raw-crop-screening`

## Research question

Does re-encoding a detected coffee bean from the original RGB image at a fixed
local resolution recover fine-grained class information that may have been lost
by full-frame detector downsampling?

This is deliberately separated from the earlier CoffeeFG ROI refiners. CoffeeFG
used ROIAlign on already-encoded detector feature maps. The present screen crops
the original RGB image first and only then resizes/encodes the object, which is
the central local-stream mechanism of DC2.

## Paper-derived setup

Zheng et al., *IEEE TIP*, 2025, define local-stream features as
`F_o,l = f_l(X_o)` where `X_o` is the object image cropped from the original
input by its bounding box. They further report an ablation over crop size and
fix 128x128 for subsequent experiments; 224x224 eventually degrades because
small instances become distorted by excessive interpolation.

Therefore the resolution set is frozen before seeing coffee results:

- 32x32
- 64x64
- 128x128
- 224x224

## Coffee adaptation boundary

This first screen is **not the complete DC2 detector**. It isolates the raw-local
stream mechanism before paying the cost of predicted-crop integration or MSFA.

- train and validation crops use GT boxes only;
- crop is performed on raw RGB pixels before resize;
- no detector feature/ROIAlign is consumed;
- all resolutions use the same MobileNetV3-Small ImageNet-pretrained local
  backbone and the same optimizer/schedule;
- context factor is fixed at 1.0 (exact box);
- only flip augmentation is used; no color augmentation is introduced;
- evaluation reports crop-classification Macro-F1, Bottom-3 F1, Worst-F1, and
  accuracy;
- these metrics are not detector mAP and must not be presented as such.

If raw-resolution signal is retained, the next step is a predicted-box raw-crop
local-stream integration, followed later by multi-stream global/local fusion.

## Fixed training setup

- data: leakage-safe Faruq-v3 grouped train/validation only;
- classes: 21;
- seed: 42;
- local backbone: torchvision MobileNetV3-Small, ImageNet weights;
- epochs per resolution: 20;
- batch size: 64;
- AdamW learning rate: 3e-4;
- weight decay: 1e-4;
- cosine learning-rate schedule;
- checkpoint selection: best validation Macro-F1 within each predeclared arm;
- test split must remain absent.

## Frozen resolution-signal gate

Let `r*` be the resolution with the best validation Macro-F1. A meaningful
raw-resolution signal requires all of:

1. `r* != 32`;
2. Macro-F1(r*) - Macro-F1(32) >= +2.00 percentage points;
3. Bottom-3-F1(r*) is not below Bottom-3-F1(32).

If all pass: `RETAIN_DC2_LOCAL_STREAM` and authorize a separate predicted-box
raw-crop integration protocol.

If not: `WEAK_RAW_RESOLUTION_SIGNAL`. This does not prove DC2 impossible; it
only means this controlled raw-resolution mechanism did not provide a clear
signal under the fixed lightweight local classifier.

## Boundaries

- no locked test access;
- no adaptive resolution additions after seeing results;
- do not compare crop Macro-F1 numerically to detector mAP as if they were the
  same metric;
- do not call this an end-to-end DC2 reproduction;
- the end-to-end/MSFA claims from the paper require later experiments if this
  local-stream screen is positive.
