# Auliya Discriminative Representation Audit v1 (Kaggle)

Status: registered implementation, no experiment results or detector training.
Base: 4efb86fbbfbb4a14485f5477ddb7faa1b53983ba.
This is the existing B->C pilot, NOT the official fold-A reproduction (676/119).
Do not combine their checkpoints or scores.

## Research questions

1. Does a common fixed visual descriptor retain more class information when
   extracted from original-resolution instance crops than from crops of the
   640-pixel letterboxed image?
2. Does linear decodability of defect identity differ across YOLO26 feature
   levels? Does combining levels help at fixed descriptor dimension?

Prior CoffeeFG ROI refinement and bilinear pooling failed; do not train another
refiner here. Prior Faruq and Coffee17 results are motivation, not independent
confirmation. Do not use group A, Faruq checkpoints, or Coffee17 data for fitting.

## Exact input contract (Kaggle Input)

Attach the existing **unmodified** pilot ZIP
auliya-robusta-sni-frequency-colab-v1.zip (from Coffee_Bean_Detection/bundles)
and the RGB pilot weights/last.pt (from experiments/auliya-robusta-sni-frequency-v1/RGB).
Upload as a private Kaggle Dataset or attach an existing private Notebook Output.
No need to upload the raw Auliya archive, test images, or official fold-A weights.
The expected SHA-256 values are those in the previous run's training log and
contract:

- pilot ZIP: b8795e189b85a13201e54a562cbe0f42fa7e0f2bfc0eb4b6d04f85d2b4cf57be
- RGB last.pt: 85b9b459fcd5079de3c27f8b897a4bd0c7160efa96bea25957c5eadd4d617ed3
- embedded development/development_contract.json:
  23d653ca2e98c185a76bb7b2095d6afc6401876cbe48fd219cb42bdc7ecbd360

The runner streams only whitelisted development/train and development/val
images and labels from the verified ZIP; it does not execute bundled code or
extract the ZIP. Training B: 375 images; validation C: 420. Group A: 1351 held
out; never read its pixels. The underlying original archive and metadata
hashes were established during the pilot; this audit trusts the exact pilot
bundle instead of re-auditing the full source archive. The preflight rejects
class order, image geometry, EXIF orientation (nonidentity requires review),
checkpoint version/arm/schedule, and polygon instance count mismatches.

## Visual stage

All colour/texture features use a masked 128x128 crop with the same descriptor
operator. Nonuniform aspect normalization is deliberate and shared; geometry
descriptors are computed before this warp. V0 and V1 compare this specific
probe's sensitivity to scene resizing, not the total visual information
available to an ideal classifier. V1 uses the 640 square letterbox pixels,
not a resized high-resolution crop. The crop mask is rasterized from GT
polygons (oracle diagnostic, not deployable inference).

| Code | Descriptor |
|---|---|
| V0 | Original crop: Lab histogram + Gabor/LBP/local differences + 8x8 masked RGB means |
| V1 | Same descriptor from 640 letterboxed crop |
| V2 | Original crop Lab histograms |
| V3 | Original crop Gabor response mean/std, quantized LBP, adjacent-pixel absolute differences (Haar-like), missing-interior flag |
| V4 | Polygon scale, shape and Hu moments in original coordinates |
| V5 | V2+V3+V4 |

Texture uses eroded interior support; no valid interior gives a zero vector
with explicit missing-support flag. Gabor orientations are not rotation
invariant; geometric scale is image-relative, not calibrated millimetres.

## Feature stage

Load only the verified RGB pilot YOLO26n-seg last.pt, Ultralytics 8.4.96,
eval mode, no gradients, no optimizer, fixed square 640 input. The model
is custom-serialized; import the pinned repo module before loading. This
diagnostic input geometry is not a reproduction of the original validation mAP.

| Code | Descriptor |
|---|---|
| F0 | One-to-one classification penultimate vector at raw class-agnostic IoU-matched anchor |
| F1 | P2 backbone layer 2, GT polygon weighted mean |
| F2 | P3 head input, GT polygon weighted mean |
| F3 | P4 head input, GT polygon weighted mean |
| F4 | P5 head input, GT polygon weighted mean |
| F5 | F2+F3+F4 |
| F6 | F1+F2+F3+F4 |

F0 uses a greedy unique anchor assignment over *all* decoded raw boxes,
independent of class and confidence. Record assigned anchor index, IoU and
raw class logits for every instance. The main probe compares all GT objects;
report IoU>=0.5 sensitivity on the same subset and support. F0 may differ
from other probes because of spatial support and level choice, not just
semantics; F1–F6 use the same oracle GT mask. No predicted mask, no mAP
claim, no learned detection head. Masking a late feature does not erase
receptive-field contamination.

## Frozen probe and decision

Fit StandardScaler, PCA and balanced multinomial logistic regression
(C=1, max_iter=2000, seed 42) on B only; predict C. Fixed primary matched PCA
ranks: 8 (all visual descriptors) and 32 (all deep descriptors). This is a
deliberately conservative common dimension; compression can discard useful
information. Raw-dimension results are secondary to diagnose that cost,
not to select whichever result looks strongest. No hyperparameter sweep.

Train and predict all 21 classes. Primary metric Macro-F1 over IDs 0–17
computed from the full 21-class confusion (do not remove errors to/from
IDs 18–20); auxiliary macro21, fixed Hard-F1 over the five named difficult
classes, worst18, all class F1/support/confusion, per-instance paired
rescue/damage and localized-only metrics. All scores are fractions [0,1].
Raw native anchor argmax is a diagnostic, not a native final detection score.
Any missing class in a localized-only subset is reflected in support and
zero-division F1, not silently dropped. The validation group is development
only. Repeated photographs may be correlated; no iid instance-level CI or
significance claim.

Predeclared interpretation for further *diagnosis*, not model promotion:
- V0 vs V1: if matched Macro18 and Hard5 improve by at least 0.02 each,
  consider resolution loss; check raw-dimension and per-class direction.
- F5 vs strongest single neck level F2/F3/F4: if matched Macro18 and Hard5
  improve by >=0.02, and Worst18 is not lower by >0.01, treat as multilevel
  headroom; the maximum comparator makes this exploratory and conservative.
- F6 vs F5: same thresholds before considering shallow injection.
- Otherwise keep the hypothesis unsupported on this probe. F0 vs F5 is
  descriptive because of unequal spatial support. Do not infer information
  absence from linear probe failure.
- No automatic selection, no retraining, no test access, no ISDE yet.
  Stage R1–R4 (global/regional mean/histogram) needs a separate frozen design
  after this result. Context/TAL audits also remain separate.

## Kaggle run

Import notebooks/Auliya_Discriminative_Representation_Audit_Kaggle.ipynb,
attach the two exact input files, enable Internet (public GitHub + package
install) and GPU (feature stage, single GPU enough), then Run All.
Visual-only CPU run is possible by disabling RUN_FEATURES in the notebook.
The notebook checks out the exact scientific-code commit, runs the focused
synthetic tests, and logs runtime/package versions before starting.

Save Version, then attach previous Notebook Output on a subsequent session
for resume. Only one prior output root is accepted. Shards are published
atomically per image, reused only under an identical run contract, and
collected again for probe fitting. Do not manually edit outputs. Keep the
same hardware/software fingerprint for resume; if changed, start a fresh
output rather than mix runs. The notebook does not mount Drive or request
credentials. GitHub cannot execute the Kaggle session from chat.

Output under /kaggle/working/auliya-discriminative-representation-v1:
stage-specific contract.json, shards, train_features.npz, val_features.npz,
validation_predictions.csv, report.json, logs and environment inventory.
Send report.json files (visual and features) and validation_predictions.csv
back for the next decision. No actual GPU/real-data run is claimed by the
implementation commit.
