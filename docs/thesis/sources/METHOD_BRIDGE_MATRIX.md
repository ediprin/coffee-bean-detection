# Method Bridge Matrix

Purpose: define the non-coffee literature that makes frequency-aware input preprocessing a technically plausible **candidate solution space** without pretending that transfer to coffee has already been proven.

## Rule of interpretation

This layer is separate from the coffee evidence matrix.

- Coffee papers establish the problem.
- These papers establish candidate mechanisms, precedents, and diagnostic concepts.
- Only our coffee-domain experiments can establish whether AF2 transfers successfully.

## Preprocessing and enhancement before YOLO

| Key | Paper | Where processing occurs | Learned? | Direct support for proposal | Boundary |
|---|---|---|---|---|---|
| PRE-01 | Liu et al. (2022), *Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions* | Input image before YOLOv3 through differentiable image-processing filters | Yes; CNN-PP predicts filter parameters | Preprocessing can be optimized for downstream detection rather than only visual quality; image processing and detector can be trained jointly with detection loss | Adverse-weather domain; not fine-grained coffee; cannot be used to claim AF2 works |
| PRE-02 | Qin et al. (2023), *DENet: Detection-driven Enhancement Network for Object Detection under Adverse Weather Conditions* | Input decomposition and enhancement before YOLOv3 | Yes; lightweight enhancement network | Laplacian-pyramid LF/HF decomposition can preserve/strengthen latent detection-relevant information before a detector | Low-light/fog domain; learned network; not parameter-free Fourier-angular processing |
| PRE-03 | Li et al. (2025), *FE-YOLO: Fourier Enhancement YOLO for End-to-End Object Detection in Low-Light Conditions* | FFT-based enhancement before YOLO; amplitude/phase processed then reconstructed | Yes; FENet/FPB + joint enhancement/detection loss | Closest direct methodological comparator: Fourier-domain enhancement can be integrated before YOLO and optimized for detection | Low-light task; learned Fourier enhancement; not angular energy selection and not coffee |
| PRE-04 | Syauqi et al. (2025), *Edge AI-Based Defect Detection in White Pepper Using CLAHE-Based Preprocessing and YOLO* | Fixed image preprocessing before YOLOv8m | No learned preprocessing network | Close agricultural seed/spice analogue showing a preprocessing-vs-raw YOLO comparison | The treatment is a **CLAHE-based composite pipeline** (gamma correction, CLAHE, blending, denoising, unsharp masking), not CLAHE alone; two classes only |
| PRE-05 | Chen et al. (2024), *Soft X-ray Image Recognition and Classification of Maize Seed Cracks Based on Image Enhancement and Optimized YOLOv8 Model* | Offline image enhancement before YOLOv8 | Fixed preprocessing; detector separately optimized | Useful separation of preprocessing effect from architecture effect; reported image enhancement adds about +1.8 pp AP in that setup | X-ray maize cracks are not RGB coffee defects; preprocessing is a multi-step spatial/wavelet pipeline |

## Fine-grained frequency mechanism bridge

| Key | Paper | Mechanism | Direct support | Boundary |
|---|---|---|---|---|
| FG-01 | Xu et al. (2025), *More Signals Matter to Detection: Integrating Language Knowledge and Frequency Representations for Boosting Fine-Grained Aircraft Recognition* | LFDet; AFAB / AFAB-2 patch-wise frequency representation | Frequency-aware processing can contribute to fine-grained object detection where subtle category cues matter; parent mechanism for the AF2 line explored in this repository | Aircraft domain. It does not prove transfer to coffee; AFAB-2 should be treated as parent inspiration, not coffee evidence |
| FG-02 | Xie et al. (2025), *Learning Discriminative Representation for Fine-Grained Object Detection in Remote Sensing Images* | Explicit discriminative-representation design for fine-grained detection | Supports the broader proposition that fine-grained detection needs discriminative representations beyond generic detection features | Remote-sensing domain; mechanism transfer requires validation |

## Frequency and angular theory

| Key | Paper | What is established | Proposal use | Boundary |
|---|---|---|---|---|
| FREQ-01 | Cao et al. (2019), *Frequency Spectrum-Based Optimal Texture Window Size Selection for High Spatial Resolution Remote Sensing Image Analysis* | Fourier energy in polar coordinates can be summarized radially and angularly; radial distributions relate to frequency/periodicity and angular distributions to directional texture structure | Primary theoretical basis for the term **frequency-angular** | Texture-analysis paper, not detector evidence |
| FREQ-02 | Zhang & Tan (2003), *Affine Invariant Classification and Retrieval of Texture Images* | Orientation-spectrum distributions and dominant angular peaks can serve as discriminative texture signatures | Supports the proposition that directional spectral distributions can carry discriminative information | Texture classification/retrieval, not coffee detection |
| FREQ-03 | Finder et al. (2024), *Wavelet Convolutions for Large Receptive Fields* (WTConv) | Wavelet decomposition retains spatial localization while separating frequency components | Useful Fourier-vs-wavelet theoretical comparator | Internal convolutional operator, not input preprocessing |
| FREQ-04 | Chen et al. (2025), *Frequency Dynamic Convolution for Dense Image Prediction* | Frequency-domain parameterization can distinguish lower-frequency/global content and higher-frequency/detail-sensitive responses | Supporting frequency-aware dense-prediction context | Internal learned convolution, not AF2 preprocessing |

## Agricultural frequency-aware feature processing

| Key | Paper | Mechanism | Proposal use | Boundary |
|---|---|---|---|---|
| AGR-01 | Zhao et al. (2026), *A Wavelet-Based Frequency-Domain Approach for Accurate Multi-Crop Disease Detection* | WCR performs DWT inside the YOLO neck/PAN; LL plus LH/HL/HH are fused/recalibrated | Agricultural evidence that low/high-frequency complementarity and directional wavelet details can be useful for multi-class detection | Internal feature processing, not external preprocessing; crop lesions are not coffee beans |
| AGR-02 | PFENet (2026), *Physics-Informed Frequency-Enhanced YOLO for Object Detection in Hazy Scenes* | Fourier/high-pass enhancement inside feature maps | Additional evidence that frequency enhancement is being integrated into YOLO-family detection | Haze-specific and feature-space; not evidence of coffee suitability |

## Classification–localization diagnosis layer

| Key | Paper | What it establishes | How it constrains our claims |
|---|---|---|---|
| DIAG-01 | Feng et al. (2021), *TOOD: Task-Aligned One-Stage Object Detection* | Classification and localization tasks can be spatially/task misaligned in one-stage detection | Do not equate a detection gain with a localization gain without diagnosis |
| DIAG-02 | Wu et al. (2020), *Rethinking Classification and Localization for Object Detection* | Classification and localization can prefer different feature representations | Supports separating discrimination/classification analysis from box-regression analysis |
| DIAG-03 | Jiang et al. (2018), *IoU-Net* | Classification confidence is not equivalent to localization confidence | Supports proposal-accessibility / classification-conditioned diagnostics and careful score interpretation |

## Where AF2 sits

The reviewed solution space can be organized as:

```text
fixed spatial/intensity preprocessing
    Syauqi / Chen
            ↓
learned task-driven preprocessing
    IA-YOLO / DENet
            ↓
learned Fourier preprocessing
    FE-YOLO
            ↓
frequency-aware internal feature processing
    WGA-YOLO / PFENet / WTConv / FDConv
```

The thesis evaluates a different point in this space:

```text
input RGB
   ↓
patch-wise Fourier analysis
   ↓
angular spectral weighting / selection
   ↓
IFFT + residual image enhancement
   ↓
YOLO26
```

with **no learned preprocessing parameters**.

## Allowed bridge statement

A proposal-safe bridge is:

> Prior work shows that image preprocessing can be optimized for detection utility, while frequency-domain methods can expose or preserve complementary global, edge, texture, and directional information. These findings make frequency-angular input preprocessing a technically plausible mechanism to test for fine-grained coffee-defect detection; they do not establish its effectiveness in the coffee domain.

## Prohibited bridge statement

Do not write:

> Coffee defects are difficult because high-frequency or angular information is missing, therefore AF2 solves the problem.

That causal chain is not established by the literature.