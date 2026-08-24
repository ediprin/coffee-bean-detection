# Method Bridge Matrix

Purpose: define the non-coffee literature that makes frequency-aware input preprocessing a technically plausible **candidate solution space** without pretending that transfer to coffee has already been proven.

This file uses the canonical key namespace defined in `CANONICAL_SOURCE_KEYS.md`. Keys in the latest master reference map take precedence over older ad-hoc keys.

## Rule of interpretation

This layer is separate from the coffee evidence matrix.

- Coffee papers establish the problem.
- These papers establish candidate mechanisms, precedents, and diagnostic concepts.
- Only our coffee-domain experiments can establish whether AF2 transfers successfully.

## Preprocessing and enhancement before YOLO / detector

| Key | Paper | Where processing occurs | Learned? | Direct support for proposal | Boundary |
|---|---|---|---|---|---|
| PRE-01 | Liu et al. (2022), *Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions* | Input image before YOLOv3 through differentiable image-processing filters | Yes; CNN-PP predicts filter parameters | Preprocessing can be optimized for downstream detection rather than only visual quality | Adverse-weather domain; not fine-grained coffee |
| PRE-02 | Qin et al. (2022), *DENet: Detection-driven Enhancement Network for Object Detection under Adverse Weather Conditions* | Laplacian-pyramid input decomposition and reconstruction before YOLOv3 | Yes | LF/HF decomposition can be trained for detection utility using detection supervision | Low-light/fog domain; learned network |
| PRE-03 | Li et al. (2025), *FE-YOLO: Fourier Enhancement YOLO for End-to-End Object Detection in Low-Light Conditions* | FFT-based input enhancement before YOLO; amplitude/phase processed then reconstructed | Yes | Closest direct learned-Fourier preprocessing comparator before YOLO | Low-light task; not angular and not parameter-free |
| PRE-04 | Syauqi et al. (2025), *Edge AI-Based Defect Detection in White Pepper Using CLAHE-Based Preprocessing and YOLO* | Fixed image preprocessing before YOLOv8m | No learned preprocessing network | Close seed/spice analogue showing raw-vs-preprocessed YOLO comparison | Treatment is a composite gamma + CLAHE + blending + denoising + unsharp pipeline; two classes only |
| PRE-05 | Chen et al. (2024), *Soft X-ray Image Recognition and Classification of Maize Seed Cracks Based on Image Enhancement and Optimized YOLOv8 Model* | Offline image enhancement before YOLOv8 | Fixed preprocessing | Useful separation of image-enhancement contribution from detector optimization | Soft-X-ray maize cracks are not RGB coffee defects |
| PRE-06 | Tu et al. (2026), WCTE transform-domain preprocessing + YOLO | Wavelet/contrast enhancement before detector | Fixed transform-domain preprocessing | Additional evidence that transform-domain input enhancement can be evaluated as a separate detector treatment | Different defect domain and operator; exact numerical claims require re-opening the primary PDF |
| PRE-07 | Cai et al. (2023), *Retinexformer* | Image enhancement before downstream vision tasks | Learned enhancement | Supports the principle that enhancement should be judged by downstream task utility, not visual quality alone | Low-light enhancement, not fine-grained coffee |
| PRE-08 | Yang & Soatto (2020), *FDA: Fourier Domain Adaptation for Semantic Segmentation* | Input-space Fourier amplitude manipulation | Fixed/domain-adaptation transform | Strong precedent that input images can be manipulated in the Fourier domain while preserving/reusing phase information | Semantic-segmentation/domain-adaptation setting, not object detection |

## Fine-grained mechanism bridge

| Key | Paper | Mechanism | Direct support | Boundary |
|---|---|---|---|---|
| FG-01 | Xu et al. (2025), *More Signals Matter to Detection: Integrating Language Knowledge and Frequency Representations for Boosting Fine-Grained Aircraft Recognition* | LFDet; AFAB / AFAB-2 patch-wise frequency representation | Frequency-aware processing can contribute to fine-grained object detection where subtle category cues matter; parent inspiration for the AF2 line explored in this repository | Aircraft domain; transfer to coffee must be tested |
| FG-02 | Xie et al. (2025), *Learning Discriminative Representation for Fine-Grained Object Detection in Remote Sensing Images* | Explicit discriminative-representation design for fine-grained detection | Supports the proposition that fine-grained detection requires stronger class-discriminative representations than generic detection alone | Remote-sensing domain |
| FG-03 | Wang et al. (2020), *An Adversarial Domain Adaptation Network for Cross-Domain Fine-Grained Recognition* | Fine-grained recognition under subtle inter-class differences | Canonical supporting example for the definition of fine-grained recognition | Classification/domain-adaptation literature, not coffee detection |

## Canonical Fourier / image-processing theory

| Key | Source | What is established | Proposal use | Boundary |
|---|---|---|---|---|
| THEORY-01 | Gonzalez & Woods, *Digital Image Processing*, 4th ed. | 2-D image transforms, DFT/FFT concepts, frequency-domain filtering and standard terminology | Primary textbook anchor for §2.8.1–2.8.2 | Theory source only; no AF2 efficacy claim |
| THEORY-02 | Bracewell, *The Fourier Transform and Its Applications*, 3rd ed. | Fourier-transform definitions and properties | Secondary mathematical foundation | Not image-detection evidence |
| SPEC-01 | Cao et al. (2019), *Frequency Spectrum-Based Optimal Texture Window Size Selection for High Spatial Resolution Remote Sensing Image Analysis* | Fourier energy in polar coordinates can be summarized radially and angularly; radial distributions relate to scale/periodicity and angular distributions to directionality | Primary research-paper basis for the term **frequency-angular** and radial/angular energy equations | Texture-analysis paper, not detector evidence |
| SPEC-02 | Zhang & Tan (2003), *Affine Invariant Classification and Retrieval of Texture Images* | Orientation-spectrum distributions and dominant angular peaks can serve as discriminative texture signatures | Supports directional spectral distributions as texture descriptors | Texture classification/retrieval, not coffee detection |
| WAVE-01 | Finder et al. (2024), *Wavelet Convolutions for Large Receptive Fields* (WTConv) | Wavelet decomposition separates frequency components while retaining spatial localization | Fourier-vs-wavelet comparator | Internal convolutional operator, not input preprocessing |

## Frequency-domain computer vision and defect processing

| Key | Paper | What it establishes | Proposal use | Boundary |
|---|---|---|---|---|
| FREQ-01 | Chi, Jiang & Mu (2020), *Fast Fourier Convolution* | Spatial and spectral processing can be complementary; Fourier branch enables global/non-local interaction | General frequency-domain computer-vision precedent | Internal feature operator, not AF2 |
| FREQ-02 | Li et al. (2024), *FDADNet: Detection of Surface Defects in Wood-Based Panels Based on Frequency Domain Transformation and Adaptive Dynamic Downsampling* | Frequency-domain processing can be integrated into low-contrast surface-defect detection | Defect-domain frequency precedent | Target/background defect separability is not the same as coffee inter-class discrimination |
| FREQ-03 | Chen et al. (2025), *Frequency Dynamic Convolution for Dense Image Prediction* | Adaptive frequency modulation can be learned for dense prediction | Recent frequency-aware vision precedent | Internal learned convolution, not input preprocessing |

## Agricultural frequency-aware feature processing

| Key | Paper | Mechanism | Proposal use | Boundary |
|---|---|---|---|---|
| AGR-01 | Zhao et al. (2026), *A Wavelet-Based Frequency-Domain Approach for Accurate Multi-Crop Disease Detection* | WCR performs DWT inside the YOLO neck/PAN; LL/LH/HL/HH are fused/recalibrated | Agricultural evidence that low/high-frequency complementarity and directional wavelet details can be useful for multi-class detection | Internal feature processing, not external preprocessing |
| AGR-02 | PFENet (2026), physics-informed frequency-enhanced YOLO for hazy object detection | Fourier/high-pass enhancement inside feature maps | Additional evidence that frequency processing is used in YOLO-family detectors | Haze-specific and feature-space |

## Classification–localization diagnosis layer

| Key | Paper | What it establishes | How it constrains our claims |
|---|---|---|---|
| DIAG-01 | Feng et al. (2021), *TOOD: Task-Aligned One-Stage Object Detection* | Classification and localization tasks can be spatially/task misaligned in one-stage detection | Do not equate a detection gain with a localization gain without diagnosis |
| DIAG-02 | Wu et al. (2020), *Rethinking Classification and Localization for Object Detection* | Classification and localization can prefer different feature representations | Supports separating discrimination/classification analysis from box-regression analysis |
| DIAG-03 | Jiang et al. (2018), *IoU-Net* | Classification confidence is not equivalent to localization confidence | Supports careful score interpretation and classification-conditioned diagnostics |

## Where AF2 sits

The reviewed solution space can be organized as:

```text
fixed spatial/intensity preprocessing
    Syauqi / Chen maize
            ↓
transform-domain fixed preprocessing
    WCTE / FDA
            ↓
learned task-driven preprocessing
    IA-YOLO / DENet
            ↓
learned Fourier preprocessing
    FE-YOLO
            ↓
frequency-aware internal feature processing
    FFC / FDADNet / WTConv / FDConv / WGA-YOLO / PFENet
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

> Prior work shows that image preprocessing can be designed for downstream detection utility, while Fourier/frequency-domain methods provide complementary ways to represent spatial structure, scale, texture and directionality. These findings make frequency-angular input preprocessing a technically plausible mechanism to test for fine-grained coffee-defect detection; they do not establish its effectiveness in the coffee domain.

## Prohibited bridge statement

Do not write:

> Coffee defects are difficult because high-frequency or angular information is missing, therefore AF2 solves the problem.

That causal chain is not established by the literature.
