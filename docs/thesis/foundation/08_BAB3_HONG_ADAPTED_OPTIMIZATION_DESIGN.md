# 08 — Hong-Adapted Bab III Optimization and Analysis Design

Date: 2026-08-25

Status: **FROZEN DESIGN DECISION BEFORE BAB III REWRITE**

This file records the methodological decisions accepted after re-reading Hong et al. (2026) and comparing that paper's methodology/evaluation structure with the existing AF2 experiment genealogy in this repository.

It is intentionally created **after** the pre-revision checkpoint and **before** rewriting `proposal/05_METHODOLOGY.md`.

The purpose is not to copy Hong's modules or exact experimental protocol. The purpose is to adapt the parts of Hong's scientific design that increase methodological value while preserving the thesis-specific AF2 treatment, grouped data contract, paired controls, and locked-test discipline.

---

## 1. Methodology adapted from Hong

The primary methodological element adapted from Hong is:

> **systematic ablation + sensitivity analysis + visualization/error analysis**

Hong's paper does not merely report a final detector. It separates overall architecture, component-level interventions, hyperparameter sensitivity, ablation, quantitative comparison, and qualitative/error analysis.

For this thesis, that philosophy is translated into an AF2-specific design:

```text
problem-targeted input preprocessing
-> factorized AF2 design study
-> selected AF2 configuration
-> paired confirmatory comparison against native YOLO26
-> tail + mechanism + visualization + error analysis
```

The thesis does **not** copy Hong's DSConv/SPPF-Attention/PConv stack.

---

## 2. AF2 candidates included in the optimization study

The main candidate study is a **factorized one-change-at-a-time AF2 design analysis** derived from the frozen repository protocol `docs/FARUQ_V3_AF2_SPECTRAL_FACTORIZATION_PROTOCOL.md`.

| Candidate | Single design factor tested | Scientific role |
|---|---|---|
| `AF2C` | retained legacy AF2 | reference configuration |
| `AF2WIN` | rectangular analysis/synthesis -> square-root Hann + normalized overlap-add | spectral-window / leakage factor |
| `AF2ORI` | 360 directional bins -> orientation representation modulo pi with 16 bins | angular representation factor |
| `AF2POL` | angular-only -> radial x angular factorization | radial-frequency structure factor |
| `AF2SOFT` | hard entropy threshold -> soft weighting | selection-function factor |
| `AF2LUM` | independent RGB processing -> luminance/shared gate | channel-representation factor |

These arms are not presented as separate modules to be stacked. They are controlled alternatives that isolate individual AF2 design decisions.

The scientific question is therefore:

> **Which AF2 design choices are actually useful for fine-grained coffee-defect detection under a matched YOLO26 detector?**

This is the principal mechanism for making the word **"Optimasi"** in the thesis title operational rather than decorative.

---

## 3. What is NOT part of the primary AF2 optimization family

`PCG1` and `WAV1` are retained only as optional **mechanistic comparators** because the repository protocol defines them as phase-congruency and wavelet alternatives, not AF2 variants.

They may answer:

```text
Is AF2-specific angular Fourier preprocessing preferable to another non-learned spectral/structural frontend?
```

but they do not answer:

```text
Which AF2 configuration is optimal?
```

Other historical experimental families such as AF2RADWAV, DIDA, calibration variants, FFAB2, RCC, and broad module-stacking branches are not promoted into the primary proposal methodology. They remain experiment genealogy unless a later advisor-driven question specifically requires them.

---

## 4. Two-level optimization design

### 4.1 Level A — Structural factor optimization

The first stage compares:

\[
\mathcal{A}=\{AF2_C,AF2_{WIN},AF2_{ORI},AF2_{POL},AF2_{SOFT},AF2_{LUM}\}.
\]

Detector architecture, dataset split, pretrained source, training budget, and evaluation protocol are kept matched.

Selection must consider more than aggregate performance. The retained repository screening logic is the starting point:

- positive Macro mAP50-95 gain;
- Bottom-3 must not deteriorate materially;
- Worst-class must remain within a fixed safety bound;
- latency is reported as an engineering trade-off, not silently ignored.

The final proposal wording should describe this as **structural sensitivity / factorized optimization**, not as arbitrary architecture search.

### 4.2 Level B — Parameter sensitivity

After a structural form is selected, a limited sensitivity analysis may be performed on AF2 parameters with direct methodological meaning.

Candidate parameter set:

\[
\Theta_{AF2}=\{m,o,\gamma,K\}
\]

where:

- \(m\): patch size;
- \(o\): patch overlap;
- \(\gamma\): entropy-conditioned angular-threshold coefficient;
- \(K\): angular-bin resolution.

`radius_ratio` is not an active AF2 parameter in the current `mode=af2` implementation and must not be presented as part of AF2 optimization unless the operator itself changes.

`chunk_size` and `eps` are engineering/numerical parameters rather than scientific optimization variables.

To control thesis scope, the final parameter-sensitivity study does not need to sweep every element of \(\Theta_{AF2}\). The strongest candidates are `gamma` and `patch_size`, because they directly alter angular selection and the spatial scale of local spectral analysis.

All exact candidate values must be frozen before observing the corresponding validation results.

---

## 5. Method freeze and confirmatory experiment

Optimization and confirmation must be separated.

```text
Development train/validation
-> structural AF2 screening
-> limited parameter sensitivity
-> choose AF2*
-> METHOD FREEZE
-> paired confirmatory experiment
```

The final confirmatory comparison is:

\[
I\xrightarrow{YOLO26}\hat{Y}_{native}
\]

versus

\[
I\xrightarrow{AF2^*}I'\xrightarrow{YOLO26}\hat{Y}_{AF2}.
\]

The paired confirmation should retain the current repository principles:

- same official pretrained source;
- matched target-head initialization;
- same grouped data split;
- same schedule;
- paired seeds, currently 42 / 123 / 2026;
- no locked-test use during method selection.

The locked test must not become a hyperparameter-selection oracle.

---

## 6. Quantitative analysis hierarchy

The thesis analysis is deliberately broader than one headline mAP value.

### Level 1 — Overall detection

- Macro mAP50-95 as primary aggregate measure;
- mAP50 as secondary context;
- per-class AP where needed.

### Level 2 — Fine-grained lower tail

- Bottom-3 class mAP50-95;
- Worst-class mAP50-95;
- per-class deltas.

### Level 3 — Mechanism diagnostic

- raw top-500 proposal accessibility;
- localization-conditioned Top-1 classification;
- correct-decision recall.

Interpretation must remain diagnostic rather than causal. In particular, unchanged raw proposal accessibility plus improved conditioned classification may be described as **more consistent with improved class discrimination**, not as proof that localization is unchanged in every sense.

### Level 4 — Efficiency

- detector parameter count;
- latency;
- throughput;
- VRAM / peak memory.

`parameter-free` must never be rewritten as `compute-free` or `lightweight` without measured evidence.

---

## 7. Hong-adapted visualization analysis

Hong's use of EigenCAM and confusion-matrix analysis is adapted, but AF2 allows a more method-specific visualization chain because the intervention occurs in input space.

The proposed qualitative panel is:

```text
Original RGB
-> selected local patch
-> FFT magnitude
-> angular density D(theta)
-> adaptive threshold tau
-> retained angular response
-> reconstructed spatial cue
-> AF2-enhanced RGB
-> YOLO26 activation / CAM if technically valid
-> predicted boxes + confidence
```

The thesis should not claim that a heatmap proves causal feature usage. Visualization is qualitative support only.

If EigenCAM or another detector visualization method is used, compatibility with YOLO26 must be verified before it is named as final methodology.

---

## 8. Error analysis adapted and extended from Hong

Hong's confusion-matrix analysis is retained as a useful precedent, but the thesis should add a paired native-vs-AF2 transition analysis.

For matched validation identities, classify outcomes into:

| Native YOLO26 | AF2-YOLO26 | Category |
|---|---|---|
| wrong | correct | AF2 rescue |
| correct | wrong | AF2 regression |
| wrong | wrong | unresolved |
| correct | correct | stable correct |

For each class \(c\):

\[
R_c=N(\text{native wrong, AF2 correct})
\]

\[
G_c=N(\text{native correct, AF2 wrong})
\]

and a descriptive net-rescue quantity may be reported as:

\[
NR_c=R_c-G_c.
\]

This is an analysis statistic proposed for this thesis; it must not be presented as a standard metric from Hong or the broader literature unless a separate source is found.

The qualitative error set should preferentially include:

1. native wrong -> AF2 correct;
2. native correct -> AF2 wrong;
3. both wrong;
4. difficult lower-tail classes identified quantitatively.

Each selected example can pair RGB/AF2/spectral views with prediction and ground truth.

---

## 9. Proposed Bab III structure after rewrite

The rewrite should move toward the following sequence:

```text
3.1 Kerangka Penelitian

3.2 Dataset Penelitian
    3.2.1 Sumber dan karakteristik dataset
    3.2.2 Taxonomy kelas
    3.2.3 Grouped split dan leakage control
    3.2.4 Augmentasi

3.3 Baseline YOLO26
    3.3.1 Arsitektur baseline
    3.3.2 Pretrained initialization

3.4 Arsitektur Metode yang Diusulkan
    Native RGB -> YOLO26
    RGB -> AF2 -> YOLO26

3.5 Preprocessing Frekuensi-Angular AF2
    3.5.1 Patch extraction
    3.5.2 FFT
    3.5.3 Angular spectral density
    3.5.4 Entropy-adaptive threshold
    3.5.5 IFFT reconstruction
    3.5.6 Residual enhancement

3.6 Analisis dan Optimasi AF2
    3.6.1 Factorization of AF2 design
    3.6.2 Structural candidate screening
    3.6.3 Parameter sensitivity analysis
    3.6.4 Configuration selection and method freeze

3.7 Rancangan Eksperimen Konfirmatori
    3.7.1 Native vs optimized AF2
    3.7.2 Repeated paired seeds
    3.7.3 Locked-test protocol

3.8 Konfigurasi Pelatihan

3.9 Metrik Evaluasi
    3.9.1 Precision / Recall / F1 where useful
    3.9.2 mAP50 and mAP50-95
    3.9.3 Macro mAP50-95
    3.9.4 Bottom-3
    3.9.5 Worst-class / per-class

3.10 Analisis Mekanisme

3.11 Analisis Visualisasi

3.12 Analisis Kesalahan

3.13 Evaluasi Efisiensi

3.14 Lingkungan Implementasi
```

The numbering may be adjusted later to campus formatting, but the scientific order should remain recognizable.

---

## 10. Claim boundaries

The following are frozen:

- Hong is a methodological template, not evidence that AF2 works.
- AF2 candidates are one-factor design alternatives, not a module stack.
- optimization uses development validation only; final locked test is not used for selection.
- negative candidate results remain valid scientific evidence.
- visualization supports interpretation but does not prove mechanism.
- rescue/regression analysis is thesis-specific unless independently sourced.
- classification-vs-localization language must remain proportional to the diagnostics actually measured.
- the final optimized AF2 configuration is not predetermined by repository history; proposal methodology must define prospective selection rules before final confirmation.

---

## 11. Title implication

With this design, the working title

**"Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi"**

has an operational mapping:

\[
\textbf{Optimasi}
=\text{factorized structural analysis + limited AF2 parameter sensitivity}
\]

and

\[
\textbf{Analisis}
=\text{overall + tail + mechanism + visualization + error + efficiency analysis}.
\]

This mapping must be preserved in the upcoming Bab III rewrite.
