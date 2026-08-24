# Alur Penelitian / Research-Flow Specification

Status: **synchronized with Bab I–III as of 2026-08-25**.

Purpose: provide the single source for the research-flow figure that will later be redrawn in the formal proposal document.

## Figure 3.1 — Arsitektur Umum Penelitian

```mermaid
flowchart TD
    A[Kajian literatur dan identifikasi masalah\nfine-grained coffee-defect discrimination] --> B[Dataset Faruq-v3 grouped\n21 classes]
    B --> C[Audit split\nparent/hash leakage gates\ntrain 1665 / val 294]
    C --> D[Official yolo26n.pt\nfrozen source hash]
    D --> E[Matched 21-class detector initialization\nsame persistent state]

    E --> N[Arm 1: D0DIRECT\nNative RGB]
    E --> P0[Arm 2: AF2DIRECT]

    P0 --> P1[Overlapping patches\n32x32, overlap 0.50]
    P1 --> P2[FFT per RGB channel]
    P2 --> P3[Angular amplitude density\n360 bins]
    P3 --> P4[Entropy-conditioned threshold\ngamma 0.10]
    P4 --> P5[Directional weighting]
    P5 --> P6[IFFT + overlap averaging]
    P6 --> P7[Residual image gate\nI' = I + I * minmax(R)]

    P7 --> T2[YOLO26n P3-P5]
    N --> T1[YOLO26n P3-P5]

    T1 --> TR1[Matched training schedule\nmax 50 epochs]
    T2 --> TR2[Matched training schedule\nmax 50 epochs]

    TR1 --> V[Validation evaluation]
    TR2 --> V

    V --> M1[Aggregate\nmAP50 / mAP50-95 / Macro]
    V --> M2[Lower-tail\nBottom-3 / Worst / per-class AP]
    V --> M3[Mechanism diagnostics\nproposal accessibility\nlocalization-conditioned Top-1\ncorrect-decision recall]
    V --> M4[Efficiency\nparams / latency / throughput / VRAM]

    M1 --> S[Paired per-seed deltas\nAF2DIRECT - D0DIRECT]
    M2 --> S
    M3 --> S
    M4 --> S

    S --> R[Sintesis hasil\noverall + tail + diagnostic + efficiency]

    X[Locked test\nnot used for method selection] -. frozen until final protocol .-> R
```

## Narrative version for proposal prose

Alur penelitian dimulai dari identifikasi masalah fine-grained pada deteksi cacat biji kopi berdasarkan kajian literatur. Dataset Faruq-v3 kemudian digunakan dalam grouped split yang telah diaudit terhadap parent overlap dan exact-hash leakage. Kedua arm eksperimen dibangun dari official `yolo26n.pt` yang sama dan menggunakan inisialisasi 21-class detector yang dipadankan.

Pada arm baseline, citra RGB diteruskan langsung ke YOLO26n. Pada arm treatment, citra terlebih dahulu diproses AF2 melalui overlapping patch extraction, FFT per channel, pembentukan angular amplitude density, entropy-conditioned directional weighting, inverse FFT, overlap averaging, dan residual image reconstruction. Citra hasil AF2 kemudian diberikan ke YOLO26n yang secara arsitektural sama dengan baseline.

Kedua arm dilatih dengan schedule yang sama. Evaluasi validation dibagi menjadi empat kelompok: aggregate detection, lower-tail/per-class performance, classification–proposal-accessibility diagnostics, dan efficiency. Hasil dibandingkan sebagai paired delta `AF2DIRECT - D0DIRECT` pada setiap seed sebelum disintesis menjadi kesimpulan penelitian.

Locked test tidak digunakan untuk memilih metode atau mengubah konfigurasi selama screening dan confirmation development.

## Figure guardrails

The final rendered figure must preserve these distinctions:

1. AF2 is **before** YOLO26, not inside backbone/neck/head.
2. Both arms share the same detector initialization source.
3. `radius_ratio` must not appear in the AF2 flow because it is inactive in `mode=af2`.
4. Mechanism diagnostics are separate from primary mAP metrics.
5. Efficiency is a separate evaluation branch because parameter-free is not compute-free.
6. Locked test should be visually separated from model-selection flow.
7. Pilot seed 42 should not be embedded as if it were part of the final result; pilot status belongs in proposal text/caption if needed.

## Figure 3.2 — Detail Operator AF2

```mermaid
flowchart LR
    I[Input RGB I] --> U[Unfold overlapping patches]
    U --> F[FFT2 + fftshift]
    F --> A[Amplitude A and original phase phi]
    A --> D[Angular density D theta]
    D --> H[Probability p and entropy H]
    H --> TH[Threshold tau = gamma / 1+exp(-H)]
    TH --> W[Directional weight w theta]
    A --> W
    W --> FF[Weighted complex spectrum]
    FF --> IF[IFFT2]
    IF --> FO[Fold + overlap averaging]
    FO --> MM[Per-image/channel min-max]
    MM --> G[Residual gate]
    I --> G
    G --> O[Output I prime]
```

Conceptual equations used by the figure:

\[
D_i^c(k)=\sum_{(u,v):b(u,v)=k} A_i^c(u,v),
\]

\[
p_i^c(k)=\frac{D_i^c(k)}{\sum_j D_i^c(j)+\varepsilon},
\]

\[
H_i^c=-\sum_kp_i^c(k)\log(p_i^c(k)+\varepsilon),
\]

\[
\tau_i^c=\frac{\gamma}{1+\exp(-H_i^c)},
\]

\[
w_i^c(k)=
\begin{cases}
0,&q_i^c(k)\le\tau_i^c,\\
q_i^c(k),&q_i^c(k)>\tau_i^c,
\end{cases}
\]

\[
I'=I+I\odot\operatorname{MinMax}(R_{AF2}(I)).
\]

## Planned seed flow

```text
seed 42   -> completed preliminary promotion screen
seed 123  -> planned confirmatory pair
seed 2026 -> planned confirmatory pair

final direct-AF2 claim
    requires synthesis across repeated seeds
    + final efficiency evidence
    + prospectively frozen test policy if/when test is opened
```
