# Bab III AF2 Page-Locator Recertification — 2026-08-25

Status: **locator addendum** to `BAB3_PRIMARY_SOURCE_HARDENING_2026-08-25.md` and `proposal/05_05_AF2_PRIMARY_SOURCE_HARDENED.md`.

This addendum supersedes only the earlier statements that the printed page containing the latter AFAB-2 equations had not yet been re-located. It does **not** weaken the existing rule that mathematical transcription must be checked against the primary PDF and executable repository implementation.

---

## 1. Primary source

`[FG-01]` — Xu et al. (2025), *More signals matter to detection: Integrating language knowledge and frequency representations for boosting fine-grained aircraft recognition*, **Neural Networks 187 (2025) 107402**, DOI `10.1016/j.neunet.2025.107402`.

---

## 2. Recertified AFAB-2 location

The primary PDF retrieval now re-exposes the continuation of **§3.3.3 Patch-specific chaotic amplitude suppressor** on printed **page 6**. The recovered text begins immediately after the threshold/weight equations with:

- the statement that `γ` is a hyperparameter;
- the explanation that amplitude is adjusted while spatial structure is preserved;
- min–max normalization of the recovered domain;
- element-wise gating between raw and recovered spatial domains;
- residual formation of the enhanced spatial domain;
- followed by the start of §3.4 CGFI on the same printed page.

This is sufficient to recertify the page location of the **latter AFAB-2 equation block (Eq. 10–13) and its immediately following gating/residual explanation as p. 6, §3.3.3**.

The parent-method locator for Bab III may therefore be written as:

```text
Xu et al. [FG-01], pp. 5–6, §3.3.3, Eq. (9)–(13)
```

with the following precision:

- Eq. (9) introduces angular density;
- Eq. (10)–(13) are the subsequent entropy/threshold/angular-suppression formulation in the AFAB-2 subsection;
- the min–max/gating/residual prose directly following those equations is on p. 6.

---

## 3. Transcription boundary remains

The locator is now closed, but **page location is not the same thing as permission to reconstruct equations from memory**.

For the thesis implementation, the executable repository remains the authority for the exact transferred discrete formula:

- `af2_entropy_threshold(...)` is annotated as the AFAB-2 Eq. (10)–(11) mapping;
- `_af2_weight(...)` is annotated as the Eq. (9)–(13) mapping;
- the repository freezes `gamma=0.10`, 360 discrete angular bins, hard thresholding, independent RGB handling, FP32 FFT, overlap folding, and the exact residual gate.

Accordingly, final proposal prose should use the combined provenance:

```text
parent method / equation family:
    Xu et al. [FG-01], pp. 5–6, §3.3.3, Eq. (9)–(13)

exact coffee-transfer implementation:
    src/coffee_detector/afab/operator.py
    configs/afab/AF2_yolo26n_chaotic_amplitude.yaml
```

This avoids falsely attributing repository-specific discretization and engineering decisions to the parent paper.

---

## 4. Assembly corrections

During final assembly, replace older open-locator wording in the source-hardened §3.5 with:

> Xu et al. menjelaskan mekanisme AFAB-2 pada §3.3.3, pp. 5–6, Eq. (9)–(13). Implementasi penelitian memetakan keluarga persamaan tersebut ke operator diskret yang dibekukan pada repository; detail diskretisasi dan engineering tidak diklaim identik dengan implementasi parent paper.

Also update the method-origin table locator to:

```text
angular density / adaptive angular suppression:
FG-01, pp. 5–6, §3.3.3, Eq. (9)–(13)
```

---

## 5. Gate status

| Gate | Status |
|---|---|
| AFAB-2 subsection located | PASS |
| printed page for latter Eq. (10)–(13) block located | **PASS — p. 6** |
| post-equation min–max/gating/residual prose located | **PASS — p. 6** |
| exact transfer implementation traceable | PASS — repository operator/config |
| final equation typesetting visual proofread in campus DOCX/PDF | PENDING ASSEMBLY QA |

The remaining item is ordinary document/typesetting QA, not an unresolved literature-evidence gap.