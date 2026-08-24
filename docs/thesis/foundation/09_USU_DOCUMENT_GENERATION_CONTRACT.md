# 09 — USU Document Generation Contract

Date: 2026-08-25

Status: **FORMAT AUTHORITY FROZEN BEFORE DOCX ASSEMBLY**

This contract defines how proposal/thesis content in Git is converted into the official USU Word format. It is based on two user-supplied files:

1. `Pedoman Tesis & Disertasi SPs.pdf` — Sekolah Pascasarjana Universitas Sumatera Utara, 2023.
   - local SHA-256: `0b8592ffe7cac8d804779d5cd3ae3c606a1b5d711d3e4147c15f7a74ab85cdf1`
2. `Format Penulisan Tesis.docx` — official/working thesis-format document supplied by the user.
   - local SHA-256: `f91e078e5226a98101873b409b7331519c55c0c1b1d6852b9800b9aa07de0f28`

The PDF is treated as the **normative formatting authority** when it conflicts with the editable DOCX. The DOCX is treated as the **layout/style scaffold and insertion target**.

---

## 1. Source-of-truth hierarchy

For generated documents, authority order is:

```text
1. USU 2023 thesis/dissertation guideline PDF
2. official editable thesis-format DOCX supplied by the user
3. thesis foundation decisions in docs/thesis/foundation/
4. proposal chapter Markdown in docs/thesis/proposal/
5. generation scripts/build configuration
```

Scientific content is edited and versioned in Git. Word is a generated/review artifact, not the primary scientific source of truth.

---

## 2. Core document format

The following rules are frozen from the 2023 SPs guideline:

- paper: A4, 21 cm x 29.7 cm;
- body font: Times New Roman 12 pt;
- normal body line spacing: 1.5;
- single spacing is used for abstract, titles/subtitles where specified, table of contents, list of tables, list of figures, list of appendices, table titles/content, figure titles, and bibliography;
- margins:
  - left: 4 cm;
  - top: 3 cm;
  - right: 3 cm;
  - bottom: 3 cm;
- new paragraph first-line indentation: 1.27 cm in the supplied DOCX implementation;
- chapter title: uppercase, bold, centered, starts on a new page;
- subchapter title: left aligned, bold, title case subject to Indonesian function-word rules;
- lower subchapter title: left aligned, bold, sentence/title capitalization according to the official hierarchy;
- hyphen/bullet lists should not be used where the guideline requires ordered downward detail; use numeric/alphabetic hierarchy.

### Important template typo resolved

The editable DOCX literally says `indent ... 1,27 pt`, but its actual paragraph property is **1.27 cm**. The generator must preserve the actual 1.27 cm first-line indentation, not 1.27 pt.

---

## 3. Known conflicts between guideline and supplied DOCX

The supplied DOCX is useful but not fully internally consistent with the 2023 PDF.

### 3.1 Main-body margins

The first sections of the DOCX use approximately 4/3/3/3 cm (left/top/right/bottom), but later sections use approximately:

```text
left 3.5 cm
right 2.5 cm
bottom 2.5 cm
top 3 cm
```

The generated proposal/thesis must normalize body sections to the PDF authority:

```text
left 4 cm
right 3 cm
top 3 cm
bottom 3 cm
```

unless the program formally provides a later superseding instruction.

### 3.2 Page-number placement

The guideline requires:

- preliminary pages: lowercase Roman numerals, centered at the bottom;
- main/ending sections: Arabic numerals;
- ordinary main-text pages: top-right;
- pages beginning with a chapter heading: bottom-right.

The supplied DOCX currently renders some main-body page numbers at bottom-center. The generator must follow the PDF rule rather than blindly preserving this template artifact.

### 3.3 Table/figure labels

The DOCX contains examples with punctuation that is not perfectly consistent across pages. The generator must follow the explicit SPs rules:

- tables numbered by chapter, e.g. `Tabel 2.1`;
- figures numbered by chapter, e.g. `Gambar 3.2`;
- table title above the table;
- figure title below the figure;
- no final period after a simple title;
- cited/source-derived illustrations must show the source;
- every table and figure must be referred to in the preceding/nearby prose.

---

## 4. Page numbering contract

The builder must create separate Word sections so numbering can follow the official scheme.

### Preliminary section

```text
lowercase Roman numerals
i, ii, iii, ...
centered bottom
```

### Main body

```text
Arabic numerals
1, 2, 3, ...
```

Placement:

- chapter-opening page -> bottom-right;
- other body page -> top-right.

This requires chapter-start section/page control rather than one global footer.

---

## 5. Proposal-mode assembly

The supplied DOCX is a **full thesis template**, while the current deliverable is a thesis proposal. Proposal generation must therefore use the same formatting system but include only proposal-relevant components.

Default proposal release structure:

```text
Title / proposal cover
required approval page(s), if requested by program/advisor
Daftar Isi
Daftar Tabel, if used
Daftar Gambar, if used

BAB I PENDAHULUAN
BAB II TINJAUAN PUSTAKA
BAB III METODE PENELITIAN
DAFTAR PUSTAKA
optional LAMPIRAN
```

Abstract, originality declaration, final thesis examiner pages, final graduation date, results chapter, conclusions chapter, and final-thesis biographical front matter are not inserted into a proposal release unless explicitly required by the program/advisor.

The campus proposal example may still guide chapter presentation, but the official USU formatting files remain the formatting authority.

---

## 6. Title-page contract

The supplied DOCX requires the thesis title to be uppercase and states a minimum of 15 words and maximum of 20 words.

Current working title:

`Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi`

is treated as **15 words** under the template's normal hyphenated-term word counting, so it fits the stated 15–20 word range.

Before a formal release, the builder must populate actual:

- student name;
- NIM / program abbreviation;
- program name;
- year;
- supervisor/approval fields required for the proposal stage.

Unknown identity/approval fields must remain explicit placeholders; they must not be guessed.

---

## 7. Citation and bibliography contract

The 2023 guideline requires a consistent **APA Style / author-year** approach and recommends Mendeley. The guide also states for a thesis:

- at least 50 reference materials;
- at least 40% of them research-journal references.

For proposal generation, citations in Git such as `[COF-01]`, `[FG-01]`, etc. are internal drafting keys only.

The DOCX release must convert them into human-readable author-year citations, for example:

```text
(Hong et al., 2026)
Xu et al. (2025)
```

The bibliography must contain only works actually cited in the generated document and must not expose internal reference IDs.

A canonical bibliography registry should be maintained separately from prose so one reference has one authoritative metadata record.

---

## 8. Equation contract

Equations in the DOCX must be native editable Word equations where practical, not raster screenshots.

Rules:

- equation centered;
- equation number aligned to the right;
- numbering by chapter, e.g. `(3.1)`, `(3.2)`;
- symbols explained immediately after the equation when required;
- cited/modified equations must state the source;
- author-developed equations should be labelled honestly as author formulation/derivation where appropriate.

For AF2, the generator should preserve the distinction between:

1. equations inherited from the parent method/paper;
2. equations expressing the repository implementation/adaptation;
3. thesis-specific diagnostic/statistical definitions.

---

## 9. Table contract

The builder must generate native Word tables.

Rules:

- title above table;
- numbered by chapter;
- single-spaced internal text;
- centered relative to text area;
- repeated heading rows for tables spanning multiple pages;
- continuation label where required;
- sources and table-specific notes below table;
- avoid excessive vertical/internal borders where the official example favors economical scientific-table styling;
- wide tables may use landscape sections when necessary.

Main proposal candidates include:

- coffee-literature comparison table;
- dataset summary;
- AF2 structural candidate table;
- training configuration table;
- evaluation-metric table;
- experimental scenario table.

---

## 10. Figure contract

The builder must place figures near their first substantive discussion.

Rules:

- centered;
- numbered by chapter;
- caption below;
- every figure referenced in prose;
- readable after scaling;
- consistent font with document;
- source stated if reproduced/adapted from another work;
- original thesis diagrams need no external source label.

Planned proposal figures include:

```text
Gambar 3.1 Kerangka penelitian
Gambar 3.2 Arsitektur native YOLO26 dan AF2-YOLO26
Gambar 3.3 Alur internal preprocessing AF2
Gambar 3.4 Strategi analisis dan optimasi AF2
```

Later thesis/result figures may include spectral visualizations, angular density, AF2 reconstruction, CAM/activation visualizations, confusion/error analysis, and efficiency plots.

---

## 11. Automated assembly strategy

The preferred pipeline is:

```text
Git Markdown scientific content
        |
        v
source/citation normalization
        |
        v
USU DOCX template clone
        |
        +-- replace chapter placeholders
        +-- insert native tables
        +-- insert figures + captions
        +-- insert Word equations
        +-- create section/page-number logic
        +-- build/update TOC/list fields
        +-- generate bibliography
        v
working DOCX
        |
        v
render every page to PNG
        |
        v
visual QA + fix
        |
        v
proposal release DOCX
```

The original user-supplied DOCX should never be destructively edited. Every generated release starts from a clean clone/template base.

---

## 12. Git/document lifecycle

Recommended artifact roles:

```text
Git Markdown
= scientific source of truth

USU master DOCX
= formatting scaffold

working DOCX
= supervisor-review artifact

release DOCX
= seminar/submission artifact
```

Advisor edits in Word should be reconciled back into Git before the next generated release whenever they change scientific content.

Suggested release names:

```text
proposal_AF2_sempro_v0.1.docx
proposal_AF2_sempro_v0.2.docx
proposal_AF2_sempro_v1.0.docx
```

Avoid using filename chains such as `final_fix_revisi_terbaru(3).docx` as version control.

---

## 13. Visual QA gate

No generated DOCX is considered ready until it is rendered and every page is visually inspected.

QA checklist:

- margins correct;
- no clipped text;
- no overlapping objects;
- no orphan chapter/subchapter headings;
- no broken equations;
- tables inside printable area;
- landscape sections return correctly to portrait;
- figure captions attached to the intended figure;
- correct Roman/Arabic page numbering and placement;
- TOC/list entries align with actual pages;
- no internal citation keys or tool markers leak into the document;
- bibliography uses one consistent final style.

After any layout-sensitive DOCX edit, render-and-review must be repeated.

---

## 14. Immediate implication for current proposal work

The next proposal-content step remains the Hong-aligned Bab III rewrite. Document assembly should begin only after Bab III's scientific structure is frozen enough that the first Word release will not immediately become obsolete.

However, the formatting path is now fixed:

```text
USU guideline PDF = normative rules
USU editable DOCX = master scaffold
Git = scientific content/version history
DOCX builder = reproducible assembly layer
rendered pages = final QA authority
```
