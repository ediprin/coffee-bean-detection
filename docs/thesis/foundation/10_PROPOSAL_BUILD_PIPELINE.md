# 10 — Reproducible Proposal Build Pipeline

Date: 2026-08-25

Status: **ACTIVE QA-HARDENED WORKING-ASSEMBLY PIPELINE**

This document records the executable Git-driven proposal assembly path. It implements the frozen rule:

```text
Git = scientific/content source of truth
SPs official guide = normative formatting authority
supplied DOCX = structural/style scaffold
DOCX/PDF = generated review/release artifacts
```

## 1. Authoritative working builder

Current executable:

`tools/thesis/build_proposal_docx_v2.py`

The earlier `tools/thesis/build_proposal_docx.py` remains as the first-generation assembly implementation and a dependency imported by v2. New workflow changes should target **v2** unless an explicit migration is being made.

The builder reads the active proposal sources directly from the repository and assembles:

- Bab I from `02_BACKGROUND.md` + `03_PROBLEM_FORMULATION.md`;
- Bab II from `04_LITERATURE_REVIEW.md`, replacing its older embedded §2.2 with `04_02_INSPECTION_QUALITY_NORMALIZED.md` and its older §2.9 with `04_09_RELATED_WORK_TABLE.md`;
- Bab III from `05_METHODOLOGY.md`, replacing its embedded §3.5 with `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`;
- a working bibliography from citation keys resolved during assembly.

The source-hardened §3.5 is therefore the assembly authority, not the stale embedded copy.

## 2. Working citation conversion and build-time guards

Internal keys such as:

```text
[COF-01]
[FG-01, pp. 5–6, §3.3.3, Eq. (9)–(13)]
```

are converted to working author-year forms such as:

```text
(Hong et al., 2026)
(Xu et al., 2025, pp. 5–6, §3.3.3, Eq. (9)–(13))
```

V2 additionally:

1. merges adjacent parenthetical groups generated from adjacent internal citation keys;
2. fails if a used citation key has no bibliography record;
3. fails if internal citation tokens survive assembly;
4. warns when the number of genuinely cited keyed sources remains below 50 instead of padding the bibliography;
5. strips working Markdown separators and fence delimiters;
6. converts unordered Markdown bullet markers to ordered-list syntax for formal assembly.

Important: the current bibliography block is still a **working metadata layer**, not the final APA authority. Final author ordering, initials, volume/issue/pages, DOI formatting, hanging indents, and the final reference-count/journal-percentage audit remain separate release gates.

Primary-source corrections already enforced by v2 include:

- de Oliveira et al. DOI `10.1016/j.jfoodeng.2015.10.009`;
- García et al. 2019 record with DOI `10.3390/app9194195`.

The de Oliveira `.009` value intentionally overrides older master-map rows containing a conflicting DOI and should later be propagated back to the canonical registry.

## 3. Word generation path

The build uses Pandoc for Markdown → DOCX conversion so that:

- headings become Word heading styles;
- Markdown tables become editable Word tables;
- TeX display equations become native Word equation objects where supported;
- a Word TOC field is generated from heading levels.

V2 supports:

```text
--reference-doc <campus-template.docx>
```

for a formal local build using the supplied campus scaffold. When `--reference-doc` is omitted, the builder prints an explicit warning and the result remains a **working build**, not a template-faithful release.

Post-processing then applies/normalizes:

- A4;
- Times New Roman 12 pt main text;
- 1.5 line spacing body;
- 1.27 cm first-line paragraph indent;
- left margin 4 cm;
- top/right/bottom margins 3 cm;
- two-line centered chapter headings (`BAB I` / `PENDAHULUAN`);
- ordinary table-caption paragraphs outside the TOC;
- table-title punctuation consistent with the official table rule;
- less aggressive table compaction than the first-generation builder.

## 4. Figures 3.1–3.4

Figure insertion is a separate deterministic pass:

`tools/thesis/insert_proposal_figures.py`

It inserts:

- Figure 3.1 — research framework;
- Figure 3.2 — Native YOLO26 vs AF2–YOLO26;
- Figure 3.3 — AF2 operator;
- Figure 3.4 — AF2 optimization genealogy and confirmatory experiment.

Working figure captions are centered below the figure, TNR 12, title case, and have no terminal period. Because the official-guide excerpt does not explicitly settle punctuation immediately after the figure number, the current separator follows the supplied campus scaffold (`Gambar 3.1. Judul`).

## 5. Equation numbering

Display-equation numbering is a dedicated pass:

`tools/thesis/apply_equation_numbers.py`

The pass detects Pandoc-generated Office Math display paragraphs (`m:oMathPara`). It deliberately ignores inline mathematics. For each display equation it creates a borderless three-column row with:

```text
balancing gutter | centered native Word equation | (bab.nomor)
```

The portrait body width is fixed to the SPs text area of 14 cm, and the equation number is positioned in a dedicated right-aligned column near the right text boundary. Numbering restarts by chapter, e.g. `(2.1)`, `(2.2)`, `(3.1)`.

The mechanism was tested on a native-OMML synthetic DOCX and rendered successfully with same-line labels `(3.1)` and `(3.2)`.

## 6. Section-aware pagination

Pagination is applied after figures and equation numbering:

`tools/thesis/apply_sps_pagination.py`

The pass inserts a next-page section break before each chapter and implements:

```text
front matter
    -> lowercase Roman
    -> center bottom

Bab I onward
    -> Arabic
    -> normal pages: top-right
    -> chapter-opening page: bottom-right
```

It also normalizes every section to A4 with official 4/3/3/3 cm margins and sets header/footer distance to 1.5 cm.

The pagination mechanism was prototyped on a synthetic multi-page DOCX and rendered before being committed. That test confirmed Roman front numbering and distinct chapter-opening/ordinary body placement. The **actual proposal** still requires full rendered-page verification.

## 7. Landscape isolation for Tabel 2.1

The long six-column related-work table is isolated after chapter pagination:

`tools/thesis/apply_landscape_tables.py`

The pass:

1. finds the unique `Tabel 2.1` caption and the immediately following table;
2. inserts next-page section breaks before the caption and after the table;
3. sets the table section to A4 landscape with the same 4/3/3/3 cm margins;
4. continues Arabic page numbering with a top-right header;
5. returns the following Bab II content to portrait orientation;
6. assigns an explicit 22.7 cm column-width budget across the six columns;
7. repeats the header row for multi-page continuation.

This mixed-orientation mechanism was tested on a synthetic multi-chapter DOCX **after** SPs pagination. The render confirmed continuous Arabic numbering across portrait → landscape → portrait pages and preserved the bottom-right first-page number on the following chapter opening. The real table still needs page-by-page inspection because its row heights and wrapping depend on actual prose.

## 8. GitHub Actions working build

Workflow:

`.github/workflows/build-thesis-proposal.yml`

Current sequence:

```text
checkout
  ↓
install Pandoc / LibreOffice / SVG tools / python-docx
  ↓
build_proposal_docx_v2.py
  ↓
SVG → PNG
  ↓
insert Figures 3.1–3.4
  ↓
apply_equation_numbers.py
  ↓
apply_sps_pagination.py
  ↓
apply_landscape_tables.py
  ↓
DOCX → PDF
  ↓
upload working artifacts
```

Artifacts:

```text
Proposal_AF2_USU_Working.docx
Proposal_AF2_USU_Working.md
Proposal_AF2_USU_Working.pdf
```

The PDF exists for layout inspection; the DOCX remains the editable review artifact.

CI currently does **not** possess the uploaded binary campus template, so it intentionally builds without `--reference-doc`. A template-faithful release must be generated in an environment where the supplied DOCX is available.

## 9. Working-release boundary

The automated build is **not yet the final submission release**.

Current release blockers are now concentrated in actual-document QA rather than missing static layout mechanisms:

1. run the actual proposal through the supplied Word scaffold using `--reference-doc`;
2. render the actual proposal and inspect every page;
3. verify real Tabel 2.1 row heights, wrapping, repeated headers, and page breaks;
4. verify all real equation widths and `(bab.nomor)` alignment;
5. verify Roman/Arabic page-number continuity on the real assembled document;
6. normalize final APA bibliography metadata and narrative-vs-parenthetical citations;
7. refresh TOC/list fields and verify their rendered page numbers.

No artifact may be labeled `submission`, `sempro-final`, or equivalent until those gates close.

## 10. Content boundary

The build process must never silently:

- rewrite scientific claims;
- replace the hardened §3.5 with the older copy;
- merge historical D0-parent factorization into the final direct-from-pretrained confirmation path;
- change RQ1–RQ4;
- add a new AF2 candidate;
- choose parameter-sensitivity values from observed validation outcomes;
- convert pilot evidence into final evidence.

Content changes happen in the Markdown source first, are reviewed/committed, and only then regenerate the document.

## 11. Revision workflow

```text
advisor feedback / content correction
        ↓
edit canonical Markdown / audit source
        ↓
commit
        ↓
QA-hardened working DOCX/PDF build
        ↓
render / visual QA
        ↓
review copy
```

This prevents the Word file from becoming an untraceable parallel source of scientific truth.
