# 10 — Reproducible Proposal Build Pipeline

Date: 2026-08-25

Status: **ACTIVE WORKING-ASSEMBLY PIPELINE**

This document records the first executable Git-driven proposal assembly path. It implements the previously frozen rule:

```text
Git = scientific/content source of truth
USU guideline/template = formatting authority
DOCX/PDF = generated review/release artifacts
```

## 1. Builder

Executable:

`tools/thesis/build_proposal_docx.py`

The builder reads the active proposal sources directly from the repository and assembles:

- Bab I from `02_BACKGROUND.md` + `03_PROBLEM_FORMULATION.md`;
- Bab II from `04_LITERATURE_REVIEW.md`, replacing its older embedded §2.2 with `04_02_INSPECTION_QUALITY_NORMALIZED.md` and its older §2.9 with `04_09_RELATED_WORK_TABLE.md`;
- Bab III from `05_METHODOLOGY.md`, replacing its embedded §3.5 with `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`;
- a working bibliography from citation keys resolved during assembly.

The source-hardened §3.5 is therefore the assembly authority, not the stale embedded copy.

## 2. Working citation conversion

The builder converts internal keys such as:

```text
[COF-01]
[FG-01, pp. 5–6, §3.3.3, Eq. (9)–(13)]
```

into working author-year prose such as:

```text
(Hong et al., 2026)
(Xu et al., 2025, pp. 5–6, §3.3.3, Eq. (9)–(13))
```

This conversion exists so internal repository keys do not leak into the Word review artifact.

Important: the current bibliography block is a **working metadata layer**, not the final APA-author-metadata authority. Full author ordering, initials, volume/issue/pages, DOI formatting, and the final minimum-reference audit must still be certified before submission release.

## 3. Word generation path

The build uses Pandoc for Markdown → DOCX conversion so that:

- headings become Word heading styles;
- Markdown tables become editable Word tables;
- TeX display equations are converted into native Word equation objects where supported;
- a Word TOC field is generated from heading levels.

A post-processing pass using `python-docx` then applies the campus working layout:

- A4;
- Times New Roman 12 pt body;
- 1.5 line spacing body;
- 1.27 cm first-line paragraph indent;
- 4 cm left margin;
- 3 cm top/right/bottom margins;
- bold 12 pt heading hierarchy;
- compact table typography;
- page numbering in the working review file.

## 4. GitHub Actions build

Workflow:

`.github/workflows/build-thesis-proposal.yml`

It builds three synchronized artifacts:

```text
Proposal_AF2_USU_Working.docx
Proposal_AF2_USU_Working.md
Proposal_AF2_USU_Working.pdf
```

The PDF exists for layout inspection; the DOCX remains the editable review artifact.

## 5. Working-release boundary

The current automated build is **not yet the final submission release**.

Open formatting tasks before final release:

1. reproduce the exact cover/front-matter composition from the supplied USU DOCX template;
2. implement Roman front-matter numbering and chapter-first-page numbering exactly according to the official guideline;
3. redraw and insert Figures 3.1–3.4;
4. verify table continuation/landscape behavior, especially the long related-work table;
5. verify every native Word equation and add chapter-based equation numbering;
6. finalize Table/Figure captions and automatic lists;
7. replace the working bibliography metadata with fully normalized APA records;
8. run the ≥50-reference / journal-percentage requirement audit if the same requirement is enforced for the proposal release;
9. visually inspect every rendered page before delivery.

No final artifact may be labeled `submission`, `sempro-final`, or equivalent until those gates close.

## 6. Content boundary

The build process must never silently:

- rewrite scientific claims;
- replace the hardened §3.5 with the older copy;
- merge historical D0-parent factorization into the final direct-from-pretrained confirmation path;
- change RQ1–RQ4;
- add a new AF2 candidate;
- choose parameter-sensitivity values from observed validation outcomes;
- convert pilot evidence into final evidence.

Content changes happen in the Markdown source first, are reviewed/committed, and only then regenerate the document.

## 7. Revision workflow

```text
advisor feedback / content correction
        ↓
edit canonical Markdown / audit source
        ↓
commit
        ↓
automated working DOCX/PDF build
        ↓
render / visual QA
        ↓
review copy
```

This prevents the Word file from becoming an untraceable parallel source of scientific truth.
