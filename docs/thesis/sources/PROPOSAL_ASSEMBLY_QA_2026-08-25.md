# Proposal Formal-Assembly QA — 2026-08-25

Status: **live static QA before full rendered-document inspection**.

Purpose: compare the current Git-driven DOCX builder against the uploaded SPs thesis/dissertation writing guide and the supplied Word scaffold. This audit distinguishes items already implemented in the build pipeline from items that still require visual verification on the actual assembled proposal.

## 1. Current verdict

\[
\boxed{\text{CONTENT ASSEMBLY READY; STATIC FORMAT HARDENING ADVANCED; FULL RENDER QA STILL OPEN}}
\]

The scientific chapter assembly is stable. The build pipeline now implements a substantial subset of the campus formatting rules, including section-aware pagination. A formal release is still blocked by actual full-document render inspection, wide-table handling, equation numbering, template-faithful local generation, and final citation/bibliography normalization.

## 2. Normative authority

When the Word scaffold and the official guide conflict, use:

1. official SPs thesis/dissertation writing guide as the normative formatting authority;
2. supplied `Format Penulisan Tesis.docx` as the structural/style scaffold;
3. Git Markdown as the scientific-content source of truth.

Key rules already extracted from the official guide:

- A4;
- Times New Roman 12 pt for main text;
- 1.5 spacing for body text, with one-spacing exceptions for captions/tables/bibliography and other listed elements;
- margins: left 4 cm, top/right/bottom 3 cm;
- first-line paragraph indent approximately 1.27 cm;
- chapter titles centered, uppercase, and each chapter begins on a new page;
- body page numbering uses Arabic numerals; normal pages are top-right, while chapter-opening pages are bottom-right;
- front matter uses lowercase Roman numerals centered at the bottom;
- no bullet-symbol lists in the formal manuscript; downward details use numbers/letters;
- table/figure numbering is chapter-based;
- figure caption is below the figure, centered/symmetric, title case, and has no terminal period;
- the official guide excerpt does not explicitly settle separator punctuation after the **figure number**; the supplied Word scaffold demonstrates `Gambar 2.1. Grafik X`, so the working proposal follows that separator convention until stronger evidence is found;
- table numbering is explicit: no period after the table number, with two spaces before the title and no terminal period;
- tables that are too wide for portrait should use landscape orientation;
- equations use chapter-based numbering positioned toward the right margin.

## 3. Static QA findings — refreshed after v0.14/v0.15

| ID | Finding | Severity | Current status | Required action / evidence |
|---|---|---|---|---|
| QA-01 | Builder originally generated DOCX without the supplied DOCX scaffold | BLOCKER for formal release | **PARTIAL** | `build_proposal_docx_v2.py` now supports `--reference-doc`. CI still omits the binary template, so CI output remains a working build. Formal local release must supply the uploaded scaffold. |
| QA-02 | Page-number layout previously used one Arabic footer rule everywhere | BLOCKER | **IMPLEMENTED; FULL-RENDER VERIFY** | `apply_sps_pagination.py` now creates Roman front matter and Arabic chapter sections, with top-right normal-page and bottom-right chapter-opening placement. Mechanism passed a synthetic render test; actual proposal still needs page-by-page verification. |
| QA-03 | Chapter headings were previously one line (`BAB I PENDAHULUAN`) | MAJOR | **FIXED IN V2** | V2 splits Heading 1 into two centered lines: `BAB I` and `PENDAHULUAN`; actual proposal render still needs visual confirmation. |
| QA-04 | Figure-caption separator rule was initially overstated | MAJOR | **CORRECTED** | Official guide supports below/centered/title-case/no-terminal-period; supplied scaffold shows `Gambar 2.1. Grafik X`. Working captions now follow that scaffold convention. |
| QA-05 | Figure captions were forced to TNR 10 | MAJOR | **FIXED** | Figure insertion now uses TNR 12, centered, single-spaced. |
| QA-06 | Related-work Table 2.1 has six columns and long prose; portrait overflow risk is high | BLOCKER pending render | **OPEN** | Render actual proposal; if width is excessive, isolate Table 2.1 in a landscape section as required by the guide. |
| QA-07 | All tables were previously forced to 9 pt | MAJOR | **PARTIAL** | V2 uses 11 pt for ordinary tables and 9.5 pt only for 6+ column tables as a working compromise. Actual render/template comparison must decide final sizing. |
| QA-08 | Markdown table titles could enter the TOC as headings | MAJOR | **FIXED IN V2** | Table-title headings are demoted to ordinary centered caption paragraphs before Word generation. |
| QA-09 | Markdown horizontal rules could become visible separator lines | MAJOR | **FIXED IN V2** | Formal assembly strips `---` lines. |
| QA-10 | Source modules contain unordered Markdown bullets | MAJOR | **FIXED AT ASSEMBLY PASS** | V2 converts unordered Markdown list markers to ordered-list syntax before Pandoc. Final render must still confirm indentation/numbering hierarchy. |
| QA-11 | Fenced text/code blocks could render as monospaced code | MAJOR | **PARTIAL** | Fence delimiters are stripped. Explanatory flow text that remains after fence removal still requires visual/prose QA; major flows are already represented by Figures 3.1–3.4. |
| QA-12 | Display equations do not yet have final chapter-based equation numbers | BLOCKER for formal release | **OPEN** | Add a numbering pass after OMML generation and place `(bab.nomor)` at the right boundary according to the guide. |
| QA-13 | Adjacent internal citation groups could become adjacent parenthetical citations | MAJOR | **FIXED IN V2** | Adjacent author-year groups are merged in the assembled Markdown. |
| QA-14 | Narrative prose can still duplicate author names after automatic parenthetical conversion | MAJOR | **OPEN** | Add narrative-vs-parenthetical citation handling or normalize sentences before release. |
| QA-15 | Missing bibliography records were silently skipped | BLOCKER | **FIXED IN V2** | Build now fails if any used citation key lacks a bibliography record. |
| QA-16 | Working bibliography strings are not guaranteed to be complete final APA records | BLOCKER for final bibliography | **OPEN** | Replace ad-hoc `BIB` strings with the canonical reference registry/BibTeX/CSL after the final reference audit. |
| QA-17 | Current citation count may remain below the final program reference-count requirement | WARNING | **ACTIVE WARNING** | Builder emits a warning if fewer than 50 unique keyed sources are cited. Do not pad references artificially; close only with genuinely used sources. |
| QA-18 | de Oliveira DOI conflicted between primary PDF and some master-map rows | BLOCKER citation integrity | **FIXED IN BUILDER; REGISTRY CONFLICT OPEN** | V2 uses primary-PDF DOI `10.1016/j.jfoodeng.2015.10.009`. Correct the master registry separately. |
| QA-19 | `COF-17` García 2019 was cited but absent from working builder bibliography | BLOCKER | **FIXED IN V2** | Complete working record added with DOI `10.3390/app9194195`. |

## 4. Implemented low-risk hardening

The following changes are now active in the build path:

1. source-key bibliography completeness check;
2. unresolved citation-token check;
3. de Oliveira primary-PDF DOI correction in builder;
4. García 2019 bibliography record;
5. adjacent citation-group merging;
6. table-caption demotion and punctuation normalization;
7. horizontal-rule removal;
8. unordered-list conversion;
9. fenced-code delimiter removal;
10. two-line chapter headings;
11. TNR 12 figure captions with scaffold-consistent separator punctuation;
12. section-aware Roman/Arabic page numbering;
13. A4 + official margin normalization on pagination sections.

None of these modifications changes the ML method, dataset, experiment, or reported pilot result.

## 5. Pagination implementation evidence

`tools/thesis/apply_sps_pagination.py` was tested on a synthetic multi-section DOCX before being added to the workflow. The rendered synthetic document confirmed the intended mechanism:

- front page: lowercase Roman `i`, centered at bottom;
- Bab I opening page: Arabic `1`, bottom-right;
- following Bab I page: Arabic `2`, top-right;
- A4 section geometry with left 4 cm and top/right/bottom 3 cm margins.

This validates the implementation mechanism, not the final proposal layout. The assembled proposal must still be rendered and inspected because actual tables, figures, TOC fields, section breaks, and long chapter content can interact differently.

## 6. Items still blocked on actual rendered-layout iteration

Do not claim these are solved before the actual proposal DOCX is rendered:

- landscape handling for Table 2.1;
- full-proposal verification of Roman/Arabic pagination and section continuity;
- exact chapter-based equation-number alignment;
- widow/orphan behavior;
- figure/table page breaks;
- table continuations if a table spans multiple pages;
- full APA typography/hanging indents;
- narrative citation duplication cleanup;
- final TOC/page-field refresh;
- template-faithful comparison against the supplied campus DOCX.

## 7. Citation-integrity discrepancy retained for registry maintenance

The primary PDF for de Oliveira et al. states:

`DOI: 10.1016/j.jfoodeng.2015.10.009`.

Some existing master-map rows store `.030`. The current builder now follows the primary PDF. The master reference registry must be corrected in a dedicated evidence-maintenance pass so the conflict does not reappear when the ad-hoc builder dictionary is replaced by a canonical bibliography database.

## 8. Shipping rule

A proposal DOCX may be called a **working build** after static source checks pass.

A proposal DOCX may be called a **format-checked release** only after:

1. generated DOCX exists;
2. canonical renderer produces page images;
3. every page is visually inspected at full size;
4. all clipping, overflow, heading, caption, equation, pagination, and table issues are corrected;
5. the document is re-rendered after the final correction.

No `sempro-final` label is allowed before this gate.
