# Proposal Formal-Assembly QA — 2026-08-25

Status: **live static QA before full rendered-document inspection**.

Purpose: compare the current Git-driven DOCX builder against the uploaded SPs thesis/dissertation writing guide and the supplied Word scaffold. This audit distinguishes items already implemented and synthetic-render-tested from items that still require visual verification on the actual assembled proposal.

## 1. Current verdict

\[
\boxed{\text{CONTENT ASSEMBLY READY; STATIC FORMAT HARDENING NEAR-COMPLETE; FULL PROPOSAL RENDER QA OPEN}}
\]

The scientific chapter assembly is stable. The build pipeline now implements section-aware pagination, chapter-based display-equation numbering, and landscape isolation for the long related-work table. These mechanisms have been render-tested on synthetic DOCX fixtures. A formal release is still blocked by actual full-document render inspection, template-faithful generation with the supplied campus scaffold, narrative-citation cleanup, and final APA bibliography normalization.

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
- the supplied Word scaffold demonstrates `Gambar 2.1. Grafik X`, so the working proposal follows that separator convention;
- table numbering is explicit: no period after the table number, with two spaces before the title and no terminal period;
- tables that are too wide for portrait should use landscape orientation;
- equation numbers use Arabic numerals in parentheses, follow the chapter number, and sit near the right text boundary.

## 3. Static QA findings — refreshed after v0.16

| ID | Finding | Severity | Current status | Required action / evidence |
|---|---|---|---|---|
| QA-01 | Builder originally generated DOCX without the supplied DOCX scaffold | BLOCKER for formal release | **PARTIAL** | `build_proposal_docx_v2.py` supports `--reference-doc`. CI still omits the binary template, so CI output remains a working build. Formal local release must supply the uploaded scaffold. |
| QA-02 | Page-number layout previously used one Arabic footer rule everywhere | BLOCKER | **IMPLEMENTED; SYNTHETIC RENDER PASS; FULL-RENDER VERIFY** | `apply_sps_pagination.py` creates Roman front matter and Arabic chapter sections with top-right normal-page and bottom-right chapter-opening placement. |
| QA-03 | Chapter headings were previously one line (`BAB I PENDAHULUAN`) | MAJOR | **FIXED IN V2** | V2 splits Heading 1 into two centered lines: `BAB I` and `PENDAHULUAN`; actual proposal render still needs visual confirmation. |
| QA-04 | Figure-caption separator rule was initially overstated | MAJOR | **CORRECTED** | Official guide supports below/centered/title-case/no-terminal-period; supplied scaffold shows `Gambar 2.1. Grafik X`. Working captions follow that scaffold convention. |
| QA-05 | Figure captions were forced to TNR 10 | MAJOR | **FIXED** | Figure insertion uses TNR 12, centered, single-spaced. |
| QA-06 | Related-work Table 2.1 has six columns and long prose; portrait overflow risk is high | BLOCKER | **IMPLEMENTED; SYNTHETIC MIXED-ORIENTATION RENDER PASS; FULL-RENDER VERIFY** | `apply_landscape_tables.py` isolates Tabel 2.1 in a next-page landscape section, sets 22.7 cm column budget, repeats the header row, and restores portrait layout afterward while continuing Arabic pagination. |
| QA-07 | All tables were previously forced to 9 pt | MAJOR | **PARTIAL** | V2 uses 11 pt for ordinary tables; the landscape Tabel 2.1 uses 9.5 pt with explicit widths. Actual render/template comparison must decide final sizing. |
| QA-08 | Markdown table titles could enter the TOC as headings | MAJOR | **FIXED IN V2** | Table-title headings are demoted to ordinary centered caption paragraphs before Word generation. |
| QA-09 | Markdown horizontal rules could become visible separator lines | MAJOR | **FIXED IN V2** | Formal assembly strips `---` lines. |
| QA-10 | Source modules contain unordered Markdown bullets | MAJOR | **FIXED AT ASSEMBLY PASS** | V2 converts unordered Markdown list markers to ordered-list syntax before Pandoc. Final render must still confirm indentation/numbering hierarchy. |
| QA-11 | Fenced text/code blocks could render as monospaced code | MAJOR | **PARTIAL** | Fence delimiters are stripped. Explanatory flow text that remains after fence removal still requires visual/prose QA; major flows are represented by Figures 3.1–3.4. |
| QA-12 | Display equations did not have chapter-based equation numbers | BLOCKER for formal release | **IMPLEMENTED; SYNTHETIC RENDER PASS; FULL-RENDER VERIFY** | `apply_equation_numbers.py` detects Pandoc `m:oMathPara` blocks, converts each to a borderless three-column equation row, centers the native Word equation, and places `(bab.nomor)` near the right boundary. Inline math is intentionally not numbered. |
| QA-13 | Adjacent internal citation groups could become adjacent parenthetical citations | MAJOR | **FIXED IN V2** | Adjacent author-year groups are merged in the assembled Markdown. |
| QA-14 | Narrative prose can still duplicate author names after automatic parenthetical conversion | MAJOR | **OPEN** | Add narrative-vs-parenthetical citation handling or normalize sentences before release. |
| QA-15 | Missing bibliography records were silently skipped | BLOCKER | **FIXED IN V2** | Build fails if any used citation key lacks a bibliography record. |
| QA-16 | Working bibliography strings are not guaranteed to be complete final APA records | BLOCKER for final bibliography | **OPEN** | Replace ad-hoc `BIB` strings with the canonical reference registry/BibTeX/CSL after the final reference audit. |
| QA-17 | Current citation count may remain below the final program reference-count requirement | WARNING | **ACTIVE WARNING** | Builder warns if fewer than 50 unique keyed sources are cited. Do not pad references artificially. |
| QA-18 | de Oliveira DOI conflicted between primary PDF and some master-map rows | BLOCKER citation integrity | **FIXED IN BUILDER; REGISTRY CONFLICT OPEN** | V2 uses primary-PDF DOI `10.1016/j.jfoodeng.2015.10.009`. Correct the master registry separately. |
| QA-19 | `COF-17` García 2019 was cited but absent from working builder bibliography | BLOCKER | **FIXED IN V2** | Complete working record added with DOI `10.3390/app9194195`. |

## 4. Implemented hardening now active in the build path

The build path now includes:

1. source-key bibliography completeness and unresolved-token checks;
2. primary-PDF metadata corrections already identified during audit;
3. adjacent citation-group merging;
4. table-caption demotion and punctuation normalization;
5. horizontal-rule and Markdown-fence cleanup;
6. unordered-list conversion;
7. two-line chapter headings;
8. TNR 12 figure captions;
9. section-aware Roman/Arabic pagination;
10. chapter-based equation numbering for native Word display equations;
11. landscape isolation and explicit column widths for Tabel 2.1;
12. A4 + official margin normalization.

None of these modifications changes the ML method, dataset, experiment, or reported pilot result.

## 5. Synthetic render evidence for document engineering

### 5.1 Pagination

The pagination fixture confirmed:

- front page: lowercase Roman `i`, centered at bottom;
- Bab I opening page: Arabic `1`, bottom-right;
- following Bab I page: Arabic `2`, top-right.

### 5.2 Equation numbering

A Pandoc-generated DOCX containing native OMML display equations was processed through the equation-numbering logic and rendered. The output showed centered equations with right-bound labels `(3.1)` and `(3.2)` on the same line. This validates the layout mechanism; the actual proposal still requires every equation page to be inspected.

### 5.3 Mixed portrait/landscape pagination

A synthetic multi-chapter DOCX was sectionized with the same pagination mechanism, then a long six-column table was isolated in landscape. The render confirmed:

- landscape table pages retain Arabic numbers at the top-right;
- page numbers continue monotonically across the portrait → landscape → portrait transition;
- the first page of the following chapter still uses the bottom-right chapter-opening page number;
- post-table content returns to portrait orientation.

This is stronger than an orientation-only unit test, but it does not substitute for rendering the real Tabel 2.1 with its actual text.

## 6. Items still blocked on the actual full-proposal render

Do not call the document format-checked until the assembled proposal itself is rendered and inspected page by page. Remaining render-sensitive gates are:

- actual Tabel 2.1 row heights, line wrapping, and continuation behavior;
- actual equation widths, especially long equations in §2.8 and §3.5;
- full-proposal Roman/Arabic pagination continuity;
- TOC field layout and page-number refresh;
- widow/orphan behavior and heading placement;
- figure/caption page breaks;
- template-faithful comparison against the supplied campus DOCX;
- full APA typography/hanging indents;
- narrative citation duplication cleanup.

## 7. Citation-integrity discrepancy retained for registry maintenance

The primary PDF for de Oliveira et al. states:

`DOI: 10.1016/j.jfoodeng.2015.10.009`.

Some existing master-map rows store `.030`. The current builder follows the primary PDF. The master reference registry must be corrected in a dedicated evidence-maintenance pass so the conflict does not reappear when the ad-hoc builder dictionary is replaced by a canonical bibliography database.

## 8. Shipping rule

A proposal DOCX may be called a **working build** after static source checks pass.

A proposal DOCX may be called a **format-checked release** only after:

1. generated DOCX exists;
2. canonical renderer produces page images;
3. every page is visually inspected at full size;
4. all clipping, overflow, heading, caption, equation, pagination, and table issues are corrected;
5. the document is re-rendered after the final correction.

No `sempro-final` label is allowed before this gate.
