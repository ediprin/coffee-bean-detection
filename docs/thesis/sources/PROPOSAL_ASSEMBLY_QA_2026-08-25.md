# Proposal Formal-Assembly QA — 2026-08-25

Status: **first static QA before rendered-document inspection**.

Purpose: compare the current Git-driven DOCX builder against the uploaded SPs thesis/dissertation writing guide and the supplied Word scaffold. This audit is intentionally stricter than the working-build pipeline. It does **not** claim that the generated DOCX has passed visual QA until the actual document is rendered and every page is inspected.

## 1. Current verdict

\[
\boxed{\text{CONTENT ASSEMBLY READY; FORMAL LAYOUT NOT YET RELEASE-READY}}
\]

The scientific chapter assembly is stable, but several document-engineering items remain between the current working DOCX and a campus-format release.

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
- figure caption is below the figure, centered, with no period after the figure number and no terminal period;
- tables that are too wide for portrait should use landscape orientation;
- equations use chapter-based numbering positioned toward the right margin.

## 3. Static QA findings

| ID | Finding | Severity | Current status | Required action |
|---|---|---|---|---|
| QA-01 | Builder creates DOCX from Pandoc rather than actually cloning the supplied DOCX template | BLOCKER for formal release | OPEN | Add optional reference/template path and use the supplied template during local/formal release; working CI may retain fallback mode until the binary template is available in CI. |
| QA-02 | All pages currently receive Arabic numbering at bottom-right | BLOCKER | OPEN | Create separate front-matter/body sections; Roman center-bottom for front matter, Arabic top-right for normal body pages, Arabic bottom-right for chapter-opening pages. |
| QA-03 | Chapter headings are currently generated as one line (`BAB I PENDAHULUAN`) | MAJOR | OPEN | Format as two lines matching campus scaffold: `BAB I` then `PENDAHULUAN`, centered. |
| QA-04 | Figure captions currently contain a period after the number (`Gambar 3.1.`) | MAJOR | OPEN | Change to `Gambar 3.1  Kerangka Penelitian`-style caption with no period after the number and no final period. |
| QA-05 | Figure captions are currently forced to TNR 10 | MAJOR | OPEN | Use TNR 12 unless a later guide-specific exception is verified. |
| QA-06 | Related-work Table 2.1 has six columns and long prose; portrait overflow risk is high | BLOCKER pending render | OPEN | Render; if width exceeds portrait, place Table 2.1 in landscape section as instructed by the guide. |
| QA-07 | All tables are currently forced to 9 pt | MAJOR | OPEN | Restore near-template sizing for normal tables; compact only wide tables as a documented exception. |
| QA-08 | Markdown table titles are heading levels and therefore may enter the TOC | MAJOR | OPEN | Convert `Tabel x.y` lines to caption paragraphs before Pandoc/Word styling. |
| QA-09 | Markdown horizontal rules (`---`) can become visible separator lines | MAJOR | OPEN | Strip them during formal assembly. |
| QA-10 | Source files contain unordered Markdown bullets | MAJOR | OPEN | Convert formal lists to numbered/lettered lists during assembly or normalize the source modules. |
| QA-11 | Fenced text/code blocks can render as monospaced code | MAJOR | OPEN | Remove fence delimiters and replace explanatory flows by figures/tables/plain prose. |
| QA-12 | Display equations have no final chapter-based equation numbers | BLOCKER for formal release | OPEN | Add equation-numbering pass after content/OMML conversion; right-align numbers according to guide. |
| QA-13 | Adjacent internal citations can become adjacent parenthetical groups | MAJOR | OPEN | Merge adjacent source-key groups before author-year conversion. |
| QA-14 | Narrative prose can duplicate author names after automatic citation conversion | MAJOR | OPEN | Introduce narrative/parenthetical citation handling or normalize sentence prose before final release. |
| QA-15 | Missing bibliography records are silently skipped | BLOCKER | OPEN | Build must fail when any used citation key has no canonical bibliography record. |
| QA-16 | Working `BIB` entries are not yet guaranteed complete APA records | BLOCKER for final bibliography | OPEN | Replace ad-hoc strings with canonical bibliography registry/BibTeX/CSL after full reference audit. |
| QA-17 | Current bibliography count may remain below the guide’s minimum-reference target | WARNING | OPEN | Emit build warning; do not pad references artificially. Close only when genuinely cited sources satisfy the final program requirement. |
| QA-18 | Current builder has a de Oliveira DOI conflict with the primary PDF | BLOCKER citation integrity | OPEN | Primary PDF gives DOI `10.1016/j.jfoodeng.2015.10.009`; correct builder and flag master-map discrepancy for later registry correction. |
| QA-19 | `COF-17` García 2019 is cited but absent from the current builder bibliography | BLOCKER | OPEN | Add the primary-source record: García, M., Candelo-Becerra, J. E., & Hoyos, F. E. (2019), *Applied Sciences*, 9(19), 4195, DOI `10.3390/app9194195`. |

## 4. Low-risk fixes approved for immediate implementation

The following can be fixed before the first rendered-page pass without changing scientific content:

1. figure-caption punctuation/case/font;
2. horizontal-rule removal;
3. table-caption demotion from heading to ordinary caption paragraph;
4. unordered-list conversion to ordered Markdown lists;
5. code-fence delimiter removal;
6. adjacent citation-group merging;
7. hard failure on missing bibliography keys;
8. add verified `COF-17` bibliography record;
9. correct de Oliveira DOI against the primary PDF;
10. two-line chapter headings;
11. less aggressive table font defaults;
12. build-time warnings for unresolved internal citation keys and low reference count.

These changes do not require a new ML experiment.

## 5. Items intentionally deferred to rendered-layout iteration

Do not claim these are solved before actual DOCX rendering:

- landscape handling for Table 2.1;
- exact Roman/Arabic pagination behavior across Word/LibreOffice;
- chapter-opening first-page footer versus regular-page header;
- exact equation-number alignment;
- widow/orphan behavior;
- figure/table page breaks;
- full APA typography/hanging indents;
- final TOC/page-field refresh.

## 6. Citation-integrity discrepancy found during QA

The primary PDF for de Oliveira et al. states:

`DOI: 10.1016/j.jfoodeng.2015.10.009`.

Some existing master-map rows currently store `.030`. For formal bibliography generation, the primary PDF must prevail. The master registry should be corrected in a separate evidence-maintenance pass rather than silently propagating the conflicting metadata.

## 7. Shipping rule

A proposal DOCX may be called a **working build** after static source checks pass.

A proposal DOCX may be called a **format-checked release** only after:

1. generated DOCX exists;
2. canonical renderer produces page PNGs;
3. every page is visually inspected at full size;
4. all clipping, overflow, heading, caption, equation, pagination, and table issues are corrected;
5. the document is re-rendered after the final correction.

No `sempro-final` label is allowed before this gate.