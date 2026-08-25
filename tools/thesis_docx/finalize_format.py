#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

FONT_NAME = "Times New Roman"
TABLE_EDGES = ("top", "left", "bottom", "right", "insideH", "insideV")
CELL_EDGES = ("top", "left", "bottom", "right")


def _replace_borders(parent, tag_name: str, edges, visible: bool) -> None:
    old = parent.find(qn(tag_name))
    if old is not None:
        parent.remove(old)
    borders = OxmlElement(tag_name)
    for edge in edges:
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single" if visible else "nil")
        if visible:
            element.set(qn("w:sz"), "8")
            element.set(qn("w:space"), "0")
            element.set(qn("w:color"), "000000")
        borders.append(element)
    parent.append(borders)


def _set_table_borders(table, visible: bool) -> None:
    tbl_pr = table._tbl.tblPr
    _replace_borders(tbl_pr, "w:tblBorders", TABLE_EDGES, visible)

    # Equation numbering uses a layout table. Remove its inherited table style as
    # well, otherwise Word/LibreOffice can still draw a bottom rule even when a
    # nil table border is present.
    if not visible:
        tbl_style = tbl_pr.find(qn("w:tblStyle"))
        if tbl_style is not None:
            tbl_pr.remove(tbl_style)

    # Explicit cell borders override any inherited/default table formatting.
    for row in table.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            _replace_borders(tc_pr, "w:tcBorders", CELL_EDGES, visible)


def _is_equation_layout_table(table) -> bool:
    return bool(
        list(table._tbl.iter(qn("m:oMath")))
        or list(table._tbl.iter(qn("m:oMathPara")))
    )


def _ensure_rfonts(r_pr) -> None:
    r_fonts = r_pr.find(qn("w:rFonts"))
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.insert(0, r_fonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        r_fonts.set(qn(f"w:{attr}"), FONT_NAME)
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        key = qn(f"w:{attr}")
        if key in r_fonts.attrib:
            del r_fonts.attrib[key]


def _force_font_in_root(root) -> None:
    # Normal Word runs, including field results, table text, headers and footers.
    for run in root.iter(qn("w:r")):
        r_pr = run.find(qn("w:rPr"))
        if r_pr is None:
            r_pr = OxmlElement("w:rPr")
            run.insert(0, r_pr)
        _ensure_rfonts(r_pr)

    # OMML equation runs. Word can still fall back for unsupported mathematical
    # glyphs, but the DOCX explicitly requests Times New Roman at run level.
    for math_run in root.iter(qn("m:r")):
        r_pr = math_run.find(qn("w:rPr"))
        if r_pr is None:
            r_pr = OxmlElement("w:rPr")
            math_run.insert(0, r_pr)
        _ensure_rfonts(r_pr)

    for r_pr in root.iter(qn("w:rPr")):
        _ensure_rfonts(r_pr)


def _force_style_fonts(doc) -> None:
    styles_root = doc.styles.element

    # Document-wide default run properties.
    doc_defaults = styles_root.find(qn("w:docDefaults"))
    if doc_defaults is None:
        doc_defaults = OxmlElement("w:docDefaults")
        styles_root.insert(0, doc_defaults)
    r_pr_default = doc_defaults.find(qn("w:rPrDefault"))
    if r_pr_default is None:
        r_pr_default = OxmlElement("w:rPrDefault")
        doc_defaults.append(r_pr_default)
    r_pr = r_pr_default.find(qn("w:rPr"))
    if r_pr is None:
        r_pr = OxmlElement("w:rPr")
        r_pr_default.append(r_pr)
    _ensure_rfonts(r_pr)

    # Force every Word style to request Times New Roman so generated TOC/list
    # entries and other automatic field content do not revert to theme fonts.
    for style in styles_root.findall(qn("w:style")):
        r_pr = style.find(qn("w:rPr"))
        if r_pr is None:
            r_pr = OxmlElement("w:rPr")
            style.append(r_pr)
        _ensure_rfonts(r_pr)


def _set_math_default_font(doc) -> None:
    settings = doc.settings._element
    math_pr = settings.find(qn("m:mathPr"))
    if math_pr is None:
        math_pr = OxmlElement("m:mathPr")
        settings.append(math_pr)
    math_font = math_pr.find(qn("m:mathFont"))
    if math_font is None:
        math_font = OxmlElement("m:mathFont")
        math_pr.insert(0, math_font)
    math_font.set(qn("m:val"), FONT_NAME)


def finalize(input_path: Path, output_path: Path) -> None:
    doc = Document(input_path)

    equation_tables = 0
    normal_tables = 0
    for table in doc.tables:
        if _is_equation_layout_table(table):
            _set_table_borders(table, visible=False)
            equation_tables += 1
        else:
            _set_table_borders(table, visible=True)
            normal_tables += 1

    _force_style_fonts(doc)
    _set_math_default_font(doc)
    _force_font_in_root(doc.element)

    # Header/footer PAGE fields are stored in separate document parts.
    seen_parts = set()
    for section in doc.sections:
        for container in (
            section.header,
            section.first_page_header,
            section.even_page_header,
            section.footer,
            section.first_page_footer,
            section.even_page_footer,
        ):
            part = container.part
            if id(part) in seen_parts:
                continue
            seen_parts.add(id(part))
            _force_font_in_root(part.element)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(
        f"Finalized {output_path}: {normal_tables} regular tables with full borders, "
        f"{equation_tables} equation-layout tables without borders; font={FONT_NAME}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final DOCX formatting pass for SPs USU proposal builds."
    )
    parser.add_argument("input")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    source = Path(args.input)
    target = Path(args.output) if args.output else source
    finalize(source, target)


if __name__ == "__main__":
    main()
