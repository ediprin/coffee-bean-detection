#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"


def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text).strip().casefold()


def pdf_pages(pdf_path: Path) -> list[str]:
    pdf = fitz.open(pdf_path)
    return [normalize(page.get_text()) for page in pdf]


def find_page(pages: list[str], text: str, start_page: int = 1) -> int:
    query = normalize(text)
    for page_number, page_text in enumerate(pages[start_page - 1 :], start=start_page):
        if query in page_text:
            return page_number
    short = query[:45]
    for page_number, page_text in enumerate(pages[start_page - 1 :], start=start_page):
        if short and short in page_text:
            return page_number
    raise RuntimeError(f"Could not locate caption in rendered PDF: {text!r}")


def find_bab1_page(pages: list[str]) -> int:
    for page_number, page_text in enumerate(pages, start=1):
        if "bab i" in page_text and "pendahuluan" in page_text and "1.1 latar belakang" in page_text:
            return page_number
    raise RuntimeError("Could not locate BAB I first page")


def ensure_entry_style(doc):
    name = "Static List Entry"
    if name in doc.styles:
        style = doc.styles[name]
    else:
        style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(12)
    style.font.bold = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{attr}"), FONT)
    pf = style.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.line_spacing = 1.0
    pf.first_line_indent = Cm(0)
    pf.left_indent = Cm(0)
    pf.right_indent = Cm(0)
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    try:
        pf.tab_stops.clear_all()
    except Exception:
        pass
    pf.tab_stops.add_tab_stop(Cm(14), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    return style


def remove_caption_toc_field_after(title_paragraph, expected_style: str) -> None:
    node = title_paragraph._p.getnext()
    if node is None or node.tag != qn("w:p"):
        return
    instructions = [item.text or "" for item in node.iter(qn("w:instrText"))]
    if any("TOC" in text and expected_style in text for text in instructions):
        node.getparent().remove(node)


def insert_entries_after(title_paragraph, entries, style_id: str) -> None:
    anchor = title_paragraph._p
    for text, page in entries:
        paragraph = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        pstyle = OxmlElement("w:pStyle")
        pstyle.set(qn("w:val"), style_id)
        ppr.append(pstyle)
        paragraph.append(ppr)

        text_run = OxmlElement("w:r")
        text_node = OxmlElement("w:t")
        text_node.text = text
        text_run.append(text_node)
        paragraph.append(text_run)

        tab_run = OxmlElement("w:r")
        tab_run.append(OxmlElement("w:tab"))
        paragraph.append(tab_run)

        page_run = OxmlElement("w:r")
        page_node = OxmlElement("w:t")
        page_node.text = page
        page_run.append(page_node)
        paragraph.append(page_run)

        anchor.addnext(paragraph)
        anchor = paragraph


def build(input_path: Path, pdf_path: Path, output_path: Path) -> None:
    doc = Document(input_path)
    style = ensure_entry_style(doc)
    pages = pdf_pages(pdf_path)
    bab1_page = find_bab1_page(pages)

    table_captions = [p.text.strip() for p in doc.paragraphs if p.style.name == "Caption Table" and p.text.strip()]
    figure_captions = [p.text.strip() for p in doc.paragraphs if p.style.name == "Caption Figure" and p.text.strip()]

    table_entries = [(caption, str(find_page(pages, caption, bab1_page) - bab1_page + 1)) for caption in table_captions]
    figure_entries = [(caption, str(find_page(pages, caption, bab1_page) - bab1_page + 1)) for caption in figure_captions]

    table_title = next(p for p in doc.paragraphs if p.text.strip().upper() == "DAFTAR TABEL")
    figure_title = next(p for p in doc.paragraphs if p.text.strip().upper() == "DAFTAR GAMBAR")

    remove_caption_toc_field_after(table_title, "Caption Table")
    remove_caption_toc_field_after(figure_title, "Caption Figure")
    insert_entries_after(table_title, table_entries, style.style_id)
    insert_entries_after(figure_title, figure_entries, style.style_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print("DAFTAR TABEL:")
    for text, page in table_entries:
        print(f"- {text} -> {page}")
    print("DAFTAR GAMBAR:")
    for text, page in figure_entries:
        print(f"- {text} -> {page}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize visible thesis lists of tables and figures.")
    parser.add_argument("input")
    parser.add_argument("pdf")
    parser.add_argument("output")
    args = parser.parse_args()
    build(Path(args.input), Path(args.pdf), Path(args.output))


if __name__ == "__main__":
    main()
