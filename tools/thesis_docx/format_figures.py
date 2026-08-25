#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

MAX_FIGURE_WIDTH = Cm(12.5)


def _paragraph_has_drawing(paragraph) -> bool:
    return bool(
        list(paragraph._p.iter(qn("w:drawing")))
        or list(paragraph._p.iter(qn("w:pict")))
    )


def format_figures(input_path: Path, output_path: Path) -> None:
    doc = Document(input_path)

    figure_paragraphs = 0
    for paragraph in doc.paragraphs:
        if not _paragraph_has_drawing(paragraph):
            continue
        figure_paragraphs += 1
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.left_indent = Cm(0)
        paragraph.paragraph_format.right_indent = Cm(0)
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.paragraph_format.space_before = Pt(6)
        paragraph.paragraph_format.space_after = Pt(6)

    resized = 0
    for shape in doc.inline_shapes:
        if shape.width <= MAX_FIGURE_WIDTH:
            continue
        ratio = shape.height / shape.width
        shape.width = MAX_FIGURE_WIDTH
        shape.height = int(MAX_FIGURE_WIDTH * ratio)
        resized += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(
        f"Formatted {figure_paragraphs} image paragraphs; "
        f"resized {resized} figures wider than 12.5 cm."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Center and constrain Markdown-generated figures in thesis DOCX."
    )
    parser.add_argument("input")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    source = Path(args.input)
    target = Path(args.output) if args.output else source
    format_figures(source, target)


if __name__ == "__main__":
    main()
