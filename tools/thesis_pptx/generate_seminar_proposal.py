#!/usr/bin/env python3
"""Generate seminar-proposal PPTX using an existing PPTX as the visual template.

Usage:
    python tools/thesis_pptx/generate_seminar_proposal.py \
      --template "ppt seminar proposal FINAL（5）.pptx" \
      --source docs/thesis/proposal/PPT_SEMINAR_PROPOSAL.md \
      --output build/seminar_proposal_kopi.pptx

Design contract
---------------
The supplied PPTX is the visual source of truth. The generator keeps the template's
slide size, master/theme, backgrounds, decorative shapes, colors, and general
composition. It reuses selected template slides as layout archetypes and replaces
only presentation text. The scientific source of truth remains the formal proposal.

This script intentionally does not create a new Git branch.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt


# 16 output slides -> 1-based slide number in the provided visual template.
# The mapping follows the closest visual archetype in the reference deck:
# 1 cover, 2 agenda, 3 intro, 4 problem, 5 scope, 6 related work,
# 7 concept, 9 dataset, 10 experiment setup, 11/12 workflow diagrams,
# 15 objective, 17 evaluation, 13 closing.
TEMPLATE_MAP = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 11, 12, 15, 17, 13]
EXPECTED_SLIDES = len(TEMPLATE_MAP)


def parse_source(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^---\s*$", text)
    slides = []
    for block in blocks:
        m = re.search(r"(?m)^## SLIDE\s+(\d+)\s+—\s+(.+?)\s*$", block)
        if not m:
            continue
        number = int(m.group(1))
        label = m.group(2).strip()
        body = block[m.end():].strip()
        title_match = re.search(r"(?m)^###\s+(.+?)\s*$", body)
        title = title_match.group(1).strip() if title_match else ""
        if title_match:
            body = body[title_match.end():].strip()
        slides.append({"number": number, "label": label, "title": title, "body": body})
    slides.sort(key=lambda x: x["number"])
    return slides


def clone_slide(prs: Presentation, source_slide):
    """Clone a slide inside the same presentation, preserving its visual objects."""
    blank_layout = prs.slide_layouts[6]
    new_slide = prs.slides.add_slide(blank_layout)

    for shape in list(new_slide.shapes):
        sp = shape._element
        sp.getparent().remove(sp)

    for shape in source_slide.shapes:
        new_el = copy.deepcopy(shape.element)
        new_slide.shapes._spTree.insert_element_before(new_el, "p:extLst")

    for rel in source_slide.part.rels.values():
        if "notesSlide" in rel.reltype:
            continue
        try:
            if rel.is_external:
                new_slide.part.rels.add_relationship(
                    rel.reltype, rel.target_ref, rel.rId, is_external=True
                )
            else:
                new_slide.part.rels.add_relationship(rel.reltype, rel._target, rel.rId)
        except Exception:
            pass
    return new_slide


def delete_slide(prs: Presentation, index: int) -> None:
    slide_id = prs.slides._sldIdLst[index]
    r_id = slide_id.rId
    prs.part.drop_rel(r_id)
    del prs.slides._sldIdLst[index]


def text_shapes(slide):
    shapes = []
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False):
            shapes.append(shape)
    return sorted(shapes, key=lambda s: (s.top, s.left))


def clean_md(s: str) -> str:
    s = re.sub(r"```text\s*", "", s)
    s = s.replace("```", "")
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"\*(.*?)\*", r"\1", s)
    s = s.replace("\\*", "*")
    return s.strip()


def body_chunks(body: str) -> list[str]:
    body = clean_md(body)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", body) if c.strip()]
    return chunks


def set_text(shape, text: str, *, size: float | None = None, bold: bool | None = None):
    tf = shape.text_frame
    tf.clear()
    lines = text.splitlines() or [""]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line.strip()
        if size is not None:
            for run in p.runs:
                run.font.size = Pt(size)
        if bold is not None:
            for run in p.runs:
                run.font.bold = bold


def fill_generic(slide, slide_data: dict):
    shapes = text_shapes(slide)
    if not shapes:
        return

    title = slide_data["title"]
    chunks = body_chunks(slide_data["body"])
    title_shape = min(shapes, key=lambda s: (s.top, -s.width))
    if title:
        set_text(title_shape, title)

    remaining = [s for s in shapes if s is not title_shape]
    remaining = sorted(remaining, key=lambda s: (s.top, s.left))

    for shape in remaining:
        shape.text_frame.clear()

    if not remaining:
        if chunks:
            set_text(title_shape, title + "\n\n" + "\n".join(chunks))
        return

    for i, chunk in enumerate(chunks):
        target = remaining[min(i, len(remaining) - 1)]
        if i < len(remaining):
            set_text(target, chunk)
        else:
            old = target.text.strip()
            set_text(target, (old + "\n\n" + chunk).strip())


def fill_cover(slide, data: dict):
    shapes = text_shapes(slide)
    for s in shapes:
        s.text_frame.clear()
    chunks = body_chunks(data["body"])
    if shapes:
        ordered = sorted(shapes, key=lambda s: (s.top, s.left))
        set_text(ordered[0], data["title"])
        for shape, chunk in zip(ordered[1:], chunks):
            set_text(shape, chunk)


def fill_three_points(slide, data: dict):
    shapes = text_shapes(slide)
    if not shapes:
        return
    title_shape = min(shapes, key=lambda s: (s.top, -s.width))
    set_text(title_shape, data["title"])
    rest = [s for s in shapes if s is not title_shape]
    for s in rest:
        s.text_frame.clear()
    chunks = body_chunks(data["body"])
    for shape, chunk in zip(rest, chunks):
        set_text(shape, chunk)


def fill_slide(slide, data: dict):
    n = data["number"]
    if n == 1:
        fill_cover(slide, data)
    elif n == 3:
        fill_three_points(slide, data)
    else:
        fill_generic(slide, data)


def build(template: Path, source: Path, output: Path):
    slides_data = parse_source(source)
    if len(slides_data) != EXPECTED_SLIDES:
        raise ValueError(
            f"Expected {EXPECTED_SLIDES} slide sections based on TEMPLATE_MAP, "
            f"found {len(slides_data)}. Update TEMPLATE_MAP if the Markdown slide count changes."
        )

    prs = Presentation(str(template))
    original = list(prs.slides)
    if len(original) < max(TEMPLATE_MAP):
        raise ValueError("Template does not contain all referenced archetype slides")

    generated = []
    for source_idx in TEMPLATE_MAP:
        generated.append(clone_slide(prs, original[source_idx - 1]))

    for _ in range(len(original)):
        delete_slide(prs, 0)

    for slide, data in zip(prs.slides, slides_data):
        fill_slide(slide, data)

    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    print(f"Generated: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Reference PPTX whose visual format is reused",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("docs/thesis/proposal/PPT_SEMINAR_PROPOSAL.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/seminar_proposal_kopi.pptx"),
    )
    args = parser.parse_args()
    build(args.template, args.source, args.output)


if __name__ == "__main__":
    main()
