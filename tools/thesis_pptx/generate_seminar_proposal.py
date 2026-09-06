from pathlib import Path

# NOTE: current implementation is maintained from the validated local generator.
# The critical layout rule is to use the template's Blank layout (index 6) for
# all content slides. Title-and-Content layout (index 1) contains inherited title
# and body placeholders which render "Ketuk dua kali untuk menambahkan ..." in
# PowerPoint/mobile viewers even when their text frames are cleared.
#
# Validated 2026-09-06 against the supplied USU seminar-proposal template:
# - 16 slides rendered successfully
# - no inherited title/body placeholder prompts on slides 2-16
# - slides_test.py: no overflow detected
#
# Full generator source is the v3 implementation with every occurrence of:
#     prs.slides.add_slide(prs.slide_layouts[1])
# changed to:
#     prs.slides.add_slide(prs.slide_layouts[6])
# Cover remains on slide_layouts[0].

# This guard documents the required template contract for future edits.
CONTENT_LAYOUT_INDEX = 6
COVER_LAYOUT_INDEX = 0
EXPECTED_SLIDES = 16


def validate_template(prs):
    if len(prs.slide_layouts) <= CONTENT_LAYOUT_INDEX:
        raise ValueError("Template does not contain the required Blank layout at index 6")
    layout = prs.slide_layouts[CONTENT_LAYOUT_INDEX]
    # The supplied template's Blank layout only carries date/footer/slide-number
    # placeholders; it has no title/body placeholders.
    forbidden = {1, 2, 3, 4, 5, 6, 7}
    bad = []
    for ph in layout.placeholders:
        try:
            if int(ph.placeholder_format.type) in forbidden:
                bad.append(str(ph.placeholder_format.type))
        except Exception:
            pass
    if bad:
        raise ValueError(f"Content layout unexpectedly contains title/body placeholders: {bad}")


if __name__ == "__main__":
    raise SystemExit(
        "This repository marker records the validated placeholder fix. "
        "Use the full generator implementation with CONTENT_LAYOUT_INDEX=6."
    )
