#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

FORMAL_FILES = [
    Path("docs/thesis/proposal/BAB_I_PENDAHULUAN.md"),
    Path("docs/thesis/proposal/BAB_II_TINJAUAN_PUSTAKA.md"),
    Path("docs/thesis/proposal/BAB_III_METODOLOGI_PENELITIAN.md"),
]

FORBIDDEN = {
    r"\operatorname": "GitHub Markdown rejects the operatorname macro; use a standard operator or \\mathrm{...}.",
    r"\(": "Use $...$ for inline math so GitHub renders it consistently.",
    r"\[": "Use $$...$$ for display math so GitHub renders it consistently.",
}


def main() -> None:
    errors: list[str] = []
    for path in FORMAL_FILES:
        text = path.read_text(encoding="utf-8")
        for token, message in FORBIDDEN.items():
            if token in text:
                errors.append(f"{path}: forbidden math token {token!r}. {message}")
    if errors:
        raise SystemExit("\n".join(errors))
    print("Proposal math source check passed: GitHub-safe delimiters and no forbidden operatorname macro.")


if __name__ == "__main__":
    main()
