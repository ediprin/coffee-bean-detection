from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .vadcp.library import load_object_library


def _checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    image = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle(
                    (x, y, min(x + cell, size[0]), min(y + cell, size[1])),
                    fill=(170, 170, 170),
                )
    return image


def _fit_cutout(image: Image.Image, size: int = 180) -> Image.Image:
    fitted = ImageOps.contain(image, (size - 20, size - 20), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(
        fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2)
    )
    return canvas


def run_cutout_visual_audit(
    object_library: str | Path,
    output_root: str | Path,
    *,
    samples: int = 18,
    seed: int = 42,
) -> dict:
    _, cutouts, _ = load_object_library(object_library, train_only=True)
    rng = random.Random(seed)
    by_class = {}
    for item in cutouts:
        by_class.setdefault(item.class_name, []).append(item)
    selected = []
    names = sorted(by_class)
    for rows in by_class.values():
        rng.shuffle(rows)
    while len(selected) < min(samples, len(cutouts)):
        progressed = False
        for name in names:
            if by_class[name] and len(selected) < samples:
                selected.append(by_class[name].pop())
                progressed = True
        if not progressed:
            break

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = 3 * 180, 210
    tiles = []
    rows = []
    for cutout in selected:
        with Image.open(cutout.image_path) as source:
            rgba = _fit_cutout(source.convert("RGBA"))
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        backgrounds = [
            Image.new("RGB", (180, 180), "black"),
            Image.new("RGB", (180, 180), "white"),
            _checkerboard((180, 180)),
        ]
        for index, background in enumerate(backgrounds):
            background.paste(rgba, (0, 0), rgba)
            tile.paste(background, (index * 180, 30))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 8), f"{cutout.class_name} | {cutout.asset_id[:8]}", fill="black")
        tiles.append(tile)
        rows.append(
            {
                "asset_id": cutout.asset_id,
                "class": cutout.class_name,
                "path": str(cutout.image_path),
            }
        )
    columns = 2
    sheet = Image.new(
        "RGB",
        (columns * tile_width, ((len(tiles) + columns - 1) // columns) * tile_height),
        "white",
    )
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
    contact_sheet = output_root / "cutout_edge_contact_sheet.jpg"
    sheet.save(contact_sheet, quality=94)
    payload = {
        "object_library": str(Path(object_library).expanduser().resolve()),
        "samples": len(rows),
        "contact_sheet": str(contact_sheet),
        "selected": rows,
    }
    (output_root / "cutout_visual_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit halo cutout pada background hitam, putih, dan checkerboard."
    )
    parser.add_argument("--object-library", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_cutout_visual_audit(
        args.object_library, args.output_root, samples=args.samples, seed=args.seed
    )
    print("=== CUTOUT EDGE AUDIT ===")
    print("Samples      :", result["samples"])
    print("Contact sheet:", result["contact_sheet"])


if __name__ == "__main__":
    main()
