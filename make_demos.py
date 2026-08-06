"""Generate before/after demo images for the README.

For each painting, produces <name>_smooth.png (the new algorithm's
output) and demo_<name>.png (original | old algorithm | new algorithm,
side by side), plus a score table comparing the two algorithms under
the multi-scale Huber smoothness loss.

Usage:  python make_demos.py [name ...]     (default: all)
"""

import sys

import numpy as np
from PIL import Image, ImageDraw

import smooth_palette as sp

PAINTINGS = {
    "starry_night": ("starry_night.png", "starry_night_palette.png"),
    "demoiselles": ("demoiselles.jpg", "demoiselles_palette.png"),
    "the_large_bathers": ("the_large_bathers.jpg",
                          "the_large_bathers_palette.png"),
    "kupka": ("input.jpg", "output.png"),
}


def label(img, text):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    tw = draw.textlength(text)
    draw.rectangle([4, 4, 12 + tw, 22], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return img


def side_by_side(images, texts, height=480):
    tiles = []
    for im, tx in zip(images, texts):
        w = int(round(im.width * height / im.height))
        tiles.append(label(im.resize((w, height), Image.LANCZOS), tx))
    gap = 8
    total_w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (total_w, height), (255, 255, 255))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0))
        x += t.width + gap
    return canvas


def main():
    sys.setrecursionlimit(100000)
    names = sys.argv[1:] or list(PAINTINGS)
    rows = []
    for name in names:
        input_file, old_file = PAINTINGS[name]
        print(f"=== {name}", flush=True)
        rgb = np.asarray(Image.open(input_file).convert("RGB"))
        result = sp.smooth_palette(rgb)
        new_file = f"{name}_smooth.png"
        Image.fromarray(result).save(new_file)

        old_img = Image.open(old_file).convert("RGB")
        demo = side_by_side(
            [Image.open(input_file), old_img, Image.fromarray(result)],
            ["original", "old algorithm", "new algorithm"],
        )
        demo.save(f"demo_{name}.png")

        loss_old = sp.smoothness_loss(sp.srgb_to_lab(np.asarray(old_img)))
        loss_new = sp.smoothness_loss(sp.srgb_to_lab(result))
        rows.append((name, loss_old, loss_new))

    print(f"\n{'painting':<20} {'old loss':>9} {'new loss':>9} {'ratio':>6}")
    for name, lo, ln in rows:
        print(f"{name:<20} {lo:>9.2f} {ln:>9.2f} {lo / ln:>6.1f}x")


if __name__ == "__main__":
    main()
