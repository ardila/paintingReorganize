"""Regenerate the before/after demo images.

Usage:  python make_demos.py [--fast] [name ...]

Stage 2 costs minutes per painting (Demoiselles ~27 min at 933x960;
cost scales with pixel count), so --fast runs stage 1 only.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

import smooth_palette as sp

PAINTINGS = {
    'demoiselles': ('demoiselles.jpg', 'demoiselles_palette.png'),
    'starry_night': ('starry_night.png', 'starry_night_palette.png'),
    'the_large_bathers': ('the_large_bathers.jpg',
                          'the_large_bathers_palette.png'),
    'kupka': ('input.jpg', 'output.png'),
}


def strip(paths, labels, height=460):
    tiles = []
    for p, lbl in zip(paths, labels):
        im = Image.open(p).convert('RGB') if isinstance(p, str) \
            else Image.fromarray(p)
        im = im.resize((int(im.width * height / im.height), height),
                       Image.LANCZOS)
        d = ImageDraw.Draw(im)
        tw = d.textlength(lbl)
        d.rectangle([4, 4, 14 + tw, 23], fill=(0, 0, 0))
        d.text((9, 8), lbl, fill=(255, 255, 255))
        tiles.append(im)
    gap = 8
    canvas = Image.new('RGB',
                       (sum(t.width for t in tiles) + gap * (len(tiles) - 1),
                        height), (255, 255, 255))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0))
        x += t.width + gap
    return canvas


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    fast = '--fast' in sys.argv
    for name in (args or list(PAINTINGS)):
        src, old = PAINTINGS[name]
        print(f'=== {name}', flush=True)
        rgb = np.asarray(Image.open(src).convert('RGB'))
        out = sp.smooth_palette(rgb, sweeps=0 if fast else 6000)
        Image.fromarray(out).save(f'{name}_smooth.png')
        strip([src, old, out],
              ['original', 'old algorithm', 'new algorithm']).save(
            f'demo_{name}.png')


if __name__ == '__main__':
    main()
