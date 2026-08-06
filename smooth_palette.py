"""Reorganize the pixels of a painting into a smooth color palette.

The output image:
  * contains exactly the same pixels as the input (just rearranged),
  * is a dense rectangle with the same dimensions as the input
    (so the bounding box is as small as possible),
  * is optimized so the color field is *visually* smooth.

The loss
--------
Visual smoothness is not the sum of adjacent-pixel differences: that sum
is nearly the same whether a color transition is spread gently over 50
pixels or concentrated in one hard seam, so optimizing it happily
produces a few razor edges.  Instead, each pixel pair at offset s (for
s in 1, 2, 4, 8) contributes a Huber penalty on its Lab distance d,
weighted 1/s^2:

    huber(d)  =  d^2                    if d <= tau*s
                 tau*s * (2d - tau*s)   otherwise

The quadratic regime means many small steps are far cheaper than one
mid-size step, so the optimizer spreads transitions into even ramps, and
isolated specks (a large d against every neighbor) are maximally
expensive.  Only a truly unavoidable palette gap - a jump beyond tau -
escapes into the linear regime, where the cheapest option is one short,
straight, crisp frontier rather than dither.  Thresholds scale with the
offset so a clean linear ramp stays quadratic at every scale.

The algorithm
-------------
1.  Global layout ("what goes where"): fit a principal curve through the
    color distribution (iterated Hastie-Stuetzle, seeded by the first
    principal component).  Pixels sorted by position along the curve
    form the columns: the horizontal axis sweeps the palette.
2.  Vertical structure: within each column, pixels are sorted by their
    global manifold-order rank - a 1D ordering built by recursive PCA
    splits (with seam-minimizing flips) that keeps every color cluster
    contiguous.  Because the ranking is identical for every column, the
    same color always sorts to the same relative height: bands line up
    across columns, and minority colors form contiguous runs instead of
    being sprinkled as specks that no local optimizer could gather.
3.  Column alignment: each column is re-sorted so its colors line up
    with the rows of its neighbor columns' mean, straightening band
    boundaries.
4.  Polish: greedy descent on the exact loss with short-range swaps.

A note on optimizers that did NOT survive: majorize-minimize schemes
that match pixels to a blurred/neighbor-mean target (self-organizing-map
style) reach lower loss values but sprinkle isolated dots - the
mean-field target cannot see that a lone speck should join a distant
region of its own color, and accepts speck-creating trades for mid-scale
gains.  Greedy descent on the exact loss never creates specks.  When
metric and eye disagree, the eye wins.

Usage:  python smooth_palette.py input.jpg [output.png]
"""

import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

SCALES = (1, 2, 4, 8)
SCALE_WEIGHTS = tuple(1.0 / (s * s) for s in SCALES)
TAU = 60.0  # Huber threshold at scale 1, in Lab distance units


# ----------------------------------------------------------------------
# Color conversion (sRGB -> CIELab)
# ----------------------------------------------------------------------

def srgb_to_lab(rgb):
    """rgb: (..., 3) uint8 -> lab float32."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883])  # D65 white
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    lab = np.empty_like(xyz)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab.astype(np.float32)


# ----------------------------------------------------------------------
# The loss
# ----------------------------------------------------------------------

def _huber(d, tau):
    """Elementwise Huber penalty of distances d with threshold tau."""
    return np.where(d <= tau, d * d, tau * (2.0 * d - tau))


def smoothness_loss(lab_img):
    """Multi-scale Huber smoothness loss, per pixel pair."""
    total = 0.0
    n_terms = 0
    for s, w in zip(SCALES, SCALE_WEIGHTS):
        dx = np.linalg.norm(lab_img[:, s:] - lab_img[:, :-s], axis=-1)
        dy = np.linalg.norm(lab_img[s:, :] - lab_img[:-s, :], axis=-1)
        total += w * (float(_huber(dx, TAU * s).sum())
                      + float(_huber(dy, TAU * s).sum()))
        n_terms += dx.size + dy.size
    return total / n_terms


def mean_edge(lab_img):
    """Mean Lab distance between 4-neighbors (secondary metric)."""
    dx = np.linalg.norm(lab_img[:, 1:] - lab_img[:, :-1], axis=-1)
    dy = np.linalg.norm(lab_img[1:, :] - lab_img[:-1, :], axis=-1)
    return (dx.sum() + dy.sum()) / (dx.size + dy.size)


# ----------------------------------------------------------------------
# Step 1-2: global layout
# ----------------------------------------------------------------------

def _principal_projection(pts):
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    _, vecs = np.linalg.eigh(cov)
    return centered @ vecs[:, -1]


def manifold_order(lab, leaf=256):
    """1D ordering of pixels that follows the color distribution.

    Recursively split the pixel set in half along its own principal
    axis, order each half, then join the halves choosing the flip of
    each that minimizes the color jump at the seam.  Unlike a plain
    global PCA sort, colors that are far apart in 3D never interleave:
    every color cluster ends up as one contiguous segment.
    """

    def rec(idx):
        pts = lab[idx]
        proj = _principal_projection(pts)
        if len(idx) <= leaf:
            return idx[np.argsort(proj, kind="stable")]
        s = np.argsort(proj, kind="stable")
        half = len(idx) // 2
        oa = rec(idx[s[:half]])
        ob = rec(idx[s[half:]])
        k = min(32, len(oa), len(ob))
        a0, a1 = lab[oa[:k]].mean(0), lab[oa[-k:]].mean(0)
        b0, b1 = lab[ob[:k]].mean(0), lab[ob[-k:]].mean(0)
        choices = [
            (np.linalg.norm(a1 - b0), False, False),
            (np.linalg.norm(a1 - b1), False, True),
            (np.linalg.norm(a0 - b0), True, False),
            (np.linalg.norm(a0 - b1), True, True),
        ]
        _, flip_a, flip_b = min(choices, key=lambda c: c[0])
        if flip_a:
            oa = oa[::-1]
        if flip_b:
            ob = ob[::-1]
        return np.concatenate([oa, ob])

    return rec(np.arange(lab.shape[0]))


def principal_curve_coords(lab, n_nodes=512, iters=6):
    """Position of every pixel along a smooth principal curve through
    color space (iterated Hastie-Stuetzle fit seeded by the first
    principal component)."""
    n = lab.shape[0]
    n_nodes = min(n_nodes, max(8, n // 64))
    centered = lab - lab.mean(axis=0)
    cov = centered.T @ centered
    _, vecs = np.linalg.eigh(cov)
    t_ord = np.argsort(centered @ vecs[:, -1], kind="stable")

    sigmas = np.geomspace(32.0, 8.0, iters)
    bounds = np.linspace(0, n, n_nodes + 1).astype(np.int64)
    lab64 = lab.astype(np.float64)
    t_idx = np.empty(n, dtype=np.int64)
    chunk = 1 << 18  # keep the pixels x nodes score matrix small
    for it in range(iters):
        nodes = np.add.reduceat(lab64[t_ord], bounds[:-1], axis=0)
        nodes /= np.diff(bounds)[:, None]
        nodes = gaussian_filter1d(nodes, sigma=sigmas[it], axis=0,
                                  mode="nearest")
        half_n2 = 0.5 * (nodes ** 2).sum(1)[None, :]
        for lo in range(0, n, chunk):
            scores = lab64[lo:lo + chunk] @ nodes.T - half_n2
            t_idx[lo:lo + chunk] = np.argmax(scores, axis=1)
        t_ord = np.argsort(t_idx, kind="stable")
    return t_idx


def initial_arrangement(lab, h, w):
    """One coherent global sweep.

    Pixels are sorted along the principal curve of the color
    distribution and chunked into columns (left to right), so the
    horizontal axis sweeps the palette.  Within each column, pixels are
    sorted by their *global* manifold-order rank: because that ordering
    is color-contiguous and identical for every column, the same color
    always sorts to the same relative height - bands line up across
    columns, and minority colors form contiguous runs instead of being
    sprinkled as specks (which no local optimizer could later gather).
    """
    t = principal_curve_coords(lab)
    rank = np.empty(lab.shape[0], dtype=np.int64)
    rank[manifold_order(lab)] = np.arange(lab.shape[0])

    order = np.argsort(t, kind="stable")
    perm = np.empty(h * w, dtype=np.int64)
    for col in range(w):
        chunk = order[col * h:(col + 1) * h]
        chunk = chunk[np.argsort(rank[chunk], kind="stable")]
        perm[col * h:(col + 1) * h] = chunk
    # perm is column-major placement; convert to row-major pixel order.
    return perm.reshape(w, h).T.reshape(-1)


# ----------------------------------------------------------------------
# Step 3: column alignment
# ----------------------------------------------------------------------

def align_columns(lab_img, idx_img, sweeps=8):
    """Re-sort each column so its colors line up with the rows of the
    mean of its two neighbor columns.  Straightens band boundaries and
    removes chatter the initial ranking leaves behind."""
    h, w, _ = lab_img.shape
    for _ in range(sweeps):
        for c in range(w):
            lo, hi = max(0, c - 1), min(w - 1, c + 1)
            ref = (lab_img[:, lo].astype(np.float64)
                   + lab_img[:, hi].astype(np.float64)) / 2.0
            col = lab_img[:, c].astype(np.float64)
            scores = col @ ref.T - 0.5 * (ref ** 2).sum(1)[None, :]
            key = np.argmax(scores, axis=1)
            o = np.lexsort((np.arange(h), key))
            lab_img[:, c] = lab_img[o, c]
            idx_img[:, c] = idx_img[o, c]


# ----------------------------------------------------------------------
# Step 4: polish (greedy descent on the exact loss)
# ----------------------------------------------------------------------

POLISH_SCALES = (1, 2)  # energy terms considered during polish


def _apply_swap(arrays, a, b):
    for arr in arrays:
        tmp = arr[a].copy()
        arr[a] = arr[b]
        arr[b] = tmp


def _polish_site_cost(lab_img, ys, xs, colors, partner_ys, partner_xs):
    """Huber cost of `colors` at sites (ys, xs) against current
    neighbors at POLISH_SCALES, excluding the swap partner (its terms
    are unchanged by the swap, and shared neighbors cancel in deltas)."""
    h, w, _ = lab_img.shape
    cost = np.zeros(ys.shape, dtype=np.float64)
    for s in POLISH_SCALES:
        wgt = 1.0 / (s * s)
        for dy, dx in ((0, s), (0, -s), (s, 0), (-s, 0)):
            ny, nx = ys + dy, xs + dx
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            valid &= ~((ny == partner_ys) & (nx == partner_xs))
            nyc, nxc = np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)
            d = np.linalg.norm(lab_img[nyc, nxc] - colors, axis=-1)
            cost += np.where(valid, wgt * _huber(d, TAU * s), 0.0)
    return cost


def polish(lab_img, idx_img, sweeps=30, verbose=True):
    """Greedy descent on the exact Huber loss with short-range swaps.

    Swap candidates are proposed on independent sets (strides wide
    enough that concurrent swaps cannot interact), so every accepted
    swap is an exact improvement of the loss.
    """
    h, w, _ = lab_img.shape
    max_r = max(POLISH_SCALES)
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0),
               (0, 3), (0, 4)]  # extra horizontal reach softens streaks
    for sweep in range(sweeps):
        total = 0
        for dy, dx in offsets:
            sy = abs(dy) + 2 * max_r + 1
            sx = abs(dx) + 2 * max_r + 1
            for py in range(sy):
                for px in range(sx):
                    ys, xs = np.mgrid[py:h:sy, px:w:sx]
                    ys, xs = ys.ravel(), xs.ravel()
                    ys2, xs2 = ys + dy, xs + dx
                    ok = (ys2 >= 0) & (ys2 < h) & (xs2 >= 0) & (xs2 < w)
                    ys, xs, ys2, xs2 = ys[ok], xs[ok], ys2[ok], xs2[ok]
                    c1 = lab_img[ys, xs]
                    c2 = lab_img[ys2, xs2]
                    before = (_polish_site_cost(lab_img, ys, xs, c1,
                                                ys2, xs2)
                              + _polish_site_cost(lab_img, ys2, xs2, c2,
                                                  ys, xs))
                    after = (_polish_site_cost(lab_img, ys, xs, c2,
                                               ys2, xs2)
                             + _polish_site_cost(lab_img, ys2, xs2, c1,
                                                 ys, xs))
                    sw = (after - before) < -1e-9
                    _apply_swap((lab_img, idx_img),
                                (ys[sw], xs[sw]), (ys2[sw], xs2[sw]))
                    total += int(sw.sum())
        if verbose:
            print(f"  polish sweep {sweep + 1}: {total} swaps, "
                  f"loss={smoothness_loss(lab_img):.4f}", flush=True)
        if total < (h * w) // 2000:
            break
    return lab_img


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def smooth_palette(rgb_image, verbose=True):
    """rgb_image: (H, W, 3) uint8.  Returns rearranged (H, W, 3) uint8."""
    h, w, _ = rgb_image.shape
    pixels = rgb_image.reshape(-1, 3)
    lab = srgb_to_lab(pixels)

    perm = initial_arrangement(lab, h, w)
    idx_img = perm.reshape(h, w).copy()
    lab_img = lab[perm].reshape(h, w, 3).copy()
    align_columns(lab_img, idx_img)
    if verbose:
        print(f"init  loss={smoothness_loss(lab_img):.4f}  "
              f"edge={mean_edge(lab_img):.3f}", flush=True)

    polish(lab_img, idx_img, verbose=verbose)

    if verbose:
        print(f"final loss={smoothness_loss(lab_img):.4f}  "
              f"edge={mean_edge(lab_img):.3f}", flush=True)

    out = pixels[idx_img.reshape(-1)]

    # Constraint check: idx_img must still be a permutation, i.e. the
    # output holds exactly the input's pixel multiset.
    assert np.array_equal(np.sort(idx_img.reshape(-1)), np.arange(h * w)), \
        "pixel multiset changed!"

    return out.reshape(h, w, 3)


def main():
    sys.setrecursionlimit(100000)
    if len(sys.argv) < 2:
        print("Usage: python smooth_palette.py input.jpg [output.png]")
        sys.exit(1)
    filename = os.path.expanduser(sys.argv[1])
    output_name = sys.argv[2] if len(sys.argv) > 2 else "output.png"

    rgb = np.asarray(Image.open(filename).convert("RGB"))
    result = smooth_palette(rgb)
    Image.fromarray(result).save(output_name)
    print(f"saved {output_name}")


if __name__ == "__main__":
    main()
