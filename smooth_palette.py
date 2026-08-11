"""Reorganize the pixels of a painting into a smooth colour palette.

The output holds exactly the same pixels as the input, in the same dense
rectangle, rearranged so the colour field reads as smooth.

Two stages.

1. SLIDING-WINDOW INITIALISATION  (seconds)
   The original algorithm sorted each output column using only that
   column's own pixels.  The height of a colour boundary is then an
   order statistic of ~h samples, so it carries sampling noise of order
   sqrt(h) rows - and neighbouring columns, estimating it independently,
   disagree.  That disagreement is what showed up as jagged "spikes" and
   vertical chatter.

   Here each column is cut from a WINDOW holding a few percent of the
   picture: the sort direction and the composition are estimated from
   that whole window, so the noise falls by sqrt(number of pooled
   columns), and consecutive windows overlap by ~97%, so neighbouring
   columns are cut from almost the same sample and can barely disagree.
   Dealing every k-th pixel of the sorted window keeps each column
   spanning the full range (taking a contiguous block instead collapses
   the picture to colour = f(x)).

2. PHYSICAL REFINEMENT  (minutes)
   Pixels are then annealed under an energy with two terms:

       E = sum_{s!=t} w(p_s - p_t) ||C_s - C_t||^2  -  lam * sum_s C_s . F(p_s)

   * the one-body field F(p) = a*x*u1 + b*y*u2 (u1,u2 = principal colour
     axes) fixes the COMPOSITION.  Minimising it alone is exactly the
     optimal-transport map from the PC1/PC2 projection onto the grid, so
     the left-to-right palette sweep becomes a genuine equilibrium
     instead of a state that decays as the run continues.  A purely
     pairwise energy cannot do this: being invariant to rotating the
     picture, it can only prefer concentric blobs.
   * the two-body kernel w fixes the TEXTURE.  It is a difference of
     Gaussians - repulsive below ~2.5px, attractive from ~3-20px.  Plain
     attraction at r=1 gathers the leftover colour dimension into 2-4px
     clumps that read as grain; making the kernel hollow leaves that
     residual as 1px dither, which the eye integrates away.

   Pixels EXCHANGE rather than move, so the arrangement is a permutation
   at every instant and every site stays filled by construction.  Since
   the kernel is short-range and separable, the field w * C costs two
   Gaussian blurs rather than a padded FFT.

Note the annealing gets WORSE before it gets better: on Demoiselles the
local roughness runs 6.3 -> 16.6 at peak heat -> 4.6 at the end, and the
composition dips before exceeding its starting value.  Runs shorter than
~2500 sweeps only show the damage.

Colour space is RGB throughout.  Working in CIELab was tried and made
every painting visibly worse: its cube root expands differences among
dark colours, so the sort spends resolution separating shadows the eye
cannot distinguish, which surfaces as streaking.

Usage:  python smooth_palette.py input.jpg [output.png] [--fast]
"""

import argparse
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------
# Stage 1: sliding-window initialisation
# ----------------------------------------------------------------------

def _pc1(x):
    c = x - x.mean(0)
    _, v = np.linalg.eigh(c.T @ c)
    return v[:, -1]


def sliding_palette(rgb, window_frac=0.03):
    """Columns cut from a sliding pool of `window_frac` of the picture."""
    h, w, _ = rgb.shape
    px = rgb.reshape(-1, 3).astype(np.float64)

    order = np.argsort((px - px.mean(0)) @ _pc1(px), kind='stable')
    n_win = int(max(h, min(len(order), round(window_frac * w) * h)))

    window = order[:n_win].copy()
    ptr = n_win
    out = np.zeros((h, w, 3), dtype=np.uint8)
    direction = None
    prev = None

    for col in range(w):
        cols_px = px[window]
        v = _pc1(cols_px)
        if direction is not None and v @ direction < 0:
            v = -v                       # keep the axis pointing consistently
        direction = v

        rank = np.argsort(cols_px @ v, kind='stable')
        m = len(window)
        take = (((np.arange(h) + 0.5) * m) / h).astype(np.int64)
        np.clip(take, 0, m - 1, out=take)
        chosen = rank[take]
        dealt = window[chosen]
        colpx = px[dealt]

        if prev is not None:
            if (np.sqrt(((colpx[::-1] - prev) ** 2).sum(1)).mean()
                    < np.sqrt(((colpx - prev) ** 2).sum(1)).mean()):
                dealt, colpx = dealt[::-1], colpx[::-1]
        prev = colpx
        out[:, col, :] = colpx.astype(np.uint8)

        keep = np.ones(m, dtype=bool)
        keep[chosen] = False
        need = min(h, len(order) - ptr)
        window = np.concatenate([window[keep], order[ptr:ptr + need]])
        ptr += need

    return out


# ----------------------------------------------------------------------
# Stage 2: exchange dynamics under field + hollow kernel
# ----------------------------------------------------------------------

S_IN, S_OUT, B_AMP = 2.0, 8.0, 0.13      # difference-of-Gaussians kernel


def _g(r2, s):
    return np.exp(-r2 / (2.0 * s * s)) / (2.0 * np.pi * s * s)


def _w_at(r2):
    """Kernel value at squared separation (for the exact swap term)."""
    return _g(r2, S_OUT) - B_AMP * _g(r2, S_IN)


def _potential(img):
    """w * C, via two separable Gaussian blurs."""
    return (gaussian_filter(img, sigma=(S_OUT, S_OUT, 0), mode='nearest')
            - B_AMP * gaussian_filter(img, sigma=(S_IN, S_IN, 0), mode='nearest'))


def make_field(rgb, gamma=1.0):
    """F(p) = a*x*u1 + b*y*u2 on normalised coordinates.

    gamma=1 gives a=b, so each axis's pull is automatically proportional
    to its own standard deviation.  Whitening (b = 1/sigma_2) would blow
    up when PC2 is numerically degenerate - a greyscale ramp - and sort
    the rows by floating-point noise.
    """
    h, w, _ = rgb.shape
    px = rgb.reshape(-1, 3).astype(np.float64)
    c = px - px.mean(0)
    ev, V = np.linalg.eigh(c.T @ c)
    u1, u2 = V[:, -1], V[:, -2]
    s1, s2 = np.sqrt(ev[-1] / len(px)), np.sqrt(ev[-2] / len(px))
    a, b = 1.0, (s2 / max(s1, 1e-12)) ** (gamma - 1.0)
    ys, xs = np.mgrid[0:h, 0:w]
    X = xs / (w - 1) * 2 - 1
    Y = ys / (h - 1) * 2 - 1
    return a * X[..., None] * u1[None, None, :] + b * Y[..., None] * u2[None, None, :]


def sweep(img, F, lam, rng, T, batch=0.30, min_sep=3):
    """One parallel batch of proposed exchanges, Metropolis-accepted."""
    h, w, _ = img.shape
    n = h * w
    phi = _potential(img)
    m = int(n * batch) // 2 * 2
    s = rng.choice(n, size=m, replace=False)
    a, b = s[:m // 2], s[m // 2:]
    ya, xa = a // w, a % w
    yb, xb = b // w, b % w
    r2 = (ya - yb) ** 2.0 + (xa - xb) ** 2.0
    ok = r2 >= min_sep ** 2               # simultaneous swaps must not interact
    ya, xa, yb, xb, r2 = ya[ok], xa[ok], yb[ok], xb[ok], r2[ok]

    ca, cb = img[ya, xa], img[yb, xb]
    d = cb - ca
    pair = 2.0 * (d * (phi[ya, xa] - phi[yb, xb])).sum(-1) \
        - 2.0 * (d * d).sum(-1) * _w_at(r2)
    fld = 0.5 * (d * (F[ya, xa] - F[yb, xb])).sum(-1)
    gain = pair + lam * fld

    acc = (gain > 0) if T <= 0 else \
        ((gain > 0) | (rng.random(len(gain)) < np.exp(np.clip(gain / T, -60, 0))))
    ya, xa, yb, xb = ya[acc], xa[acc], yb[acc], xb[acc]
    tmp = img[ya, xa].copy()
    img[ya, xa] = img[yb, xb]
    img[yb, xb] = tmp


def refine(seed, rgb, lam=12.0, sweeps=6000, t0=1500.0, t1=0.02, seed_rng=11,
           verbose=True):
    """Anneal `seed` under the field + hollow kernel."""
    img = seed.astype(np.float64)
    F = make_field(rgb)
    rng = np.random.default_rng(seed_rng)
    temps = t0 * (t1 / t0) ** (np.arange(sweeps) / max(sweeps - 1, 1))
    start = time.time()
    for i, T in enumerate(temps):
        sweep(img, F, lam, rng, T)
        if verbose and (i + 1) % 500 == 0:
            print(f"  sweep {i+1}/{sweeps} T={T:8.2f} "
                  f"({time.time()-start:.0f}s)", flush=True)
    for _ in range(300):                  # settle at zero temperature
        sweep(img, F, lam, rng, 0.0)
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------

def smooth_palette(rgb, window_frac=0.03, lam=12.0, sweeps=6000, verbose=True):
    seed = sliding_palette(rgb, window_frac=window_frac)
    if sweeps <= 0:
        return seed
    out = refine(seed, rgb, lam=lam, sweeps=sweeps, verbose=verbose)
    assert np.array_equal(np.sort(out.reshape(-1, 3), axis=0),
                          np.sort(rgb.reshape(-1, 3), axis=0)), \
        "output is not a rearrangement of the input pixels"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('input')
    ap.add_argument('output', nargs='?', default='output.png')
    ap.add_argument('--window', type=float, default=0.03,
                    help='pool size as a fraction of the picture (default 0.03)')
    ap.add_argument('--lam', type=float, default=12.0,
                    help='composition-field strength (default 12)')
    ap.add_argument('--sweeps', type=int, default=6000,
                    help='annealing sweeps; 0 = initialisation only')
    ap.add_argument('--fast', action='store_true',
                    help='initialisation only (seconds instead of minutes)')
    args = ap.parse_args()

    rgb = np.asarray(Image.open(os.path.expanduser(args.input)).convert('RGB'))
    out = smooth_palette(rgb, window_frac=args.window, lam=args.lam,
                         sweeps=0 if args.fast else args.sweeps)
    Image.fromarray(out).save(args.output)
    print(f"saved {args.output}")


if __name__ == '__main__':
    main()
