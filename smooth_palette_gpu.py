"""GPU implementation of the palette rearrangement (PyTorch).

Numerically the same algorithm as `smooth_palette.py`; the whole point is
that every step is data-parallel with no dependencies between proposed
swaps, so a GPU runs it 20-50x faster and the iteration loop stops being
measured in hours.

Two changes over the CPU version, both discussed in the README:

MULTI-SCALE KERNEL.  The CPU version uses a difference of two Gaussians,
which has no tail: two regions of the same colour more than ~3*sigma
apart exert no force on each other at all, so they never merge and
survive as mid-scale mottle.  (The force between two same-colour regions
is proportional to w(d) * |A| * |B| - a large region does pull a small
one, but only within w's reach.)  Here w is a sum of Gaussians on an
octave ladder with weights ~ 1/sigma^2, which approximates a power law:
attraction at every distance, no cliff, and still just a handful of
separable blurs.

SCALE AWARENESS.  The ladder's top sigma is a fraction of the picture's
short side, so behaviour no longer depends on resolution.  Fixed pixel
sigmas were tuned on a 933x960 painting and left a 2203x1376 one covered
in blotches.

Usage:
    python smooth_palette_gpu.py in.jpg out.png [--sweeps 6000] [--lam 12]
                                 [--device cuda] [--top-frac 0.06]
"""

import argparse
import math
import time

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ----------------------------------------------------------------------
# separable Gaussian blur
# ----------------------------------------------------------------------

def _gauss1d(sigma, device, dtype):
    r = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _blur_direct(img, sigma):
    C = img.shape[0]
    k = _gauss1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    x = img.unsqueeze(0)
    x = torch.nn.functional.pad(x, (r, r, 0, 0), mode='replicate')
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1),
                                   groups=C)
    x = torch.nn.functional.pad(x, (0, 0, r, r), mode='replicate')
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1),
                                   groups=C)
    return x.squeeze(0)


def blur(img, sigma):
    """img: (3, H, W).  Gaussian blur, pyramid-accelerated for large sigma.

    A direct separable blur costs O(N * sigma), which forbids the wide
    octaves the long-range tail needs.  Blurring a 2^k-downsampled copy
    at sigma/2^k and upsampling back costs O(N) at ANY sigma, and for a
    smooth potential field the approximation error is irrelevant.
    """
    C, H, W = img.shape
    if sigma <= 12.0:
        return _blur_direct(img, sigma)
    f = 1
    while sigma / (2 * f) > 8.0 and min(H, W) // (2 * f) >= 8:
        f *= 2
    small = torch.nn.functional.avg_pool2d(img.unsqueeze(0), f)
    small = _blur_direct(small.squeeze(0), sigma / f)
    return torch.nn.functional.interpolate(
        small.unsqueeze(0), size=(H, W), mode='bilinear',
        align_corners=False).squeeze(0)


class MultiScaleKernel:
    """w(r) = -b*G_s0(r) + sum_k a_k*G_sk(r),  a_k ~ 1/sigma_k^2.

    The negative innermost term keeps the kernel HOLLOW (repulsive at
    1-2px), so the leftover colour dimension stays 1px dither instead of
    gathering into visible 2-4px clumps.  The positive ladder supplies
    attraction from a few px out to a fraction of the picture.
    """

    def __init__(self, short_side, top_frac=0.5, sigma_min=4.0, hollow=0.5,
                 sigma_hollow=1.5):
        # top_frac=0.5 means the widest octave's 3-sigma reach spans the
        # whole picture; with the ~1/r^2 weighting there is nothing left
        # to tune about "reach", so it is not exposed as a knob.
        # Octave ladder from sigma_min up to a fraction of the picture.
        # EQUAL weights, not 1/sigma^2: a Gaussian of width s contributes
        # ~1/s^2 at its own scale, so equal weights make the sum behave
        # like 1/r^2 across the ladder.  Weighting by 1/sigma^2 instead
        # lets the smallest scale dominate and kills the tail entirely.
        top = max(sigma_min * 2, top_frac * short_side)
        sig, s = [], sigma_min
        while s <= top:
            sig.append(s)
            s *= 2.0
        self.sigmas = sig or [sigma_min]
        a = 1.0 / len(self.sigmas)
        self.weights = [a] * len(self.sigmas)
        # Hollow core: repulsive below ~2.5px so the leftover colour
        # dimension stays 1px dither instead of visible clumps.
        self.s_in = sigma_hollow
        self.b = hollow * a

    def potential(self, img):
        """Single-pass pyramid evaluation of the whole octave ladder.

        Octaves are grouped by the pyramid level they would be computed
        at; each level is pooled once from the previous, blurred with its
        (small) residual sigmas, and results are accumulated coarse-to-
        fine through one chain of upsamples - instead of re-pooling from
        full resolution once per octave."""
        C, H, W = img.shape
        by_level = {}
        for s, a in zip(self.sigmas, self.weights):
            f = 1
            while s / (2 * f) > 8.0 and min(H, W) // (2 * f) >= 8:
                f *= 2
            by_level.setdefault(f, []).append((s / f, a))
        levels = sorted(by_level)
        pyr = {1: img}
        cur = img
        f = 1
        for lv in levels:
            while f < lv:
                cur = torch.nn.functional.avg_pool2d(
                    cur.unsqueeze(0), 2).squeeze(0)
                f *= 2
                pyr[f] = cur
        acc = None
        for lv in reversed(levels):
            part = torch.zeros_like(pyr[lv])
            for sig, a in by_level[lv]:
                part += a * _blur_direct(pyr[lv], sig)
            if acc is None:
                acc = part
            else:
                acc = part + torch.nn.functional.interpolate(
                    acc.unsqueeze(0), size=pyr[lv].shape[1:],
                    mode='bilinear', align_corners=False).squeeze(0)
        if acc.shape[1:] != (H, W):
            acc = torch.nn.functional.interpolate(
                acc.unsqueeze(0), size=(H, W), mode='bilinear',
                align_corners=False).squeeze(0)
        return acc - self.b * blur(img, self.s_in)

    _WTAB_STEP = 0.25

    def _w_table(self, device, dtype, max_r):
        key = (str(device), str(dtype))
        tab = getattr(self, '_wtabs', {}).get(key)
        if tab is None or tab[1] < max_r:
            r = torch.arange(0, max_r + 1.0, self._WTAB_STEP,
                             device=device, dtype=dtype)
            vals = torch.zeros_like(r)
            r2 = r * r
            for sg, a in zip(self.sigmas, self.weights):
                vals += a * torch.exp(-r2 / (2 * sg * sg)) / (2 * math.pi * sg * sg)
            vals -= self.b * torch.exp(-r2 / (2 * self.s_in ** 2)) \
                / (2 * math.pi * self.s_in ** 2)
            if not hasattr(self, '_wtabs'):
                self._wtabs = {}
            self._wtabs[key] = (vals, max_r)
            tab = self._wtabs[key]
        return tab[0]

    def w_at(self, r2):
        """Lookup-table kernel evaluation: one gather instead of one exp
        per octave per proposed pair."""
        r = torch.sqrt(r2)
        max_r = float(r.max()) if r.numel() else 1.0
        tab = self._w_table(r2.device, r2.dtype, max_r + 2.0)
        idx = (r / self._WTAB_STEP).round().long().clamp(max=tab.numel() - 1)
        return tab[idx]

    def describe(self):
        return (f"sigmas={[round(s,1) for s in self.sigmas]} "
                f"hollow_sigma={self.s_in:.1f}")


# ----------------------------------------------------------------------

def make_field(rgb_t):
    """F(p) = x*u1 + y*u2 on normalised coordinates: the composition term."""
    C, H, W = rgb_t.shape
    flat = rgb_t.reshape(3, -1).T
    c = flat - flat.mean(0, keepdim=True)
    cov = (c.T @ c).double()
    ev, V = torch.linalg.eigh(cov)
    u1, u2 = V[:, -1].to(rgb_t.dtype), V[:, -2].to(rgb_t.dtype)
    ys = torch.linspace(-1, 1, H, device=rgb_t.device, dtype=rgb_t.dtype)
    xs = torch.linspace(-1, 1, W, device=rgb_t.device, dtype=rgb_t.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    return (X.unsqueeze(0) * u1.view(3, 1, 1)
            + Y.unsqueeze(0) * u2.view(3, 1, 1))


_PERM_CACHE = {}


def _pair_sample(n, m, device, gen, refresh=16):
    """m random sites without replacement, from a cached permutation.

    A full randperm of n elements every sweep is a top-3 cost at 10 MP.
    The cache refreshes the permutation every `refresh` uses and applies
    a random roll per use, so pairings still vary sweep to sweep while
    sites within a batch stay distinct (which the swap-apply step
    requires - duplicated sites would corrupt the permutation)."""
    key = (n, str(device))
    perm, uses = _PERM_CACHE.get(key, (None, 0))
    if perm is None or uses >= refresh:
        perm = torch.randperm(n, device=device, generator=gen)
        uses = 0
    _PERM_CACHE[key] = (perm, uses + 1)
    off = int(torch.randint(0, n, (1,), generator=gen,
                            device=device).item())
    return torch.roll(perm, off)[:m]


def sweep(img, F, kern, lam, T, batch=0.30, min_sep=3, gen=None, phi=None):
    """One parallel batch of proposed exchanges, Metropolis-accepted.

    phi: pass a precomputed kern.potential(img) to reuse across several
    sweeps.  At low acceptance rates barely any pixels move per sweep,
    so a slightly stale potential is an excellent trade - it is the
    dominant per-sweep cost."""
    C, H, W = img.shape
    n = H * W
    if phi is None:
        phi = kern.potential(img)
    m = (int(n * batch) // 2) * 2
    perm = _pair_sample(n, m, img.device, gen)
    a, b = perm[:m // 2], perm[m // 2:]
    ya, xa = a // W, a % W
    yb, xb = b // W, b % W
    r2 = (ya - yb).to(img.dtype) ** 2 + (xa - xb).to(img.dtype) ** 2
    keep = r2 >= float(min_sep * min_sep)
    a, b, r2 = a[keep], b[keep], r2[keep]

    flat = img.reshape(3, -1)
    pf = phi.reshape(3, -1)
    ff = F.reshape(3, -1)
    ca, cb = flat[:, a], flat[:, b]
    d = cb - ca
    pair = 2.0 * (d * (pf[:, a] - pf[:, b])).sum(0) \
        - 2.0 * (d * d).sum(0) * kern.w_at(r2)
    fld = 0.5 * (d * (ff[:, a] - ff[:, b])).sum(0)
    gain = pair + lam * fld

    if T <= 0:
        acc = gain > 0
    else:
        u = torch.rand(gain.shape, device=img.device, dtype=img.dtype,
                       generator=gen)
        acc = (gain > 0) | (u < torch.exp(torch.clamp(gain / T, max=0.0)))

    ai, bi = a[acc], b[acc]
    tmp = flat[:, ai].clone()
    flat[:, ai] = flat[:, bi]
    flat[:, bi] = tmp
    return int(acc.sum().item()), int(gain.numel())


def sliding_palette_np(rgb):
    """Stage 1 runs on CPU: it is inherently sequential and only seconds."""
    import smooth_palette as sp
    return sp.sliding_palette(rgb, window_frac=0.03)


def calibrate_t0(img, F, kern, lam, accept=0.25, n_sample=200000, gen=None):
    """Pick the starting temperature from the energy landscape itself.

    Sample random swap proposals at the seed and look at the UPHILL ones
    (negative gain).  T0 is set so the median uphill proposal is accepted
    with probability `accept`:  exp(median_gain/T0) = accept.

    Why this beats a hand-picked constant: gain magnitudes scale with the
    image's colour variance and with the kernel, so a T0 tuned on one
    painting at one size (the old hard-coded 1500) melts a different one
    either too much or not at all.  Calibrating to an acceptance RATE is
    dimensionless and transfers.
    """
    C, H, W = img.shape
    n = H * W
    phi = kern.potential(img)
    m = min(n_sample, n // 2 * 2)
    perm = torch.randperm(n, device=img.device, generator=gen)[:m]
    a, b = perm[:m // 2], perm[m // 2:]
    ya, xa = a // W, a % W
    yb, xb = b // W, b % W
    r2 = (ya - yb).to(img.dtype) ** 2 + (xa - xb).to(img.dtype) ** 2
    keep = r2 >= 9.0
    a, b, r2 = a[keep], b[keep], r2[keep]
    flat, pf, ff = img.reshape(3, -1), phi.reshape(3, -1), F.reshape(3, -1)
    d = flat[:, b] - flat[:, a]
    gain = (2.0 * (d * (pf[:, a] - pf[:, b])).sum(0)
            - 2.0 * (d * d).sum(0) * kern.w_at(r2)
            + lam * 0.5 * (d * (ff[:, a] - ff[:, b])).sum(0))
    neg = gain[gain < 0]
    if neg.numel() == 0:
        return 1.0
    med = float(neg.median())
    return med / math.log(accept)          # both negative -> T0 positive


def _frame(img, scale):
    """Downscale ON DEVICE before the GPU->CPU transfer - at 10 MP the
    full-resolution transfer plus CPU resize dominates video capture."""
    t = img
    if scale > 1:
        t = torch.nn.functional.avg_pool2d(t.unsqueeze(0), scale).squeeze(0)
    return t.clamp(0, 255).round().to(torch.uint8).cpu().numpy() \
        .transpose(1, 2, 0)


def run(rgb, sweeps=6000, lam=12.0, device='cuda',
        t0='auto', accept=0.25, t1_frac=2e-5, seed=11, verbose=True, log_every=500,
        frame_every=0, on_frame=None, frame_scale=1, phi_every=1):
    """t0: starting temperature, or 'auto' to calibrate from the seed's
    energy landscape (recommended - transfers across image sizes).
    t1_frac: final temperature as a fraction of t0.
    frame_every/on_frame: every `frame_every` sweeps, call
    on_frame(sweep_index, uint8_HxWx3_numpy) - for building videos."""
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu'
                       else 'cpu')
    seed_img = sliding_palette_np(rgb)
    img = torch.tensor(seed_img.transpose(2, 0, 1).astype(np.float32),
                       device=dev)
    ref = torch.tensor(rgb.transpose(2, 0, 1).astype(np.float32), device=dev)
    H, W = img.shape[1], img.shape[2]
    kern = MultiScaleKernel(min(H, W))
    F = make_field(ref)
    gen = torch.Generator(device=dev).manual_seed(seed)
    if t0 == 'auto':
        t0 = calibrate_t0(img, F, kern, lam, accept=accept, gen=gen)
    t1 = max(float(t0) * t1_frac, 1e-4)
    if verbose:
        print(f"device={dev} {W}x{H} {kern.describe()} "
              f"T0={t0:.1f} T1={t1:.3g}", flush=True)

    def grab(i):
        if on_frame is not None and frame_every and i % frame_every == 0:
            on_frame(i, _frame(img, frame_scale))

    start = time.time()
    grab(0)
    phi = None
    for i in range(sweeps):
        T = t0 * (t1 / t0) ** (i / max(sweeps - 1, 1))
        if phi is None or i % max(phi_every, 1) == 0:
            phi = kern.potential(img)
        sweep(img, F, kern, lam, T, gen=gen, phi=phi)
        grab(i + 1)
        if verbose and (i + 1) % log_every == 0:
            el = time.time() - start
            print(f"  {i+1}/{sweeps} T={T:8.2f} {el:6.1f}s "
                  f"({el/(i+1)*1000:.1f} ms/sweep)", flush=True)
    for j in range(300):
        if j % max(phi_every, 1) == 0:
            phi = kern.potential(img)
        sweep(img, F, kern, lam, 0.0, gen=gen, phi=phi)
        grab(sweeps + j + 1)

    out = img.clamp(0, 255).round().to(torch.uint8).cpu().numpy() \
        .transpose(1, 2, 0)
    assert np.array_equal(np.sort(out.reshape(-1, 3), axis=0),
                          np.sort(rgb.reshape(-1, 3), axis=0)), \
        "output is not a rearrangement of the input pixels"
    return out


def run_constant_motion(rgb, sweeps=6000, lam=12.0, motion=0.03,
                        device='cuda', seed=11, gain_k=0.15, verbose=True,
                        log_every=500, frame_every=0, on_frame=None,
                        frame_scale=1, phi_every='auto'):
    """Servo the temperature so a fixed fraction of proposed swaps is
    accepted on every sweep ("constant motion" annealing).

    Instead of imposing a temperature curve, hold the OBSERVABLE - the
    fraction of proposals accepted - at `motion`.  A multiplicative
    controller nudges T after every sweep: too much motion -> cool, too
    little -> warm.  The consequence is an emergent coarse-to-fine
    schedule: while coarse structure is still negotiable, 3% motion is
    available at high T; once it locks in, the controller must cool to
    keep anything moving, so T decays at exactly the rate the painting's
    own freezing dictates.  This is the practical form of the Lam /
    constant-thermodynamic-speed schedules from the annealing
    literature, with the target set low so the seed is never melted -
    the run sculpts continuously instead of exploding and refreezing.

    Ends with the usual greedy settle to drain residual thermal noise.
    """
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu'
                       else 'cpu')
    seed_img = sliding_palette_np(rgb)
    img = torch.tensor(seed_img.transpose(2, 0, 1).astype(np.float32),
                       device=dev)
    ref = torch.tensor(rgb.transpose(2, 0, 1).astype(np.float32), device=dev)
    H, W = img.shape[1], img.shape[2]
    kern = MultiScaleKernel(min(H, W))
    F = make_field(ref)
    gen = torch.Generator(device=dev).manual_seed(seed)
    T = calibrate_t0(img, F, kern, lam, accept=motion, gen=gen)
    if phi_every == 'auto':
        # staleness budget: fraction of pixels moved between potential
        # rebuilds ~ motion * batch * 2 * phi_every, held near 2.5%
        phi_every = int(np.clip(0.025 / (motion * 0.6), 1, 16))
    if verbose:
        print(f"device={dev} {W}x{H} {kern.describe()} motion={motion} "
              f"T_start={T:.1f} phi_every={phi_every}", flush=True)

    def grab(i):
        if on_frame is not None and frame_every and i % frame_every == 0:
            on_frame(i, _frame(img, frame_scale))

    start = time.time()
    grab(0)
    phi = None
    for i in range(sweeps):
        if phi is None or i % max(phi_every, 1) == 0:
            phi = kern.potential(img)
        acc_n, prop_n = sweep(img, F, kern, lam, T, gen=gen, phi=phi)
        a = max(acc_n / max(prop_n, 1), 1e-4)
        ratio = (a / motion) ** (-gain_k)          # too much motion -> cool
        T = float(np.clip(T * np.clip(ratio, 0.7, 1.4), 1e-4, 1e9))
        grab(i + 1)
        if verbose and (i + 1) % log_every == 0:
            el = time.time() - start
            print(f"  {i+1}/{sweeps} T={T:10.3f} motion={a:.4f} "
                  f"{el:6.1f}s", flush=True)
    for j in range(300):
        if j % max(phi_every, 1) == 0:
            phi = kern.potential(img)
        sweep(img, F, kern, lam, 0.0, gen=gen, phi=phi)
        grab(sweeps + j + 1)

    out = img.clamp(0, 255).round().to(torch.uint8).cpu().numpy()         .transpose(1, 2, 0)
    assert np.array_equal(np.sort(out.reshape(-1, 3), axis=0),
                          np.sort(rgb.reshape(-1, 3), axis=0)),         "output is not a rearrangement of the input pixels"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output', nargs='?', default='output_gpu.png')
    ap.add_argument('--sweeps', type=int, default=6000)
    ap.add_argument('--lam', type=float, default=12.0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    rgb = np.asarray(Image.open(args.input).convert('RGB'))
    out = run(rgb, sweeps=args.sweeps, lam=args.lam, device=args.device)
    Image.fromarray(out).save(args.output)
    print(f"saved {args.output}")


if __name__ == '__main__':
    main()
