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
    k = _gauss1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    x = img.unsqueeze(0)
    x = torch.nn.functional.pad(x, (r, r, 0, 0), mode='replicate')
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1).expand(3, 1, 1, -1),
                                   groups=3)
    x = torch.nn.functional.pad(x, (0, 0, r, r), mode='replicate')
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1).expand(3, 1, -1, 1),
                                   groups=3)
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
        out = torch.zeros_like(img)
        for s, a in zip(self.sigmas, self.weights):
            out += a * blur(img, s)
        return out - self.b * blur(img, self.s_in)

    def w_at(self, r2):
        acc = torch.zeros_like(r2)
        for s, a in zip(self.sigmas, self.weights):
            acc += a * torch.exp(-r2 / (2 * s * s)) / (2 * math.pi * s * s)
        acc -= self.b * torch.exp(-r2 / (2 * self.s_in ** 2)) \
            / (2 * math.pi * self.s_in ** 2)
        return acc

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


def sweep(img, F, kern, lam, T, batch=0.30, min_sep=3, gen=None):
    """One parallel batch of proposed exchanges, Metropolis-accepted."""
    C, H, W = img.shape
    n = H * W
    phi = kern.potential(img)
    m = (int(n * batch) // 2) * 2
    perm = torch.randperm(n, device=img.device, generator=gen)[:m]
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
    return int(acc.sum().item())


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


def run(rgb, sweeps=6000, lam=12.0, device='cuda',
        t0='auto', accept=0.25, t1_frac=2e-5, seed=11, verbose=True, log_every=500,
        frame_every=0, on_frame=None):
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
            on_frame(i, img.clamp(0, 255).round().to(torch.uint8)
                     .cpu().numpy().transpose(1, 2, 0))

    start = time.time()
    grab(0)
    for i in range(sweeps):
        T = t0 * (t1 / t0) ** (i / max(sweeps - 1, 1))
        sweep(img, F, kern, lam, T, gen=gen)
        grab(i + 1)
        if verbose and (i + 1) % log_every == 0:
            el = time.time() - start
            print(f"  {i+1}/{sweeps} T={T:8.2f} {el:6.1f}s "
                  f"({el/(i+1)*1000:.1f} ms/sweep)", flush=True)
    for j in range(300):
        sweep(img, F, kern, lam, 0.0, gen=gen)
        grab(sweeps + j + 1)

    out = img.clamp(0, 255).round().to(torch.uint8).cpu().numpy() \
        .transpose(1, 2, 0)
    assert np.array_equal(np.sort(out.reshape(-1, 3), axis=0),
                          np.sort(rgb.reshape(-1, 3), axis=0)), \
        "output is not a rearrangement of the input pixels"
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
