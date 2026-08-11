"""Phase 1 prototype: unconstrained particle dynamics for pixel layout.

Pixels become free particles in the continuous rectangle.  No grid, no
permutation constraint - that is deferred to a one-shot optimal
transport quantization later.  The energy is deliberately reshaped for
the free setting:

  * ATTRACTION between similar colours, repulsion between dissimilar:
    E_attr = -sum_ij w(r_ij) cbar_i . cbar_j  with cbar = colour minus
    palette mean, so the dot product is signed.  w is the same 1/r^2
    octave ladder as the grid version.
  * EXCLUDED VOLUME, colour-independent: a short-range repulsion off the
    local density, so density control is decoupled from colour affinity.
    (In the grid version the lattice provided excluded volume for free.)
  * FIELD: each particle feels a constant force lam*(cbar.u1, cbar.u2) -
    the linear composition field's gradient, which is a per-colour
    compass heading.

Forces are computed particle-in-cell: splat centred colour and mass onto
a grid (bilinear), convolve once per octave, sample the gradients back
at particle positions.  O(N log N) per step, same as the grid sweeps.

Integration: damped momentum ("heavy ball"), reflecting walls, adaptive
dt so the fastest particles move ~1 px per step.  Run until the cloud's
mean speed collapses.
"""
import sys

import numpy as np

import torch
from smooth_palette_gpu import MultiScaleKernel, blur


def _splat(pos, vals, H, W):
    """Bilinear splat of per-particle vals (N,K) onto an (K,H,W) grid."""
    x = pos[:, 0].clamp(0, W - 1.001)
    y = pos[:, 1].clamp(0, H - 1.001)
    x0 = x.floor().long()
    y0 = y.floor().long()
    fx = (x - x0.to(x.dtype))
    fy = (y - y0.to(y.dtype))
    grid = torch.zeros(vals.shape[1], H, W, dtype=pos.dtype)
    flat = grid.view(vals.shape[1], -1)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            idx = (y0 + dy).clamp(max=H - 1) * W + (x0 + dx).clamp(max=W - 1)
            wgt = (wx * wy)
            flat.index_add_(1, idx, (vals * wgt[:, None]).T)
    return grid


def _sample(grid, pos):
    """Bilinear sample of an (K,H,W) grid at particle positions -> (N,K)."""
    K, H, W = grid.shape
    x = pos[:, 0].clamp(0, W - 1.001)
    y = pos[:, 1].clamp(0, H - 1.001)
    x0 = x.floor().long()
    y0 = y.floor().long()
    fx = (x - x0.to(x.dtype))
    fy = (y - y0.to(y.dtype))
    flat = grid.view(K, -1)
    out = torch.zeros(pos.shape[0], K, dtype=pos.dtype)
    for dx, wx in ((0, 1 - fx), (1, fx)):
        for dy, wy in ((0, 1 - fy), (1, fy)):
            idx = (y0 + dy).clamp(max=H - 1) * W + (x0 + dx).clamp(max=W - 1)
            out += flat[:, idx].T * (wx * wy)[:, None]
    return out


def _grad(field):
    """Central-difference spatial gradient of (K,H,W) -> d/dx, d/dy."""
    gx = torch.zeros_like(field)
    gy = torch.zeros_like(field)
    gx[:, :, 1:-1] = 0.5 * (field[:, :, 2:] - field[:, :, :-2])
    gy[:, 1:-1, :] = 0.5 * (field[:, 2:, :] - field[:, :-2, :])
    return gx, gy


class FreeLayout:
    def __init__(self, rgb, lam=1.0, k_rep=1.0, rep_sigma=2.0, seed=11):
        H, W, _ = rgb.shape
        self.H, self.W = H, W
        n = H * W
        px = torch.tensor(rgb.reshape(-1, 3).astype(np.float64))
        self.cbar = px - px.mean(0)
        self.cbar /= self.cbar.abs().max()          # colours in ~[-1,1]

        cov = self.cbar.T @ self.cbar
        _, V = torch.linalg.eigh(cov)
        u1, u2 = V[:, -1], V[:, -2]
        # per-particle constant field force (compass heading)
        self.f_field = torch.stack([self.cbar @ u1, self.cbar @ u2], 1)
        self.f_field *= lam

        ys, xs = np.mgrid[0:H, 0:W]
        self.pos = torch.tensor(
            np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1).astype(float))
        g = torch.Generator().manual_seed(seed)
        self.pos += 0.25 * torch.randn(self.pos.shape, generator=g,
                                       dtype=self.pos.dtype)
        self.vel = torch.zeros_like(self.pos)
        self.kern = MultiScaleKernel(min(H, W))
        # contact-scale colour octave: without it there is no colour
        # force below sigma_min=4 and unlike colours interpenetrate
        # freely at 1-3px, which shows up as pervasive speckle
        self.kern.sigmas = [1.5] + list(self.kern.sigmas)
        self.kern.weights = [2.0 * self.kern.weights[0]] + list(self.kern.weights)
        self.k_rep, self.rep_sigma = k_rep, rep_sigma
        self.rgb = rgb

    def forces(self):
        both = torch.cat([self.cbar, torch.ones(len(self.pos), 1,
                                                dtype=self.pos.dtype)], 1)
        grids = _splat(self.pos, both, self.H, self.W)
        cgrid, density = grids[:3], grids[3:4]

        phi = torch.zeros_like(cgrid)
        for s, a in zip(self.kern.sigmas, self.kern.weights):
            phi += a * blur(cgrid, s)
        gx, gy = _grad(phi)
        fx = (_sample(gx, self.pos) * self.cbar).sum(1)
        fy = (_sample(gy, self.pos) * self.cbar).sum(1)
        f_attr = torch.stack([fx, fy], 1)

        # Pressure that stiffens with density (rho^2 equation of state)
        # at two scales, so clumps of any size push back against the
        # multi-scale attraction - single-scale contact repulsion only
        # resists at a clump's skin while its pull grows with its mass.
        f_rep = torch.zeros_like(f_attr)
        for sig, amp in ((self.rep_sigma, 1.0), (4 * self.rep_sigma, 0.5)):
            rho = blur(density, sig)
            p = rho * rho
            rgx, rgy = _grad(p)
            f_rep -= amp * torch.cat(
                [_sample(rgx, self.pos), _sample(rgy, self.pos)], 1)
        return f_attr, f_rep

    def calibrate(self):
        """Fix force coefficients ONCE, near the uniform start.  Per-step
        renormalisation (the first attempt) re-injects motion forever and
        abolishes equilibrium; fixed coefficients let forces cancel."""
        f_attr, f_rep = self.forces()
        A = float(f_attr.norm(dim=1).quantile(0.95)) + 1e-12
        R = float(f_rep.norm(dim=1).quantile(0.95)) + 1e-12
        Fm = float(self.f_field.norm(dim=1).max()) + 1e-12
        self.a_attr = 1.0 / A
        self.a_rep = self.k_rep / R
        self.a_field = 0.15 / Fm

    def step(self, friction=0.15, max_step=1.5):
        if not hasattr(self, 'a_attr'):
            self.calibrate()
        f_attr, f_rep = self.forces()
        f = self.a_attr * f_attr + self.a_rep * f_rep \
            + self.a_field * self.f_field
        self.vel = (1 - friction) * self.vel + f
        speed = self.vel.norm(dim=1)
        scale = torch.clamp(max_step / (speed + 1e-9), max=1.0)
        dt = 1.0
        self.pos += dt * self.vel * scale[:, None]
        # reflecting walls
        for d, hi in ((0, self.W), (1, self.H)):
            low = self.pos[:, d] < 0
            high = self.pos[:, d] > hi
            self.pos[low, d] *= -1
            self.pos[high, d] = 2 * hi - self.pos[high, d]
            self.vel[low | high, d] *= -0.5
            self.pos[:, d].clamp_(0, hi - 1e-3)
        return float((self.vel * scale[:, None]).norm(dim=1).mean())

    def render(self, scale=2):
        """Last-write-wins scatter render; voids show white."""
        img = np.full((self.H, self.W, 3), 255, np.uint8)
        p = self.pos.numpy()
        xi = np.clip(p[:, 0].astype(int), 0, self.W - 1)
        yi = np.clip(p[:, 1].astype(int), 0, self.H - 1)
        img[yi, xi] = self.rgb.reshape(-1, 3)
        if scale > 1:
            img = np.kron(img, np.ones((scale, scale, 1), np.uint8))
        return img
