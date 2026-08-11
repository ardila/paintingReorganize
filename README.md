# paintingReorganize

Reorganize the pixels of a painting into a smooth colour palette. The
output holds **exactly the same pixels** as the input, in the same dense
rectangle, rearranged so the colour field reads as smooth.

## Install

    pip install -r requirements.txt

## Use

    python smooth_palette.py painting.jpg output.png          # full quality (minutes)
    python smooth_palette.py painting.jpg output.png --fast   # initialisation only (seconds)

`--fast` runs stage 1 alone: already a large improvement over the
original algorithm, and quick enough to iterate with.

## Examples

Each strip is: original | old algorithm | new algorithm.

Picasso — Les Demoiselles d'Avignon

![Demo](demo_demoiselles.png "Les Demoiselles d'Avignon")

Van Gogh — The Starry Night

![Demo](demo_starry_night.png "The Starry Night")

Cézanne — The Large Bathers

![Demo](demo_the_large_bathers.png "The Large Bathers")

Kupka — Mme Kupka Among Verticals

![Demo](demo_kupka.png "Mme Kupka Among Verticals")

Full-quality runs, 6000 sweeps each: Kupka 9 min (0.40M px), Bathers
13 min (0.54M px), Demoiselles 27 min (0.90M px), Starry Night 95 min
(3.03M px).  Cost is linear in pixel count.

## Stage 1 — sliding-window initialisation

The original `palette.py` sorted each output column using only that
column's own pixels. The height of a colour boundary is then an order
statistic of ~h samples, so it carries sampling noise of order √h rows —
and neighbouring columns, estimating it independently, disagree. That
disagreement is exactly what appeared as jagged "spikes" and vertical
chatter.

Each column is instead cut from a **window** holding a few percent of the
picture. The sort direction and the composition are estimated from that
whole window, so the noise falls by √(pooled columns), and consecutive
windows overlap by ~97%, so neighbours can barely disagree. Dealing every
k-th pixel of the sorted window keeps each column spanning the full range
— taking a contiguous block instead collapses the picture to colour =
f(x).

Window size is a **noise-versus-two-dimensionality dial**: 1% is cleanest
but flattens toward a horizontal sweep; 15% is richly 2-D but grainy. 3%
is the default.

## Stage 2 — physical refinement

Pixels are annealed under

    E = Σ_{s≠t} w(p_s − p_t)‖C_s − C_t‖²  −  λ Σ_s C_s · F(p_s)

**The one-body field `F(p) = a·x·u₁ + b·y·u₂`** (u₁,u₂ = principal colour
axes) fixes the **composition**. Minimising it alone is precisely the
optimal-transport map from the PC1/PC2 projection onto the grid, so the
palette sweep becomes a genuine *equilibrium* rather than a state that
decays as the run continues. A purely pairwise energy cannot achieve
this: being invariant under rotating the picture, it can only prefer
concentric blobs — with λ=0 the layout collapses into a bullseye.

**The two-body kernel `w`** fixes the **texture**. It is a difference of
Gaussians — repulsive below ~2.5px, attractive from ~3–20px. Plain
attraction at r=1 gathers the leftover colour dimension into 2–4px clumps
that read as grain; a hollow kernel leaves that residual as 1px dither,
which the eye integrates away.

Pixels **exchange** rather than move, so the arrangement is a permutation
at every instant and every site stays filled by construction — no density
term to tune. Because the kernel is short-range and separable, `w * C`
costs two Gaussian blurs instead of a padded FFT (~3.4× faster).

### The annealing gets worse before it gets better

On Demoiselles, local roughness runs **6.3 → 16.6** at peak heat **→ 4.6**
at the end, while the composition dips before exceeding its starting
value. Runs shorter than ~2500 sweeps show only the damage. This is the
single most misleading thing about the method.

| stage | multi-scale Huber (RGB) | fragmentation | local roughness |
|---|---|---|---|
| stage 1 only | 77.4 | 0.004 | 6.28 |
| + refinement, λ=25 | 42.5 | 0.085 | 4.79 |
| **+ refinement, λ=12** | **38.4** | 0.064 | **4.62** |

λ has an optimum: λ=0 gives a bullseye, λ≳75 over-constrains and starts
pinching colour regions apart. Lower λ fragments regions less.

## Notes from the development

**RGB, not CIELab.** Working perceptually seems obviously right and made
every painting visibly worse. Lab's cube root expands differences among
dark colours, so the sort spends resolution separating shadows the eye
cannot distinguish, and that surfaces as streaking. Measuring in Lab
compounds the error, because the metric then shares the algorithm's blind
spot.

**Smoothness metrics do not settle everything.** Eight were tried
(adjacent-pixel sums, multi-scale Huber in Lab and RGB, pyramid
roughness, directional anisotropy, stripe coherence, fragmentation). All
of them rank stage 1 above stage 2, and human judgement puts stage 2
clearly ahead. The reason is that they measure how *large* colour jumps
are and never what *shape* boundaries take: stage 1's straight,
grid-aligned seams cost the same as stage 2's curved, colour-following
ones. `fragmentation` — the share of each colour outside its largest
connected region — is the only measure here that captures structure
rather than smoothness, and it is the one that agreed with the eye on λ.

**Approaches that were tried and rejected**: manifold-rank layouts,
greedy exact-loss polish, blur-target matching, blue-noise rebalancing,
sliced optimal transport, semi-discrete transport (power diagrams),
Isomap "unrolling" of the colour manifold, and deferring misfit pixels to
a holding pool. The last is instructive — deferral degrades the pool's
principal axis, which is the very thing the deferral test depends on, so
eligibility collapses (99.7% → 12%) and the layout falls apart.

**A regression test worth keeping**: a scrambled black-to-white gradient
must come back as constant columns. Its colour cloud is 1-D
(λ₂/λ₁ ≈ 10⁻¹³), so any method that normalises the second axis amplifies
floating-point noise and sorts rows by it. Rank-based layouts score 0.00
error; sliced optimal transport scored 11.5/255.

## An honest note on the metrics

The multi-scale Huber loss (RGB, τ=25) rates the *old* algorithm ahead of
the new one on all four paintings:

| painting | original | old algorithm | new algorithm |
|---|---|---|---|
| Demoiselles | 129.9 | **25.7** | 38.4 |
| Starry Night | 768.0 | **47.6** | 67.5 |
| Large Bathers | 220.2 | **40.1** | 40.7 |
| Kupka | 173.6 | **60.9** | 62.5 |

Human judgement reverses this, consistently and not marginally. The old
algorithm's outputs are blurrier, which minimises adjacent-pixel
differences, while their boundaries are straight and grid-aligned and
their columns visibly striped. Every metric here scores *how large*
colour jumps are; none scores what *shape* the boundaries take, and that
is what the eye is responding to. Treat the numbers as a guide to
parameters (they track λ correctly) and not as the objective.
