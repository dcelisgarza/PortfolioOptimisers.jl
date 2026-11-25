---
status: proposed
---

# A Stress Statement is equation rows on the vine estimator, and the sampler draws it exactly or declares its fallback

## Context

Map [#1080](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1080) builds a
vine-copula synthetic-data prior. ADR 0153 put the Stress Statement on `VineCopulaEstimator`,
copied it onto the fitted Result, and gave the draw a `SimulationResult`. Decision ticket
[#1087](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1087) had to fix the form of a
statement, the route that draws it, the fallback when no exact route exists, and what the Result
reports.

The maintainer's example: a crypto-like asset B at daily frequency, stressed to
`B ≤ −0.10 ∪ B ≥ 0.10`. The facts, measured on 2026-11-23 on the 20-asset S&P 500 panel
(T = 8312, B = AMD), on VineCopulas.jl `main` at `ce4e0436` and Copulas.jl `main` at `ceb476dcb`
(1.1.0), single thread, and on the reference implementation at its newest tag:

- **The tail mass.** P(|r_B| ≥ 0.10) is 0.0256 in the data and 0.0244 under the fitted Student-t
  marginal. A Student-t with ν = 3 and a standard deviation of 4 % gives 0.0227.
- **VineCopulas.jl `main` conditions on points.** `rand(rng, vine, n; fixed = (js, Ujs))` holds the
  coordinates `js` at stated uniforms and draws the rest exactly, when
  `admits_conditioning(vine, js)` is true, that is when `js` heads a Sampling Order. Otherwise it
  refuses. `fit(RVineCopula, U; sampling_tail = js)` puts `js` at the tail of the order where the
  trees allow it.
- **Which sets head an order.** With `sampling_tail`: every single asset (20 of 20), the 19
  tree-1 edges among the 190 pairs and no other pair, and 18 of the 1 140 triples. Without it: one
  asset, one pair, one triple. `sampling_tail` changes only the reading of the trees into an order
  (`_rvine_peel`), not the trees: a peel of one cached set of trees reproduces the public fits
  exactly. A public refit costs 182–191 s here.
- **The one-asset exact draw costs nothing.** 10 000 scenarios with u_B uniform on the union of the
  two tail intervals took 13.9 s, against 13.5 s unconditional. Every row is in the set. Kendall
  τ(B, C) with C = MSFT is 0.43 under the stress against 0.27 unconditional. Rejection from the
  unconditional vine accepts 2.46 % and costs 55 ms per accepted scenario.
- **A block of two assets drawn by a Guided Draw is exact and cheap.** For {B, C} both under
  |r| ≥ 0.10, rejection on the full vine accepts 0.14 % at 972 ms per accepted scenario. The Guided
  Draw accepts 41.9 % at 1.46 ms. The two agree: P(box) 0.00140 ± 5.6e-5 against
  0.00139 ± 6.8e-7, τ(B, C) 0.521 ± 0.027 against 0.517, every other column mean within
  Monte-Carlo error. For both assets under their own 5 % quantile, the two give τ 0.180 ± 0.009
  and 0.177.
- **The reference implementation** conditions by a scalar, a per-scenario array or a `(min, max)`
  pair per asset. Its one-asset draw is exact when the asset is central. When the asset is not
  central, which is its default, the statement reaches only the column of B: τ(B, C) is −0.010
  against 0.101 by rejection, and the column means miss by up to 48.9 standard errors, with a
  warning as the only signal. A block draws each u as an independent uniform on its interval: τ is
  0.015 against 0.167. A union cannot be stated. Two statements on one asset keep the last, with no
  warning. Its prior reports nothing about the route or the probability, and a re-stress refits.
  It draws a scenario in 0.28 ms at full depth, against 1.32 ms for VineCopulas.jl `main`.
- **The cost in VineCopulas.jl** is partly a per-observation rebuild of the Student-t kernel:
  preparing it once per edge gives 0.76 ms and bit-identical draws. That is upstream draft 13.
- **The string grammar exists.** Entropy pooling views are equation strings in a
  `LinearConstraintEstimator`, `parse_equation` takes one of `==`, `<=`, `>=`, and
  `replace_group_by_assets(…; bl_flag = true)` reads a group as the mean of its members, the
  Black-Litterman meaning.

## Decision

**The form.** A `StressStatement` holds `val`, a vector of equation strings, `path`, a dictionary
from an asset name to a vector of one value per scenario, and `strict = false`, as the views do. A
row is split on `||` into pieces, and each piece is parsed by `parse_equation`, which does not
change. A union is the `||` of its pieces. The rows on one asset intersect. A coefficient on a
single asset rescales the bound, and a negative one turns it. A row may name a linear combination
of assets, and a group name in a row is the mean of its members. Names resolve through Universe
Sets, so inside a `FactorPrior` they are the factors, and a factor stress projects onto the assets
through the loadings with no new code. `≠`, `<` and `>` are not in the grammar.

**The refusals use the existing machinery.**

| Case | Where | Machinery and result |
| --- | --- | --- |
| an operator other than `==`, `<=`, `>=` | constructor | `parse_equation`, `ParseError` |
| a malformed or too long row | constructor | `parse_equation` with `EQUATION_LIMITS` |
| rows on one asset with an empty intersection | constructor | `assert_ordered_bounds(lo, hi)` |
| a point and an interval on one asset | constructor | the point if it is inside, else `assert_ordered_bounds` |
| an asset in both `val` and `path` | constructor | `@argcheck`, naming both |
| an empty or non-finite `path` | constructor | `assert_nonempty_finite_val` |
| `path` vectors of unequal length | constructor | `@argcheck`, `DimensionMismatch` |
| a row whose coefficients are all zero | constructor | `@argcheck` |
| a group name | fit | `replace_group_by_assets(…; bl_flag = true)`, the mean |
| an unknown name | fit | `name_to_val!` and `strict_diagnostic`: throw under `strict`, else warn and drop the row |
| a name on the Non-Investable Axis | fit | the row drops whole and the entry of the fit reports it once (ADR 0125) |
| every asset fixed by a point or a path | fit | `@argcheck`: nothing is left to draw |
| `length(path) != n` | `simulate` | `@argcheck`, `DimensionMismatch` |
| `n` above `max_n_sim` | `simulate` | `assert_resource_cap` |

**The fit arranges the Sampling Order.** `VineCopulaEstimator` gains a field `ord`:

- `ForcedSamplingOrder(; strict = false)`, the default, passes the stated assets as the
  `sampling_tail`. For a block of two or more it also boosts their edges into tree 1 through a
  `tree_criterion` function `(u_a, u_b, a, b, D) -> Real`. Constant boosts of +2 and +3 dominate
  `|τ| ≤ 1`, so they give the spanning tree of the reference's `+2·max` and `+3·max`. After the fit
  it checks `admits_conditioning`. When the check fails, `strict = true` throws and `strict = false`
  warns, and the sampler takes the declared fallback.
- `TailSamplingOrder()` passes only the `sampling_tail`. The trees stay those that the data picks,
  and a block that they do not admit takes the fallback with no warning.

**The sampler.**

1. It reads the fitted trees again with the stated assets as the tail, with no refit. Until
   VineCopulas.jl ships upstream draft 14, it calls the internal `_rvine_peel` with rebuilt
   `_RVFitEdge` records (ADR 0196: no public route gives the same result). Draft 14 lists each
   internal the prior calls, so that the upstream author and the maintainer can choose what to make
   public. This peel orders the tail by label, so a point that is not generated first in a block
   with an interval takes the fallback until the ordered tail of draft 14 lands.
2. Points and paths that head the order take `fixed = (js, U)`, the route **exact**. Sets that head
   the order take the Guided Draw, the route **guided**. Combination rows accept or reject each
   scenario of that exact draw, which stays exact.
3. The stated value of a point is written back exactly after the quantile map, and `jitter`
   (ADR 0153) then repairs the constant column. `SimulationResult` names the column in its mask.

**The fallback.** `VineCopulaEstimator` gains a field `fb`:

- `RejectionFallback()`, the default. A set on a block that no order admits is drawn by rejection
  from the unconditional vine, the route **rejection**, which is exact. A point on such a block
  refuses with a sentence that names `ForcedSamplingOrder` and the tree-1 rule.
- `KernelFallback(; bandwidth = nothing, min_ess = 100, resample = true)`. A point on such a block,
  or on a combination, is drawn by importance with a Gaussian kernel on the normal scores
  z = Φ⁻¹(u), the route **kernel**, which is approximate and declared so. `bandwidth = nothing` is
  the Silverman rule 1.06 · n^(−1/(4+k)), with k the number of point coordinates; a number replaces
  it. Below a Kish count of `min_ess` the draw refuses; a value below 1 is a fraction of `n`.
  `resample = true` resamples by the weights. `false` gives the weighted panel, with `w` and `eoc`
  on the `LowOrderPrior` and the inner `pe` rebound, as ADR 0153 set for its importance route.

**The cap.** `RESOURCE_LIMITS` gains `max_n_draws = 1_000_000`, with the preference key
`"max_n_draws"` (ADR 0041). It caps the proposals of every loop: the Guided Draw, rejection and the
kernel. After the first 10 000 proposals the sampler projects the total as `n / acceptance`, and
refuses at once if it is over the cap. The message gives the acceptance, `p_joint` with its
standard error, and the knob to raise.

**The report.** `SimulationResult` gains `p_row`, the probability of each row under the fitted
model (the marginal mass of a set; the CDF level of a point or of each path value), `p_joint`, the
probability of all the rows together with its standard error (the acceptance of a rejection draw,
or P(S₁) times the mean acceptance of a Guided Draw), and the rows that a departed asset dropped.
A small probability gets a number and no warning.

**Upstream.** The map works off the `main` branch of both Copulas.jl and VineCopulas.jl, as the
maintainer ruled on 2026-11-23. Drafts 13 to 17 are comments on #1080, for the maintainer to post:
13, prepare the pair kernel once per edge; 14, a re-read of a fitted vine into a new order, with the
list of internals; 15, a Guided Draw for a box or a union; 16, a release from `main`; 17, the
`order` export that clashes with DataFrames.jl. No Copulas.jl gap was found.

## Considered options

- **Typed sets keyed by name, or typed sets with strings that parse into them.** Considered: one
  form for all four cases, and a typed interval refuses `lo ≥ hi` in its constructor. Rejected by
  the maintainer: the strings use the syntax of the views, need no new set types, and already write
  a combination. The one case that a string cannot hold, the per-scenario vector, is `path`.
- **Three order algorithms.** "Force, check, fall back" without a warning was a third type.
  Rejected: it is `ForcedSamplingOrder(; strict = false)` less its warning, and `strict` already
  carries that difference in the library.
- **`strict = true` as the default of the order.** Rejected by the maintainer, for the library's
  `strict = false` precedent.
- **Best effort only, or a refusal at the fit.** Both are now members or settings of the family
  above, not the only behaviour.
- **Points: refuse, kernel, or both.** The maintainer kept both behind a fallback field, with the
  refusal as the default.
- **A kernel on the uniform scale.** Rejected: a tail point at u = 0.0004 puts the kernel past the
  boundary at 0. The normal scores are symmetric there.
- **A plain cap, an acceptance floor, or a cap per type.** Rejected for one global limit with an
  early projection, so a 0.001 % event fails in seconds and the cost stays bounded.
- **The group as a sum, as one row per member, or a refusal.** Rejected for the mean: a statement
  is on returns, and the Black-Litterman views read a group of returns as its equal-weight basket.
  One row per member is written as one row per name.
- **A departed asset refuses.** Rejected: the Departed View rule keeps a walk-forward running in
  the windows where the asset is absent, and the entry reports the drop once.
- **The library's own peel on the public API, or no interim at all.** The maintainer chose the
  internal peel, listed on draft 14, as the case for what upstream should make public.
- **The reference implementation's block draw.** Rejected: independent uniforms on the intervals
  lose the dependence inside the block, as measured above.

## Consequences

- `src/` gains `StressStatement`, the order family (`ForcedSamplingOrder`, `TailSamplingOrder`),
  the fallback family (`RejectionFallback`, `KernelFallback`), the fields `ord` and `fb` on
  `VineCopulaEstimator`, the limit `max_n_draws`, and the fields `p_row`, `p_joint` and the dropped
  rows on `SimulationResult`. The abstract families stay unexported, and their members are
  exported.
- The extension calls `_rvine_peel`, `_RVFitEdge` and `_PairSelection` of VineCopulas.jl until
  upstream draft 14 lands. Each build ticket that calls another internal adds it to the table on
  draft 14.
- ADR 0152 is amended: the pin `Copulas = "0.1.41 - 0.1.41"` no longer holds, because
  VineCopulas.jl `main` needs Copulas.jl 1.x and no registered VineCopulas.jl works with it.
- ADR 0153 is amended: its importance route is `KernelFallback`, and its `Bool` is `resample`.
- `GLOSSARY.md` gains **Stress Statement**, **Sampling Order** and **Guided Draw**, and
  **Simulation Result** names the four routes and the probabilities.
- Build ticket [#1092](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1092) is
  sharpened to this decision.
