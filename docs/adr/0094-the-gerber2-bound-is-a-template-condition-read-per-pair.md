---
status: accepted
---

# The `Gerber2` bound is a template condition, and its denominator is read per pair

## Context

[#494](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/494) and
[#500](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/500) are children of
[#417](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/417). Each reported the same
symptom from a different cause: `GerberIQCovariance` under `Gerber2` answered a "correlation" matrix
with entries outside `[-1, 1]`. ADR 0093 measured both and fixed neither, because each needed a
decision that #495 did not need.

`Gerber2` reduced a pair to the net score `pos - neg`, and `standardise_comovement!` afterwards
divided the assembled matrix by the geometric mean of its own diagonal. That division is a
Cauchy-Schwarz bound only when two separate things hold.

### Cause 1: the template

The diagonal entry of asset `i` is built from the weights of the pair `(i, i)`. Both returns are
then the same number, so they fall in the same magnitude class and the co-movement is concordant.
Exactly six of `FullGerberIQ`'s twenty-one weights sit on that diagonal, one per class, and the
other fifteen each join two distinct classes, `C(6, 2) = 15`. Cauchy-Schwarz needs every one of the
fifteen to be at most the geometric mean of the two diagonal weights of the classes it joins.

`clamp_gerber_iq_n` enforced **four** of the fifteen for `FullGerberIQ` and **two** of the six for
`PartialGerberIQ`. The four were the ones whose defaults were already written that way. #494 recorded
the gap and left the reading of a discordant channel open.

Four of the **shipped** `FullGerberIQ` defaults broke the rule. `n15`, `n16`, `n18` and `n21` each
defaulted to the geometric mean of two **mixed** weights rather than of the two **diagonal** weights
of its channel, and each exceeded its bound by `(n4^2 / (n1 n11))^(1/4) = 1.0299`. A four-row,
two-asset sample returned exactly that number.

### Cause 2: the scaler

`gerber_iq_scaling` fixes the units a pair's thresholds are measured in. Only
`AssetVolatilityGerberIQScaler` reads each asset's own volatility. The fall-through reads the pair
mean, and a `Function` reads what it likes. Under such a scaler an asset's magnitude class **moves
with its partner**, so the class it takes off the diagonal is not the class the assembled diagonal
recorded, and Cauchy-Schwarz has nothing to stand on.

Four rows and two assets show it with a template that meets every condition of cause 1. Asset one is
ten times as volatile as asset two. Under `min` both are judged against asset two's volatility, so
asset one is large and asset two is small and the pair entry is `n7`. On each diagonal both are
small, so each diagonal entry is `n1`. The statistic is `n7 / n1`, which is `sqrt(2)`. On the
`SP500` sample of `test_08_moments.jl`, eleven entries left `[-1, 1]` and the largest was `1.224`.

### What the source says

The source is Gerber, Smyth, Markowitz, Miao, Ernst and Sargen, *Squeezing financial noise*,
SSRN 4986939 (2025), the `gerber2025squeezing` key. Its main text states one statistic, equation
(4), which is the `Gerber1` branch. Footnote 1 of section 3.3.2 points at its internet appendix,
which states an alternative form whose denominator is a geometric mean, and which "necessitates
constraints on the values of certain ηk".

The appendix states the constraint and proves it in its equations (A2) to (A9):

> A **necessary and sufficient** condition for |ρij| ≤ 1 is achieved by **scaling the weights** for
> squeezing channels such that the weight of any off-diagonal channel does not exceed the geometric
> mean of its corresponding diagonal channel weights. Specifically, this requires
> η²(ω_αβ) ≤ η(ω_αα) η(ω_ββ) with α, β ∈ {a, b, c, d}. Diagonal weights are permitted to vary freely
> in [0, 1].

Three things follow. The rule is the source's, not an inference. "Scaling the weights" is the
source's own remedy, so a clamp is the sanctioned repair. The condition runs over **every** pair of
classes, so a discordant channel obeys it exactly as a concordant one does, which answers the
question #494 left open.

The appendix's own denominator is not the one this library shipped. It projects each co-movement of
the **pair** onto the lead diagonal in the `i` direction and in the `j` direction, and it sums those
projections over the pair's own observations. The library divided by a diagonal built at the pair
`(i, i)` instead. That difference is cause 2.

## Decision

**The bound is a condition on the template, and the denominator is read in the pair's own units.**
Three changes carry it, and the first two are the source's rule.

**1. The clamp covers every channel that joins two classes.** `clamp_gerber_iq_n` lowers all fifteen
mixed weights of `FullGerberIQ` and all six of `PartialGerberIQ`, each onto the geometric mean of
the two diagonal weights of the classes its channel joins. `PartialGerberIQ` takes a case split,
because a concordant pair is judged against `dcp`/`dcn` and a discordant one against `ddp`/`ddn`: a
return beyond a discordant boundary need not be beyond the concordant one that fixes its diagonal
class, so where the two boundaries disagree the method takes the smaller of the two candidates.

**2. Every shipped default is written from the diagonal weights.** Only the six diagonal weights of
`FullGerberIQ`, and the four of `PartialGerberIQ`, carry a free default. Each remaining default is
the geometric mean of the two diagonal weights of its channel. The default template therefore meets
the bound whatever the diagonal weights are set to, and the clamp never moves a default. Writing the
defaults this way is what keeps rule 1 from silently rewriting a shipped estimator.

**3. `Gerber2` accumulates its denominator per pair.** A new `iq_add_diagonal` folds two more sums
into the pair accumulator, `di` and `dj`. At each observation on which asset `i` left the noise zone
it adds the weight of the projected co-movement `(x_i, x_i)`, judged in **this pair's** units, and
`dj` is the mirror. `comovement_finalise` then answers `(pos - neg) / sqrt(di * dj)`. `gerber_IQ` no
longer calls `standardise_comovement!`.

Rule 3 keeps the classic reduction. Under uniform weights `di` is the number of observations on
which asset `i` crossed, so the branch is still the classic Gerber version-2 denominator, and it
still reduces to `GerberCovariance`. It is **not** the appendix's form, whose sums run over the
observations on which **both** assets crossed and which reduces to `Gerber0` instead. That form is a
fourth reduction the library does not ship.

**`sqrt(di * dj)`, not `sqrt(di) * sqrt(dj)`.** The first makes the diagonal exactly one, because the
pair `(i, i)` makes the numerator and both projections the same sum. The second keeps bit-exact
continuity with the classic family at the cost of a diagonal that is one to within a unit in the last
place. The exact diagonal is worth more: ADR 0093 records that `posdef!` reads the diagonal with an
exact `isone` test, and that the two branches it chooses between answer differently.

**`posdef!` is not touched.** Its exact `isone` test stays exact. This decision removes the reason
the test was fragile for this family rather than changing the test for every family. #500's second
finding stays open on those terms.

## Consequences

- **The statistic is bounded, and the bound needs no precondition on `sc`.** Every scaler is legal
  under every marker. The numerator and the denominator read an asset's class in the same units, so
  a pair-dependent scaler moves both together.
- **The diagonal of a `Gerber2` Gerber IQ matrix is exactly one**, so `posdef!` takes the
  correlation branch by construction. ADR 0093's `iszero` guard keeps its remaining case, the
  degenerate asset, and loses its Gerber IQ rounding case.
- **Two pinned columns of `test/assets/covariance.csv.gz` move**, and they are regenerated in the
  same commit. Column 40 is `FullGerberIQ()` with `Gerber1()`, which moves by `2.08e-6` because four
  shipped defaults changed. Column 41 is `PartialGerberIQ()` with `Gerber2()` and a `min` scaler,
  which moves by `2.37e-4` because it is #500's own reproduction. The other thirty-nine columns were
  recomputed and are bit-exact, so the regenerated file changes only what this decision changes.
- **`FullGerberIQ()` and `PartialGerberIQ()` now answer the same matrix.** The Full default sets
  `dp1 == dp2` and `dn1 == dn2`, so its moderate bands are empty and its six classes collapse to
  Partial's four, and the Partial default sets `ddp == dcp` and `ddn == dcn`. Once every mixed
  default is the geometric mean of its two diagonal weights, the two templates coincide there. An
  inconsistent `n15` was the only thing that separated them before.
- **The reduction to `GerberCovariance` is bit-exact for `Gerber0` and `Gerber1`, and exact to one
  unit in the last place for `Gerber2`.** The classic family divides by `sqrt(di) * sqrt(dj)`
  through `standardise_comovement!`, and this family groups the same three operations as
  `sqrt(di * dj)`. `test_08i_gerberiq.jl` pins the gap at one `eps`.
- **A caller who passes a hand-tuned template sees more weights lowered than before.** The clamp was
  always silent and always replaced `kind`; it now covers the whole table. A template that already
  met the bound is returned unchanged, and `clamp_gerber_iq_n(FullGerberIQ(), Gerber2()) ==
  FullGerberIQ()`.
- **`standardise_comovement!` keeps its three other families.** The classic Gerber and the
  Smyth-Broby families threshold each asset in its own units, so their assembled diagonal is the
  number the pair would have computed. Nothing there moves.
- **The source's alternative form is not implemented.** It is a fourth denominator, it differs from
  `Gerber2` by up to `0.7587` on the `SP500` sample, and its equation (A1) as typeset carries the
  concordance sign inside both denominator sums while its proof treats them as sums of non-negative
  weights. A future ticket that adds it must implement the proof and not the typography.
- **`test_08i_gerberiq.jl` holds the contract.** `The Gerber2 bound holds (#494, #500)` pins that no
  shipped default owes the clamp anything, that both four-row reproductions now return one exactly,
  that the discordant channel `n13` is clamped like any other, that a pair-dependent scaler answers
  the same matrix as a separable one on the volatility-spread sample, that the diagonal is exactly
  one under all three scalers, and that sixty random clamped templates never leave `[-1, 1]`.
