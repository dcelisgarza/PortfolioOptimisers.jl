---
status: accepted
---

# A zero Gerber band edge makes an exactly zero return neutral

## Context

[#491](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/491), a child of
[#417](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/417), opened on a raise from
the sweep of [#454](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/454).
[`GerberCovariance`](../../src/08_Moments/05_GerberCovariance.jl) answered a matrix whose diagonal
is not one, so it was not a correlation matrix:

```julia
# Eight observations, two assets, each column centred exactly. Rows 4, 5 and 6 carry an
# exactly zero return.
X = [2.0 2.0; 2.0 -2.0; -2.0 -2.0; 2.0 0.0; 0.0 2.0; 0.0 0.0; -2.0 2.0; -2.0 -2.0]

diag(cor(GerberCovariance(; t = 0.0, pdm = nothing, alg = Gerber0()), X))  # [0.4286, 0.4286]
diag(cor(GerberCovariance(; t = 0.0, pdm = nothing, alg = Gerber1()), X))  # [0.75, 0.75]
```

### The cause

`gerber_updown` built the two bands with a closed comparison on each side:

```julia
ts = sd * ce.t
U .= X .>= ts
D .= X .<= -ts
```

The guard of `t` is `assert_nonempty_nonneg_finite_val(t, :t)`, and `val_dict[:gerbt]` states
`0 <= t`, so `t = 0` is legal. At `t = 0` the band edge `ts` is zero, and `x >= 0` and `x <= -0`
both hold when `x` is exactly zero. The observation was marked in **both** `U` and `D`.

Downstream, `H = U - D` is zero on that observation while `V = U + D` is two, so
`concordance_counts` recovered an `nconc` and an `ndisc` that were no longer counts of
observations. `Gerber2` was unaffected on the diagonal, because it normalises by the diagonal of
`H' * H` itself.

The overlap needs a return that is exactly zero after centring, so the defect is reachable from
synthetic data and from a caller who centres the data themselves.

### The two readings

The ticket named both.

 1. **Tighten the guard to `0 < t`.** The Gerber statistic is defined for a strictly positive
    threshold, and Riskfolio-Lib asserts `0 < threshold < 1`. `val_dict[:t]` already carries
    `0 < t < 1`, so the neighbouring wording exists.
 2. **Make the bands disjoint at a zero edge.** An exactly zero return becomes neutral rather than
    both up and down. `t = 0` stays legal and becomes the sign concordance, which is what a reader
    expects of a zero threshold.

## Decision

**A zero band edge makes an exactly zero return neutral.** `gerber_updown` adds a sign test to
each band:

```julia
zx = zero(eltype(X))
U .= (X .>= ts) .& (X .> zx)
D .= (X .<= -ts) .& (X .< zx)
```

**The sign test binds only at a zero edge.** `ce.t` and `sd` are both non-negative, so `ts` is
non-negative. For a positive `ts` the test is redundant, because `x >= ts > 0` already implies
`x > 0`. Every result at a positive threshold is therefore unchanged, bit for bit.

**Both bands are strict at a zero edge, not one of them.** The ticket wrote "make one band
strict", which would put a zero return in `U` alone. That is not the neutrality the same sentence
asked for: `H = U - D` would then be `1` on a return that has no sign, and the statistic would not
be the sign concordance. Excluding zero from both bands gives `H` the sign of the return and `V`
the indicator that the return is not zero.

**`t = 0` stays legal.** `val_dict[:gerbt]` keeps `0 <= t`. Reading 1 rejects a `t` that a caller
may be passing today, and it removes a threshold that now has a clean meaning.

## Consequences

- **A zero threshold is the sign concordance.** `U` marks a positive return, `D` marks a negative
  one, and a zero return is neutral. `Gerber1` counts it as neutral, which is what its neutral
  matrix `Nt = .!U .& .!D` already means.
- **The diagonal is unit again.** On the sample above every variant answers a unit diagonal, so
  the result is a correlation matrix at `t = 0` as it is at every other threshold.
- **No result at a positive threshold moves.** The added test cannot fire when the band edge is
  positive, and `Statistics.cor` raises `sd` to at least `eps`, so the `cor` path reaches a zero
  edge only through `t = 0`.
- **A caller who passes `t = 0` over data with exact zeros sees a different number.** That is the
  defect this ADR fixes, and it is the whole size of the behaviour change.
- **`gerber_updown` is reachable with a zero `sd`.** A caller who calls it directly rather than
  through `Statistics.cor` can pass a zero standard deviation for one asset. That asset's edge is
  zero, and the same rule applies to it alone.
- **The math dictionary states the rule once.** `math_dict[:U_gerber]`, `math_dict[:D_gerber]` and
  `math_dict[:Nneut_gerber]` carry the sign test, so every Gerber docstring that interpolates them
  states it.
- **`test_08_moments.jl` holds the contract.** The testset `Gerber statistic (#454)` pins that the
  bands of the sample at `t = 0` are the bands at `t = 0.5`, that the exactly zero returns are in
  neither band, and that all three variants answer a unit diagonal.
