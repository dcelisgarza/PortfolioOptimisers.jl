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

### The same edge in the Gerber IQ family

[#498](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/498), opened on a raise from
the documentation ticket [#456](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/456),
found the same edge in
[`GerberIQCovariance`](../../src/08_Moments/35_GerberIQCovariance.jl):

```julia
X = [1.0 2.0; 0.0 0.0; -1.0 3.0; 2.0 -1.0; 0.0 1.0; -2.0 -2.0]

ce = GerberIQCovariance(; c = 0.0, kind = BasicGerberIQ(; d = 0.0, n = 1.0),
                        sc = AssetVolatilityGerberIQScaler(),
                        decay = ExpGerberIQDecay(; e = 0.0, y = 0.0), pdm = nothing,
                        alg = Gerber1(), me = SimpleExpectedReturns(; w = nothing))

diag(cor(ce, X))     # [0.6667, 1.0]
```

The shape differs. That family has one noise threshold `c` rather than two band edges, and
`comovement_step` gated on it with a strict `<` on the **absolute** return:

```julia
if axi < st.ci && axj < st.cj
    return acc
end
```

At `st.ci == 0` the test `axi < 0` is never true, so no observation is dropped. An observation
whose return is exactly zero then reached the classification, where `axi >= st.ci` holds but
neither sign test does, and it fell through to the neutral branch. There the two bands
*overlapped* at a zero edge; here the gate *failed to exclude* a zero return. Both make a zero
return count as something it is not.

For an off-diagonal pair that is merely generous. For the diagonal pair `(i, i)` it is wrong: the
neutral count holds the observations on which **exactly one** asset crossed, and an asset compared
with itself can never be one of those. `Gerber1` divides by `pos + neg + nn`, so the
inflated `nn` pulled the diagonal below one. `Gerber0` and `Gerber2` read no neutral count and were
unaffected, exactly as `Gerber2` was unaffected in the classic family.

The same `c = 0` also broke the reduction that ties the two families together. Gerber IQ with every
weight set to one and no decay is the Gerber statistic, and at a zero threshold the two answered
different matrices.

### The same edge in the Smyth-Broby family

[#499](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/499), opened while #498 was
fixed, found the same edge in
[`SmythBrobyCovariance`](../../src/08_Moments/06_SmythBrobyCovariance.jl):

```julia
# The column means are exactly zero, so rows 3 and 4 carry an exactly zero centred return
# for asset 1.
X = [2.0 2.0; -2.0 -2.0; 0.0 1.0; 0.0 -1.0]

ce(a) = SmythBrobyCovariance(; alg = a, c1 = 0.0, c2 = 0.0, c3 = 1e6, pdm = nothing,
                             me = SimpleExpectedReturns(; w = nothing))

diag(cor(ce(SmythBroby1()), X))       # [0.6899, 1.0]
diag(cor(ce(SmythBrobyGerber1()), X)) # [0.6899, 1.0]
diag(cor(ce(SmythBrobyCount1()), X))  # [0.5, 1.0]
```

The shape is the Gerber IQ one. `comovement_step` gated the indecision zone with a strict `<` on
the **centred, standardised** magnitude, so at `pol.c2 == 0` the test `ari < 0` is never true and
no observation is dropped. An observation whose centred return is exactly zero then reached the
classification, where `ari >= c2` holds but neither sign test does, and it fell through to the
neutral branch. The three `*1` markers divide by `pos + neg + nn`, so the inflated `nn` pulled the
diagonal below one. The `*0` and `*2` markers read no neutral count and agreed anyway.

`c2 = 0` is legal: the constructor validates `c1`, `c2` and `c3` with
`assert_nonempty_nonneg_finite_val`, and asserts only `c2 < c3`.

**This family has two gates, and only one of them reads a centred quantity.** The **indecision**
zone gates on `c2` over the centred, standardised return, and it is the analogue of the Gerber band
edge. The **confusion** zone gates on `c1` over the **raw, uncentred** return. At `c1 = 0` that
gate drops nothing either, so a raw return of exactly zero survives it. Whether the rule binds
there as well is the question #499 put to this ADR.

### The two readings

Ticket #491 named both. Tickets #498 and #499 named only the second one, and for the same reason:
a threshold that a caller may be passing today should keep a clean meaning rather than be
rejected.

 1. **Tighten the guard to `0 < t`.** The Gerber statistic is defined for a strictly positive
    threshold, and Riskfolio-Lib asserts `0 < threshold < 1`. `val_dict[:t]` already carries
    `0 < t < 1`, so the neighbouring wording exists.
 2. **Make the bands disjoint at a zero edge.** An exactly zero return becomes neutral rather than
    both up and down. `t = 0` stays legal and becomes the sign concordance, which is what a reader
    expects of a zero threshold.

## Decision

**A return of exactly zero never crosses a Gerber threshold, whatever the threshold is.** The rule
holds for the whole Gerber family, and each member states it in the shape its own gate takes.

**`GerberCovariance` adds a sign test to each band.** `gerber_updown` becomes:

```julia
zx = zero(eltype(X))
U .= (X .>= ts) .& (X .> zx)
D .= (X .<= -ts) .& (X .< zx)
```

**`GerberIQCovariance` gets one crossing predicate, and `comovement_step` uses it for both
axes.** The three sites that repeated the threshold comparison now read the same answer:

```julia
@inline function iq_crossed(x::Number, ax::Number, c::Number)
    return ax >= c && !iszero(x)
end
```

**`SmythBrobyCovariance` gets the same predicate on its indecision zone.** `comovement_step`
repeated the `c2` comparison in three places, and all three now read one answer per axis:

```julia
@inline function sb_crossed(r::Number, ar::Number, c::Number)
    return ar >= c && !iszero(r)
end
```

**The Smyth-Broby confusion zone does not take the sign test.** The rule is about a **centred**
quantity. A centred return of exactly zero is an asset that did not move away from its own mean,
so it has no sign, and a gate that lets it through hands the classification an observation it
cannot classify. The confusion zone reads the **raw, uncentred** return, whose zero is an
arbitrary point of the scale of the data: an asset whose raw return is zero moved by `-mu` against
its mean, which is a deviation that does carry a sign. Two further facts settle it. That gate only
*rejects* and never classifies, so on its own it produces no wrong count. And at `c1 = 0` it means
"no confusion zone", which is the meaning a reader expects of a zero threshold; a sign test would
turn it into "reject the observations on which both raw returns are exactly zero", which is a rule
about the origin of the data and not about the statistic.

**The sign test binds only at a zero edge.** `ce.t` and `sd` are both non-negative, so `ts` is
non-negative; `ce.c` and the scaler's factor are both non-negative, so `st.ci` is non-negative; and
`ce.c2` is non-negative by its own guard. For a positive edge the test is redundant, because
`x >= ts > 0` already implies `x > 0`, and `ax >= c > 0` already implies that `x` is not zero.
Every result at a positive threshold is therefore unchanged, bit for bit.

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
- **The Gerber IQ diagonal is unit at every threshold.** The pair `(i, i)` crosses on both axes or
  on neither, so it never reaches the neutral accumulator. `Gerber1` therefore answers
  `pos / pos` on the diagonal, as `Gerber0` and `Gerber2` already did.
- **The reduction to `GerberCovariance` holds at `c = 0`.** Gerber IQ with every weight set to
  one, no decay and the per-asset volatility scaling reproduces the Gerber statistic at a zero
  threshold as it does at a positive one. Before this decision the docstring of `gerber_IQ` had to
  state that the reduction needed a positive threshold, and that sentence is gone.
- **The Gerber IQ classification is exhaustive.** With `iq_crossed` on both axes, an observation
  on which both assets crossed carries a product that is not zero, so it is concordant or
  discordant and never neutral. The neutral branch is exactly the case its own docstring names:
  one asset crossed and the other did not.
- **`test_08_moments.jl` holds the contract.** The testset `Gerber statistic (#454)` pins that the
  bands of the sample at `t = 0` are the bands at `t = 0.5`, that the exactly zero returns are in
  neither band, and that all three variants answer a unit diagonal. The testset
  `Gerber IQ zero threshold (#498)` pins the unit diagonal of all three Gerber IQ branches at
  `c = 0`, the reduction to `GerberCovariance` at that threshold, and that no result at a positive
  threshold moves. The testset `Smyth-Broby zero indecision zone (#499)` does the same for that
  family and adds the two gates: that `c2 = 0` no longer inflates the neutral count, and that
  `c1 = 0` still admits every observation.
- **The Smyth-Broby diagonal is unit at every `c2`.** The pair `(i, i)` crosses on both axes or on
  neither, so it never reaches the neutral accumulator, and the three `*1` markers answer
  `pos / pos` on the diagonal as the `*0` and `*2` markers already did.
- **The Smyth-Broby classification is exhaustive.** Two crossings give a product that is not zero,
  so an observation on which both assets crossed is concordant or discordant and never neutral.
  The neutral branch is exactly the case its own docstring names.
- **A zero confusion threshold still admits every observation.** `c1 = 0` keeps its meaning of
  "no confusion zone". The behaviour change of this decision in the Smyth-Broby family is confined
  to a caller who passes `c2 = 0` over data that carry an exactly zero centred return.
- **The rule now covers every member of the Gerber lineage.** `GerberCovariance`,
  `GerberIQCovariance` and `SmythBrobyCovariance` each state it in the shape of their own gate, so
  a new member of the lineage inherits a rule that is already written down.
