---
status: accepted
---

# A histogram edge is widened by the spacing at the value, not by the epsilon of the type

## Context

[#493](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/493), a child of
[#417](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/417), opened on a raise from the
sweep of [#459](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/459).
[`calc_hist_data`](../../src/08_Moments/10_Histogram.jl) widened both histogram edges by
`eps(eltype(x))`:

```julia
xjl = minimum(xj) - eps(eltype(xj))
xjh = maximum(xj) + eps(eltype(xj))
```

`eps(eltype(x))` is the machine epsilon of the **type**, `2.220446049250313e-16` for a `Float64`.
The spacing at a **value** of magnitude `m` is `eps(m)`, which is `4.44e-16` at `m = 3.5`. So the
widening is half of one unit in the last place there, and `m + eps(Float64)` rounds to even: it
returns `m` itself about half the time. A `StatsBase.Histogram` bin is closed on the left, so the
upper edge is exclusive, and the largest observation is then binned **out**.

Over `randn(StableRNG(555), 600, 6)`, three of the six columns lose their maximum. On the pair
`(2, 5)` at eight bins:

| quantity | as shipped | with the fix |
| --- | ---: | ---: |
| joint histogram total | 598 | 600 |
| `H(X_2)` | 1.6775377064421677 | 1.683319735731528 |
| `H(X_5)` | 1.5499741632686022 | 1.5597176994273774 |
| `I(X_2; X_5)` | 0.03695594625897657 | 0.03969760755165243 |

The mutual information rises by 7.4 %.

The table on the ticket itself reports different numbers for the same seed and the same pair, and
both are correct. The ticket measured that matrix with two of its columns made correlated, and
pasted a snippet that omits the lines building them. The numbers above are the plain sample, and
they are what `test_08_moments.jl` pins.

### A constant column gives `NaN`

The same rounding closes both edges onto one value when the data is constant. A column of `3.5`
gives an all-zero histogram, `sum(hx)` is zero, and `hx / sum(hx)` is a vector of `NaN`:

```julia
c = fill(3.5, 20)
PortfolioOptimisers.calc_hist_data(c, c, 8)[1]  # NaN
```

The entropy of a constant variable is zero, not undefined.

### The severity scales the wrong way

`eps(Float64)` is many units in the last place near zero, so the widening works on returns data. It
fails at a magnitude of two or more — on prices, on levels and on any standardised series, which is
exactly the data a caller is most likely to hand a mutual-information estimator directly.

### What moves, measured

The lower widening is harmless, because the first bin is closed on the left. Removing it moves every
bin edge by less than one unit in the last place, so a count changes only where an observation sits
within that distance of an edge. The measurement below counts the pairs whose `calc_hist_data`
output is not bit-for-bit identical, over eight bin rules and every pair of a data set:

| data | pairs that move | pairs |
| --- | ---: | ---: |
| the test suite's SP500 and Factors returns | **0** | 1680 |
| the same returns, standardised | 1320 | 1680 |
| `randn(StableRNG(555), 600, 6)` | 120 | 168 |

The library's own pinned numbers do not move at all. Every raw return of the test data is below
`0.15` in magnitude, where `eps(Float64)` is hundreds of units in the last place and the shipped
widening does what it was written to do.

## Decision

**A histogram edge is widened by the spacing at the value.** `calc_hist_data` takes the minimum
itself as the lower edge, and `nextfloat` of the maximum as the upper one:

```julia
xjl = minimum(xj)
xjh = nextfloat(maximum(xj))

xil = minimum(xi)
xih = nextfloat(maximum(xi))
```

**The lower edge is not widened.** A `StatsBase.Histogram` bin is `[a, b)`, so the first bin already
holds an observation that sits exactly on the lower edge. A widening there buys nothing and moves
every other edge.

**`nextfloat` is the widening, not `maximum(x) + eps(maximum(x))`.** The two agree over most of the
range, and `nextfloat` is exactly one unit in the last place at every magnitude, including a binade
boundary and a subnormal.

**`range(xjl, xjh; length = bins + 1)` keeps the upper edge exact.** Julia's `range` reproduces both
endpoints, so the last edge is `nextfloat(maximum(x))` bit for bit, and the maximum is strictly
below it.

## Consequences

- **No pinned number of the library moves.** `test/assets/covariance.csv.gz` holds the six
  `MutualInfoCovariance` columns, and every one of them is unchanged. The measurement above is the
  reason: the test data is returns, and the shipped widening already worked there.
- **A caller who hands standardised data to a mutual-information estimator gets a different number
  than before.** That is the defect this ADR fixes, and it is the whole size of the behaviour
  change. The direction is one way: an observation that was dropped is now counted, so an entropy
  can only rise or stay.
- **A constant column now has an entropy of zero rather than `NaN`.** The change reaches
  [`mutual_info`](../../src/08_Moments/10_Histogram.jl),
  [`variation_info`](../../src/08_Moments/10_Histogram.jl) and `mutual_variation_info`, and through
  `variation_info` it reaches [`VariationInfoDistance`](../../src/09_Distance/02_Distance.jl) and
  the clustering and hierarchical optimisers below it.
- **The fix does not make the estimate correct, only unbiased at the edge.** A histogram estimate of
  mutual information stays biased by its bin count, which is what
  [`HacineGharbiRavier`](../../src/08_Moments/10_Histogram.jl) and the three width rules address.
- **`test_08_moments.jl` holds the contract.** It pins that the maximum of a column is binned in at
  a magnitude where the shipped widening rounded away, that a constant column has a finite entropy,
  and that the returns of the test data answer the same numbers as before.
