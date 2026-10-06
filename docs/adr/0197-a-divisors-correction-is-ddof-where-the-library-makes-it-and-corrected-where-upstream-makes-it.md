---
status: accepted
---

# A divisor's correction is `ddof` where the library makes it and `corrected` where upstream makes it

## Context

The divisor of a spread is the number of observations `T`, less a correction. The library names
that correction with two keywords.

- `ddof::Integer` is taken by `L1Norm`, `L2Norm`, `SquaredL2Norm` and `LpNorm`, by `EvenMoment`, and
  by the aliases that build an `EvenMoment`. The tracking error and the tracking risk measure reach
  it through the `NormError` they hold.
- `corrected::Bool` is taken by `SimpleVariance`, `StdValue`, `VarValue` and the
  `StatsBase.SimpleCovariance` inside `GeneralCovariance`.

[#1525](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1525) asked for one keyword,
because two names for one choice cost a reader a lookup at every site. It offered two options:
`corrected` everywhere, or `ddof` everywhere. The two keywords do not name one choice, though.

- At every `ddof` site the library makes the correction itself: `norm_factor` returns
  `sqrt(T - ddof)`, `T - ddof` or `(T - ddof)^(1/p)`, and the functor of `EvenMoment` divides by
  `length(val) - ddof`. Any non-negative integer has a meaning there.
- At every `corrected` site `Statistics` or `StatsBase` makes it. Without weights the flag selects
  `T - 1` or `T`. Under `FrequencyWeights` it selects `sum(w) - 1`, under `AnalyticWeights`
  `sum(w) - sum(w .^ 2) / sum(w)`, and under `ProbabilityWeights` a factor `n / ((n - 1) sum(w))`.
  No integer `ddof` states the last two.

A rename to `corrected` would take the integer away from sites that compute their own divisor. A
rename to `ddof` would make every weighted site refuse an integer outside `{0, 1}`, or invent a
meaning for it under each weight type.

The first plan for the weight spread of the realised factor attribution
([#1515](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1515), row R95 of
[#1416](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1416)) mixed the two. It took
`ddof::Integer = 1` and computed the spread with `std(W; dims = 1, corrected = ...)`. Such a site
states an integer and applies a flag, so `ddof = 2` either raises or silently means `ddof = 1`.

## Decision

**The keyword names who makes the correction.**

1. A site where the library subtracts the correction from the number of observations itself takes
   `ddof::Integer`, and refuses a negative value.
2. A site that passes the correction to `Statistics` or `StatsBase` takes `corrected::Bool`, which
   is the keyword those packages define.
3. One definition never names both. A definition that names `ddof` never calls an upstream spread
   that applies a correction of its own.

A partial fit that reproduces a correction `StatsBase` makes on the batch path subtracts the flag
itself, as `n - corrected`. It keeps `corrected`, because it reads the field of an estimator whose
batch path delegates, and one field cannot carry two names.

The defaults do not change. Each is the decision of its site: ADR 0020 fixes the defaults of the
`NormError` members, and ADR 0087 fixes `corrected = true`.

`.github/instructions/julia-source-code.instructions.md` § *A divisor's correction is named by who
makes it* states the rule. `test/test_77_correction_keyword_census.jl` gates its third point: it
parses `src/` and `ext/`, and it fails on a definition that names `ddof` and also names `corrected`
or calls `std`, `var`, `cov`, `cor`, `stdm`, `varm`, `varcorrection` or a `mean_and_*` spread.

## Consequences

- **No public keyword changes.** Every site already followed the rule when it was written, so no
  caller sees a rename and no deprecation is needed.
- **A reader can tell from the keyword who makes the correction.** `ddof` tells the reader to look
  for `T - ddof` in this library. `corrected` tells the reader to read the divisor in the manual of
  `Statistics` or `StatsBase`.
- **The weight spread of #1515 keeps `ddof`, and computes its own divisor.** It subtracts `ddof`
  from the number of observations and takes no upstream `std`. The census refuses the first plan.
- **A new spread picks its keyword from its implementation.** A site that later moves from its own
  divisor to an upstream call renames its keyword in the same change, and the census refuses the
  change that does not.
- **The census reads names, not meaning.** It does not see a `ddof` that reaches an upstream call
  through a helper in another definition. Review holds that case.
