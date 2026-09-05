---
status: accepted
---

# The library writes one bias-correction default, and does not track the upstream one

## Context

[#444](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/444), a child of
[#415](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/415), opened on a raise.
`StdValue`, `VarValue` and `SimpleVariance` each carry `corrected::Bool = true`. Under a plain
`StatsBase.Weights` that combination raises an `ArgumentError` rather than returning a value,
because that weight type declares no bias correction. The raise fires at the reduction, not at the
constructor.

[`factory`](../../src/02_Tools.jl) reaches the same state with no measure in the caller's hand. It
replaces the `w` field through the `@wprop` channel and leaves `corrected` alone, so a weighted
`factory` call on a default `StandardisedValue` raises.

The ticket asked whether the library should refuse the combination, and where. The work found a
prior question: **which default the library holds** must be settled first, because every candidate
fix rests on it.

### The proposal that was tested

Adopt the upstream default rather than hold one. The field becomes `Option{Bool}` and it is
`nothing`; the reduction writes no `corrected` keyword, so `Statistics` and `StatsBase` each apply
their own. A release of theirs then moves our numbers with no edit of ours, and the library carries
no copy of a value it does not own.

That proposal was implemented and measured. It does not survive the measurement.

### Upstream is not self-consistent

Measured against `StatsBase` v0.34.12 and Julia 1.12, on `X = [1 2; 3 4; 5 7; 2 1]` and its first
column `v`, with `a = AnalyticWeights([0.1, 0.2, 0.3, 0.4])`:

| Call | Upstream default | Value |
| --- | --- | --- |
| `Statistics.var(v)` | `corrected = true` | `2.91667` |
| `Statistics.cov(X)[1, 1]` | `corrected = true` | `2.91667` |
| `StatsBase.cov(SimpleCovariance(), X)[1, 1]` | `corrected = false` | `2.1875` |
| `Statistics.var(v, a)` | `corrected = nothing` | `2.0` |
| `StatsBase.cov(SimpleCovariance(), X, a)[1, 1]` | `corrected = false` | `2.0` |

`Statistics.var` and `Statistics.cov` agree on the unweighted branch. `StatsBase.SimpleCovariance`
disagrees with both, over the same data. `StatsBase` resolves its weighted `nothing` through
`depcheck`, which returns `false`, while its deprecation message states that the default becomes
`true` in the future. That is the exact value which raises under a plain `Weights`.

A library that writes no `corrected` therefore answers `2.91667` for a variance and `2.1875` for a
covariance over the same unweighted data. **Tracking upstream costs internal consistency**, because
there is no single upstream default to track.

### What the written value already buys

[`GeneralCovariance`](../../src/08_Moments/03_Covariance.jl) writes
`StatsBase.SimpleCovariance(; corrected = true)`. That written `true` is what makes the covariance
path agree with the variance path, and it is a deliberate departure from `SimpleCovariance`'s own
`false`. The library is consistent on both branches today. The sentinel proposal broke the weighted
branch: a weighted `SimpleVariance` fell to `2.0` while a weighted `GeneralCovariance` stayed at
`2.85714`.

## Decision

**The library writes one bias-correction default, and it is `corrected = true`.** Every site that
takes the field declares it, and no site defers to a callee's default. The value is written, not
inherited, so a change upstream does not move it.

**No constructor refuses `corrected = true` beside a plain `StatsBase.Weights`.** `StatsBase`
raises at the reduction with a message that names the three weight types that work, and it raises
on the variance path and the covariance path alike. A guard of ours would refuse the combination
the day `StatsBase` gives `Weights` a correction. It could not see the `factory` path in any case,
because `factory` writes `w` after the constructor runs.

**The raise is documented where a reader meets it.**
[#440](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/440) states the four
denominators and the raise in the `# Mathematical definition` and `# Details` sections of
`VarValue` and `StdValue`, and in the `# Details` of `StandardisedValue`.
`src/08_Moments/04_SimpleVariance.jl` states it beside its weighted formula.

## Consequences

- **A plain `StatsBase.Weights` raises under the default, on every path.** This is the behaviour
  that ships. It is consistent rather than convenient. The caller passes an `AnalyticWeights`, a
  `FrequencyWeights` or a `ProbabilityWeights`, or sets `corrected = false`.
- **The `factory` path raises too.** `factory` replaces `w` and leaves `corrected` at `true`, so a
  weighted `factory` call on a default measure reaches the raise. Nothing between the call and the
  reduction can see it, and the documentation is what warns the reader.
- **A `StatsBase` release cannot move our numbers through this field.** The value is ours.
- **The written `corrected = true` on `StatsBase.SimpleCovariance` is load-bearing.** It is not
  redundant, and removing it would split the covariance path from the variance path. This ADR is
  the reason it is written.
- **`test_05_tools.jl` holds the consistency.** It pins the raise on both paths, and it pins that a
  weighted variance and a weighted covariance answer the same number under the same weights.
