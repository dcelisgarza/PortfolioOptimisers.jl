---
status: accepted
---

# A type is kept by default, and `nothing` spells a skipped procedure

## Context

The map [#1852](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1852) audits the
library for types that it can simplify: wrappers, near duplicates, nested structs that can flatten,
and types that hold no data. Each candidate gets a verdict ticket. The verdicts need one rule to
cite, or each ticket argues the question again
([#1853](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1853)).

One question came back many times: how does a field say "off"? The library has three spellings.

- `nothing`, as in `hac_lags = nothing` or `MaximumRatio(; ohf = nothing)`.
- A singleton type, as in `RawStatistic()`, `NoSlack()` or `IdentityScaling()`.
- A `Bool` or a `Symbol`, as in `PricesToReturns(; padding = false)`.

[#1867](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1867) found a struct with two
spellings. `RegimeAdjustedExpWeightedVariance` turned `regime_method` off with `nothing` and
`hac_floor` off with `NoHacFloor()`. The research of
[#1858](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1858) found more cases of this
kind, and the maintainer ruled on each one.

## Decision

### Keep a type by default

Keep a struct or an abstract type by default. Simplify a candidate only when all four checks fail:

1. No method of the library dispatches on the type.
2. No second site reuses it.
3. It gives no measured runtime or compile benefit, such as specialisation, type stability or a
   zero-cost singleton.
4. No ADR names it deliberate.

The maintainer then rules on each candidate. Nested structs, near duplicates and types that hold no
data are often features of the library. They give reuse and dispatch at no extra cost.

### Spell "off" with `nothing` only where a procedure is skipped

1. **A descriptive singleton stays.** A singleton that names what the field does, such as
   `IdentityScaling()`, `RawStatistic()` or `PlainGradient()`, is kept, also when its method is
   the identity. A singleton is a dispatch point, so a caller can overload it, and its name tells
   the reader what happens.
2. **`nothing` is correct only where a procedure is skipped, or for a simple default.** A skipped
   procedure is `hac_floor = nothing`, which applies no floor, or `peel = nothing`, which peels no
   asset. A simple default is `MaximumRatio(; ohf = nothing, kmin = nothing)`, where `nothing`
   sizes the value from the resolved characteristic.
3. **A `Bool` or a `Symbol` is acceptable**, mainly where it reads better than `nothing`. ADR 0044
   still prefers a named type where a flag selects between two sources.

A singleton that only skips a procedure, and that names no rule of its own, becomes `nothing`. The
work of #1867 removed two: `NoHacFloor` (ADR 0190) and `NoPeel` (ADR 0117).

### Cases ruled under this decision

| Field | Spelling | Why |
| --- | --- | --- |
| `hac_floor` of the regime-adjusted estimators | `nothing` | It skips the floor. `NoHacFloor` is removed. |
| `CoveragePolicy.peel` | `nothing` | It skips the peel. `NoPeel` is removed. |
| `debias = RawStatistic()` | singleton | One of three debias rules. `debiases` and `reads_dependence` dispatch on it. |
| `intensity = UnclampedIntensity()` | singleton | One of three clamp rules. `intensity = nothing` would read as "no shrinkage". |
| `PassiveAggressiveMeanReversion.slack = NoSlack()` | singleton | PAMR, a variant of the source with its own step formula. |
| `MirrorDescent.grad = PlainGradient()` | singleton | Descriptive. `AdaptiveOptimistic` checks for it. |
| `NumericPanelInput.alg = NoPanelFill()` | singleton | A policy that refuses a blank cell, with an error that names the Panel Field. |
| `OrthogonalUncertaintySet.scaling = IdentityScaling()` | singleton | Descriptive, and it matches `IdentityMetric()` in the same struct. |
| `ForecastReversion.scale = nothing` | `nothing` | It skips the preconditioning. |
| `PricesToReturns.gap_return_alg = nothing` | `nothing` | The simple default rule. |
| `PricesToReturns.ret_method`, `padding` | `Symbol`, `Bool` | Acceptable as they are. |
| `mtx_sqrt = nothing` | `nothing`, beside the new `PlainCholeskySquareRoot()` | The simple default. Its siblings are named leaves, so the plain Cholesky factor gets a name too ([#1878](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1878)). |
| `GerberIQCovariance.sc = nothing` | `nothing`, beside the new `PairMeanGerberIQScaler()` | The simple default, named for the same reason. The method on `nothing` and the fallback of the family are separate, and the fallback refuses a subtype with no method of its own (#1878). |

The scan of #1867 also kept `IdentityMetric`, `RawVolatility`, `NoSystematicRepair`, `NoDecay`,
`IdentityTarget` and `NoDefault`. Each one has its own behaviour or a descriptive name.
`NoDefault` must stay a type, because `nothing` is already a valid default of a time-dependent field.

## Considered options

- **Simplify by default.** Remove every type that a smaller form can replace. This was rejected.
  It costs dispatch points that callers overload, and specialisation that a singleton gives at no
  cost. The work to undo a removal is larger than the cost to keep the type.
- **`nothing` for every "off" value.** This was rejected. `intensity = nothing` and
  `tgt = nothing` would read as "no shrinkage", which is wrong. The name of a descriptive
  singleton tells the reader more than `nothing` does.
- **A singleton for every "off" value.** This was rejected. `nothing` is the spelling of an
  optional value in the whole library. A singleton that only skips a procedure adds a public name
  that tells the reader nothing more.

## Consequences

- A verdict ticket of #1852 cites this ADR and records the four checks.
- A new field that can skip a procedure takes `Option{<:AbstractX}` with a method on `Nothing`.
- #1867 ruled the `hac_floor` case on 2026-10-27, before this ADR existed. Its closing comment
  records the ruling, not the four checks.
- Related: ADR 0044 prefers a named type over a flag. ADR 0154 states when an abstract type is
  `public`.
