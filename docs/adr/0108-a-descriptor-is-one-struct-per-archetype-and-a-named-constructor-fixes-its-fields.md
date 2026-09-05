---
status: accepted
---

# A descriptor is one struct per computational archetype, and a named constructor fixes its fields

## Context

The cross-sectional factor prior of map
[#643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/643) builds its Factor
Exposures from Descriptors: a per-asset value at every observation, computed from the Panel Fields
of an Asset Panel. The reference implementation ships **45 descriptor classes**. Forty-one of them
are named finance descriptors, a book-to-price, a return on equity, a sales growth rate; three are
generic growth bases; one passes a field through.

[Issue #657](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/657) censused them and
found that the 45 classes compute in about sixteen ways. Twenty-two of them are one field divided by
another at the same observation. One is a logarithm. Five are a lagged comparison of a field with
itself or with a scale. The rest are exponentially weighted or rolling statistics of returns and
volumes, in six shapes. A named class of the reference differs from its siblings only in the field
names it reads and in the defaults it fixes.

The map's governing rule, stated by the maintainer on
[#648](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/648), is that a decision must
reproduce the reference implementation, and may only **add** capability or **simplify** the design.
[Issue #650](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/650) decided the shape
under that rule with the maintainer in the loop, and asked the first build to write this record.

Two facts about the library shaped the choice.

 1. **The library already spells a variant as a slot, not as a type.** `Covariance` takes a
    moment Algorithm, `FullMoment` or `SemiMoment`, and there is no `SemiCovariance` struct. A
    downside volatility in the reference is a class; here it is the same slot on the same struct.
 2. **A struct is configuration, and a Descriptor Estimator holds no data.** Every parameter of a
    reference class is a field name or a numeric default. Nothing in a named class is a
    computation the archetype does not already do.

## Decision

### One struct per computational archetype

Each way of computing is one `@concrete struct` whose fields are the Panel Field names it reads
and the numeric parameters of the computation. The first three ship in this decision:

| Archetype | Fields | Reference classes it covers |
| --- | --- | --- |
| `PanelFieldRatio` | `num`, `den`, `nonneg`, `pos` | the 22 point-in-time ratios |
| `PanelFieldLog` | `field` | `LogMarketCap` |
| `Passthrough` | `field` | `Passthrough` |
| `GrowthRate` | `field`, `lag` | `GrowthRate` and three named growth rates |
| `ChangeToScale` | `field`, `scale`, `lag` | `ChangeToScale` and `EarningsChangeToPrice` |
| `ChangeInIntensity` | `field`, `scale`, `lag` | `ChangeInIntensity` and `CapexToAssetsChangeInIntensity` |

[#718](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/718) added the four
exponentially weighted archetypes under the same rule, and its downside variant is the
`FullMoment` / `SemiMoment` slot of fact 1:

| Archetype | Fields | Reference classes it covers |
| --- | --- | --- |
| `EWMean` | `decay`, `min_obs`, `skip`, `exponentiate` | `EWMomentum` |
| `EWVolumeRatio` | `num`, `den`, `decay`, `min_obs` | `EWShareTurnover` and `EWAmihudIlliquidity` |
| `DaysToCover` | `num`, `den`, `decay`, `min_obs` | `DaysToCover` |
| `EWVolatility` | `ce`, `alg`, `mar` | `EWVolatility` and `EWDownsideVolatility` |

[#720](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/720) added the two rolling
archetypes, where the sign is a field rather than two structs:

| Archetype | Fields | Reference classes it covers |
| --- | --- | --- |
| `RollingLogReturn` | `window`, `skip`, `sign`, `exponentiate` | `RollingMomentum` and `Reversal` |
| `RollingMax` | `window` | `MaxReturn` |

The exponentially weighted beta family follows in
[#719](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/719).

An exponentially weighted archetype states a `decay` and a `min_obs`, as
`RegimeAdjustedExpWeightedVariance` spells them, and every named constructor takes a `half_life`
instead and converts it. Two archetypes carry no named constructor of their own, because their
reference class is the archetype: a named constructor there would be a second keyword method of
one function, which Julia cannot dispatch. Each takes `half_life` as a keyword that fixes the
defaults of the fields it converts to.

A numerator or a denominator is one Panel Field name, or a vector of `name => coefficient` pairs
read as their sum. That is how a gross profit, `sales - cost of revenue`, or a total capital,
`debt + equity`, enters a ratio without a struct of its own.

### Every named descriptor is an exported constructor function

`BookToPrice()` returns `PanelFieldRatio(; num = "book_equity", den = "market_cap")`.
`SalesGrowthRate()` returns `GrowthRate(; field = "sales_ttm", lag = 252)`. Each named constructor
accepts a keyword that renames every field it fixes, so a panel that spells its fields differently
needs no rename, and a ratio over any two fields is one call away. The named constructors are
exported; the two abstract roots are not.

`typeof(BookToPrice())` is `PanelFieldRatio`, and `BookToPrice()` prints as one. A named
descriptor has no identity of its own beyond the fields it fixes, which is the point.

### The guards are fields, and a non-positive denominator is `NaN`

The reference implementation checks its inputs two ways. Some ratios raise on a non-positive
denominator; others return `NaN` there. Some raise on a negative numerator. The port keeps both
kinds of guard as **fields of the archetype**, `nonneg` and `pos`, so a named constructor states
the census row's guard and a caller can turn it on for a ratio of their own.

The port makes one deliberate change: **every ratio answers `NaN` where its denominator is not
strictly positive**, through `positive_divide`. The reference raises on some of them. A refusal is
not a mode a caller can use, and a `NaN` in one cell costs one cell of the Descriptor where a
raise costs the fit, so the change removes no capability and adds one. The `nonneg` guard stays a
refusal, as `GrowthRate`'s refusal of a negative field does: a negative dividend or a negative
sales figure is a data error the caller must see.

### A blank reads back as `NaN`

The library resolves every blank before it reaches a carrier
([#664](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/664)), and records the
resolution in an observed-mask column. Every Descriptor reads through `panel_field_values`, which
undoes the resolution, so a fill value never enters a ratio as data. The reference implementation
keeps its blanks as `NaN` inside the field, so its descriptors need no such step; the port needs
it because of a decision the reference did not make.

## Alternatives refused

**One struct per reference class, 45 structs.** It reproduces the reference one to one and it was
the first shape considered. It was refused because 22 of the structs would carry an identical
`descriptor` method, and a reader of the reference would meet 45 types where sixteen computations
exist. A field-name change to the panel would then be 22 edits. The rule permits a simplification,
and this is one.

**A single `Descriptor(formula)` type over a small expression language.** It reduces the structs
to one. It was refused because a Descriptor that computes an exponentially weighted statistic over
returns has state the expression cannot express, so the language would grow into an interpreter,
and because the type of the estimator would no longer say what it computes.

**Named structs as subtypes of the archetype.** `struct BookToPrice <: PanelFieldRatio` is not
possible: a concrete type has no subtypes. A `BookToPrice` wrapper that forwards to a
`PanelFieldRatio` reintroduces the 45 types.

## Consequences

- A named descriptor is a function, so `typeof` names the archetype and not the descriptor. A
    reader who wants the name must read the fields. Every exported function costs a Capability
    Catalogue entry, and this decision costs 28 of them plus the six archetypes.
- A consumer that dispatches on a descriptor dispatches on the archetype. Nothing in the
    library dispatches on a named descriptor, and the composition ticket
    ([#650](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/650) §3) decided that a
    descriptor carries no name inside a Factor Exposure.
- A new named descriptor of an existing archetype is one exported function and one catalogue
    entry, with no new struct and no new `descriptor` method.
- The `nonneg` and `pos` guards must name a field the ratio reads, and the constructor refuses
    one that does not, so a typo in a guard cannot pass in silence.
- Every Descriptor pays one read of its observed-mask column per field. A field that cannot
    blank carries no such column and pays nothing.
