---
status: accepted
---

# The asset table states the clock, and the layer carries every absence and names it

## Context

[ADR 0133](0133-the-conversion-computes-a-return-and-ingestion-is-the-only-door.md) makes ingestion
the only door, and keeps the friendliest call in the library a one-liner by routing it through the
layer:

```julia
rd = prices_to_returns(X)          # X::TimeArray

# is exactly
pr = price_ingestion(PriceIngestion(), X)
rd = prices_to_returns(pr)
```

That moves `PriceIngestion()`'s defaults from *what a caller who reached for the layer asked for* to
*what every caller gets by not asking*. ADR 0133 named this as something it does not settle.

Four measurements frame the decision. Each was taken on `dev` at `e3b2b60ed2`.

1. **A benchmark with more history than the assets makes the implied volatilities unusable.** A
   five-day asset panel, a five-day `iv` on the asset clock and a fifteen-day benchmark refuse the
   ingestion outright — `IsEmptyError: ... 10 of the 15 emitted observations are absent from iv`.
   Nothing is wrong with the data. A symmetric join moved the observation clock to the benchmark's,
   and `iv`, aligned to the assets as an implied volatility series naturally is, then failed the
   subset check. ADR 0133 recorded the clock move on the asset axis — a 253-row asset slice against
   the full 8,313-row index gives 8,312 observations and 161,200 non-finite asset cells — and left
   the default to this decision.

2. **The two spellings of an absent implied volatility are treated oppositely, and the wrong one is
   refused.** `missing` passes, because a `Union{Missing, Float64}` array is not `ArrNum` and the
   numeric branch never fires, so it reaches the carrier unvalidated. `Inf` passes, being
   non-negative. `NaN` — the layer's own spelling for absence — throws, and the message blames
   non-negativity: `DomainError: all(x -> 0 <= x, iv) must hold`. `iv` is the one series the layer
   does not put through `unify_gaps`.

3. **A number type the panel chose is discarded.** `unify_gaps` rebuilds a non-float series as
   `[ismissing(x) ? NaN : Float64(x) for x in v]`, so an integer panel returns `Float64`. The
   coercion census in `test/test_55_numeric_coercion_census.jl` matches `float(` and `Int(ceil(`,
   the two spellings that had spread, and does not match `Float64(`.

4. **Two tables of different float widths cannot be joined at all.** `Float32` assets beside a
   `Float64` factor table raise `MethodError: no method matching merge(::TimeArray{Float32,...},
   ::TimeArray{Float64,...})`. `TimeSeries.merge` requires one shared value type, and each series is
   widened independently, so nothing reconciles them. A `Float32` panel with no second table
   survives as `Float32`, so the layer is open to the width right up to the point a covariate
   arrives.

## Decision

### The asset table states the clock

`join_method` is a field of `PriceIngestion`, and its default is `:left`. The asset table states the
universe, so it states the clock; the factor, benchmark and implied-volatility series are aligned to
it, padded where they are silent. This is what `price_ingestion`'s `# Algorithm` already claims —
*join the factor and the benchmark series onto the asset clock* — so the claim becomes true rather
than being retracted.

The rule that keeps the field is ADR 0133's, read for the layer rather than the conversion: **a
keyword survives on the ingestion layer if and only if it states something about the sources the
layer cannot derive.** Which clock is authoritative is a caller intent the data does not disclose.
The layer has a right default for it, not a derivation, so `:outer` and `:inner` remain reachable.

### The value type is derived from the series, and a type that cannot hold an absence is refused

The unification target is read off the tables: promote the non-missing element types of the asset,
factor and benchmark series, and widen only where there is an absence to spell. A `Float32` panel
stays `Float32`. `Float32` beside `Float64` promotes and joins. An integer panel takes the float
type that represents it, derived rather than named.

A type that cannot carry the layer's absence — a `Rational` panel that holds a gap — is refused by
name. The library must stay open to number types it has never seen, and one absence it cannot spell
is a refusal, not a reason to close the layer to the rest.

### An absence is carried on every axis the layer touches

The implied volatilities are a carried series like the factors and the benchmark: `unify_gaps` gives
them the one spelling, the join aligns them to the emitted clock, and a silent `iv` pads `NaN`. The
requirement that `iv` cover the emitted clock is dropped, and `PricesResult`'s guard on `iv` becomes
non-negative where a value is present.

The estimator that reads them defends itself, where the carrier no longer does.
`Statistics.cov(ce::ImpliedVolatility, …)` and its `cor` twin intersect the coverage mask with the
columns whose implied volatilities are complete over the window. Reduce-and-expand then does what it
already does for an asset with absent returns: the column is excluded from the fit, and
`expand_moment` writes its `NaN` row and column. This needs no vocabulary on the implied-volatility
axis, and it reaches the hand-built keyword path — `cov(ce, X; iv = …)` validates only shape — which
no carrier boundary ever protected.

### What the layer padded is named

`strict` is a field of `PriceIngestion`, defaulting to `false`. The layer reports what it padded —
which table, how many observations, which columns — warning by default and refusing under `strict`.
This is the shape a Held Gap is named with ([ADR 0118](0118-a-fold-zeroes-a-held-gap-once-and-a-value-level-verb-reduces-to-the-investable-mask.md))
and the shape `PriceGapFill` reports with
([ADR 0130](0130-a-universe-policy-is-fitted-and-the-only-fill-is-a-span-bounded-price-convention.md)).

A carried absence is invisible where a refused one was not, and the axes it lands on carry no
universe. A caller running a walk-forward sets `strict` and is then certain no covariate was
invented.

### The span is a stated value, and the Span Rule is the only derivation

`span` is `Option{<:AbstractMatrix{Bool}}`, defaulting to `nothing`, which derives the **Listing
Span** by the **Span Rule**. It is an override, not a switch: no value means *emit no span*, because
[ADR 0132](0132-the-layer-emits-one-carrier-and-alignment-splits-into-a-fixed-axis-and-a-provenance-check.md)
has the layer always emit a panel and `pnl === nothing` already means *this carrier was not built by
the layer*. [ADR 0129](0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md)
is unchanged: the bound is a matrix of booleans, so a caller's declaration enters at the same point
and no rule family is minted.

### The frequency is the source's

`collapse_args` is a `Tuple`, defaulting to `()`, which leaves the clock alone. Collapsing states
what frequency an analysis wants, and the source clock is the only frequency the layer knows. After
`:left`, this is the only field that can move the emitted clock.

### There is no second spelling of absence, and no second way to configure

`padvalue` is not a field. The layer has one spelling for an absent price and a second would
contradict `unify_gaps`.

`prices_to_returns(X::TimeSeries.TimeArray)` runs `PriceIngestion()` and carries no configuration
surface. A caller who wants a different join, a collapse, a declared span or `strict` writes the two
steps, which is also the only place the layer's pieces are visible.

## Consequences

- **The measured refusal stops firing.** The same five-day panel, five-day `iv` and fifteen-day
  benchmark returns `(5, 2)` on exactly `timestamp(X)`, with no non-finite cell and the benchmark
  aligned.

- **The emitted clock gains an invariant on the default path.** `timestamp(pr.X) == timestamp(X)`
  unless `collapse_args` is non-empty, so a caller declaring their own listing calendar can size it
  against the table they hold. ADR 0129's caller-calendar route no longer needs a trial run to learn
  its own shape.

- **The layer treats its four inputs by one rule.** The asset table states the clock; the factor,
  benchmark and implied-volatility series are aligned to it. The implied volatilities were already
  aligned unconditionally, with no knob of their own, and the other two now agree with them.

- **A caller who wants the union or the intersection has both**, and neither is silent:

  ```julia
  price_ingestion(PriceIngestion(; join_method = :outer), X; B = B)   # the union clock
  price_ingestion(PriceIngestion(; join_method = :inner), X; B = B)   # the intersection
  ```

- **A caller who wants no ingestion builds the carrier.** `prices_to_returns(pr::PricesResult)`
  takes one directly, and a carrier stating no span states no universe, which is ADR 0132's single
  meaning for `pnl === nothing`.

- **`CONTEXT.md` mints no term.** Its **Coverage Universe** *Avoid* line gains one clause: an
  estimator reading a second input narrows further within its own fit, which is that estimator's
  reduction rather than a different Coverage Universe. The definition itself is unchanged, because
  the Coverage Universe is still finiteness of the return and the active mask.

- **ADR 0133 is rewritten in place**, not having reached `main`. Its *what this ADR does not settle*
  paragraph is answered here, and its `join_method` paragraph now points at this decision rather
  than deferring to it.

- **Three defects were found while deciding, and are filed rather than folded in.**
  `assert_finite(val::ArrNum)` is `any(isfinite, val)`, so one finite cell satisfies a check named
  *finite*; it sits under a family with 291 call sites, so the blast radius is the maintainer's.
  `test_55_numeric_coercion_census.jl` does not match the `Float64(` spelling of the coercion it
  gates. The unguarded `cov(ce::ImpliedVolatility, X; iv = …)` keyword path is filed with its fix
  named, being closed by the coverage decision above.

## Alternatives considered

- **Delete `join_method` and join onto the asset clock unconditionally.** It buys a permanent
  invariant and one rule for all four inputs, and every value it removes is one caller-side line.
  It was refused because that line is not free: a caller reproducing the union clock merges *before*
  the door, where the tables have not been unified, so they must share a value type already or reach
  for `unify_gaps`, which is unexported. The knob keeps the union clock a keyword.

- **Default to `:outer`, and correct the docstring instead.** It discards no covariate observation
  and is honest about what is absent. It was refused because the honesty is on the wrong axis: the
  padding it admits is asset padding invented by a covariate's length, which the **Span Rule** then
  reads as listings and delistings, and it leaves the measured `iv` refusal live.

- **Default to `:inner`.** Its one use in the repository — *the benchmark on my asset clock* — is
  served by `:left` in every case and served correctly in one more: a covariate with a hole inside
  the asset clock deletes that observation for every asset under `:inner`, and pads only the
  covariate under `:left`.

- **Refuse an absent implied volatility rather than carrying one.** A `NaN` implied volatility is
  silently wrong rather than loud in every consumer — `cor2cov!` writes a `NaN` row and column into
  the covariance with no throw, and `predict_realised_vols` under `ImpliedVolatilityPremium` reads
  only the last row, so a `NaN` there poisons one asset and a `NaN` elsewhere is ignored. The
  refusal was refused because the library's own answer to *this asset has no data in this window* is
  to exclude the asset and carry on, not to fail the fit; the coverage narrowing gives exactly that,
  and gives it on the path that reads the data.

- **Mint an algorithm family for the span**, mirroring `AbstractGapReturnAlgorithm` and
  `AbstractPanelFillAlgorithm`. Deferred rather than refused: it buys a place for a second
  derivation rule, and no second rule is yet wanted. It is recorded in map
  [#955](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/955)'s *Not yet specified*.

- **Keep `Float64` as the unification target and refuse a mixed panel by name.** It fixes the
  confusing `MethodError` at a fraction of the cost. It was refused because it leaves the layer
  closed to `Float32` and to every number type the library has not seen, which is the rule
  `.github/instructions/julia-source-code.instructions.md` § *Numeric types come from the data*
  states.
