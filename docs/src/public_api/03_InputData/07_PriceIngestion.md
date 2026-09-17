```@meta
Description = "Price ingestion, public API of PortfolioOptimisers.jl: PriceIngestion, price_ingestion."
```

# Price ingestion

## The ingestion layer

[`PriceIngestion`](@ref) assembles raw price series into the carrier the conversion reads. It runs
**once on the whole panel** and is deliberately not a [`Pipeline`](@ref) step: unification of the
two absent-price conventions, the factor and benchmark join, and the frequency collapse each move
or renumber the observations, folds are cut once on the carrier's clock, and a hyperparameter that
changes the test set cannot be scored against one that does not. Running outside the `Pipeline` is
also what lets the **Span Rule** read the whole panel, which a step — seeing only a window — cannot.

The carrier it emits holds the Listing Span in its `span` field. [`PricesToReturns`](@ref) projects
that onto the returns clock and hands the [`ReturnsResult`](@ref) an [`AssetPanel`](@ref) stating
the universe — **always**, a gapless panel included, so `pnl === nothing` on a returns carrier means
one thing only: the carrier was not built by the layer. The gapless case costs nothing to say: both
masks become a [`PortfolioOptimisers.AllTrueMask`](@ref), which stores no cell.

The layer spells an absent price `NaN`, which is the library's one spelling for absence, and the
conversion carries it into the returns rather than deleting the observation or the asset that holds
one. A carrier the layer did not build states no span, and then no universe: the conversion does not
guess one from the window, because a delisting straddling the window end reads there as an asset
that was never listed. See
`docs/adr/0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md`
and
`docs/adr/0132-the-layer-emits-one-carrier-and-alignment-splits-into-a-fixed-axis-and-a-provenance-check.md`.

**The asset table states the clock.** Under the default `join_method = :left` the factor,
benchmark and implied-volatility series are aligned to the asset clock and padded `NaN` where they
are silent, so `timestamp(pr.X) == timestamp(X)` unless `collapse_args` is non-empty; `:outer` and
`:inner` stay reachable. What the join padded is named — which table, how many observations, which
columns — warning by default and refusing under `strict`. The value type the carrier holds is
derived from the series rather than named: a `Float32` panel stays `Float32`, `Float32` beside
`Float64` joins in `Float64`, an integer panel takes the floating-point type the return arithmetic
gives it, and a type that cannot spell an absence is refused by name at the first gap it would have
to spell. An absent implied volatility is carried like an absent price, and
[`ImpliedVolatility`](@ref) narrows its Coverage Universe to the columns whose implied
volatilities are complete. See
`docs/adr/0135-the-asset-table-states-the-clock-and-the-layer-carries-every-absence-and-names-it.md`.

## Types

```@docs
PriceIngestion
```

## Functions

```@docs
price_ingestion
```
