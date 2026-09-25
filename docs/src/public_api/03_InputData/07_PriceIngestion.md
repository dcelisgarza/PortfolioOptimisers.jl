```@meta
Description = "Price ingestion, public API of PortfolioOptimisers.jl: PriceIngestion, price_ingestion."
```

# Price ingestion

## From raw prices to a prices result

[`PriceIngestion`](@ref) turns raw price series into the `PricesResult` that the conversion to
returns reads. It runs once, on the whole history, and it is not a [`Pipeline`](@ref) step. It also
spells every absent price, `missing` or a non-finite number, as `NaN`. Two of its steps change
the observations themselves. It joins the factor and benchmark series to the prices, and it can
collapse the data to a lower frequency. Cross-validation cuts its folds once, on the
observations of the result, so a step that adds, removes or renumbers observations cannot run
inside a fold. A hyperparameter that changed the test window would also give scores that you
cannot compare. Because it sees the whole history, `PriceIngestion` also finds the listing span of
each asset from all of its prices. A step inside a pipeline sees one window, which does not show
whether an asset is listed before or after it.

The result holds the listing span of each asset in its `span` field. [`PricesToReturns`](@ref)
moves the spans onto the dates of the returns and gives the [`ReturnsResult`](@ref) an
[`AssetPanel`](@ref) that states the universe. It does so even when no price is missing, so a
`ReturnsResult` with `pnl === nothing` did not come from `PriceIngestion`. When no price is
missing, both universe masks are a [`PortfolioOptimisers.AllTrueMask`](@ref), which stores no
cells.

`PriceIngestion` writes an absent price as `NaN`, the one value the library uses for a missing
number. The conversion to returns keeps it as a `NaN` return, and deletes neither the observation
nor the asset. A `PricesResult` that `PriceIngestion` did not build has no spans, so the conversion
gives its returns no universe. The conversion does not guess a universe from one window.

The asset prices set the observation dates. With the default `join_method = :left`,
`PriceIngestion` aligns the factor, benchmark and implied-volatility series to the dates of the
asset prices, and writes `NaN` where a series has no value. The result then has the timestamps of
the asset prices, `timestamp(pr.X) == timestamp(X)`, unless `collapse_args` is non-empty. The
`:outer` join keeps the union of the dates, and the `:inner` join keeps their intersection.
`PriceIngestion` reports every observation it padded, with the table, the number of observations
and the columns. It warns by default, and throws an error when `strict` is set.

The element type of the result comes from the input series. A `Float32` panel stays `Float32`,
`Float32` next to `Float64` gives `Float64`, and an integer panel takes the floating-point type of
its returns. A type that has no value for an absent number throws an error that names the type.
The error comes at a gap, before a join that can pad, and when `PriceIngestion` aligns the implied
volatilities, even if the join or the alignment then pads nothing. An absent implied volatility
is kept like an absent price, and [`ImpliedVolatility`](@ref) then fits only on the columns whose
implied volatilities have no gaps.

## Types

```@docs
PriceIngestion
```

## Functions

```@docs
price_ingestion
```
