# Preprocessing partial fit

## The online form of the data steps

A [`Pipeline`](@ref) is a host of the online step (ADR 0142), and the data steps before its row
owner take the step in the form their class allows. A row-local step folds a block of
observations and emits the block its transform gives it, through
[`PortfolioOptimisers.partial_fit_transform`](@ref): [`PricesToReturns`](@ref) keeps the last
price row it saw, [`PriceGapFill`](@ref) with a [`CarriedPrice`](@ref) keeps the carried prices,
and [`MissingDataFilter`](@ref) keeps its missing counts. Each reads its fitted Result out through
[`fit_preprocessing`](@ref) with no data. A window-valued configuration answers
[`PortfolioOptimisers.supports_partial_fit`](@ref) `false`, and the Pipeline refuses it at warm-up
by name.

```@docs
PortfolioOptimisers.vcat_carrier_rows
PortfolioOptimisers.assert_pinned_carrier
PortfolioOptimisers.vcat_optional
PortfolioOptimisers.vcat_panel_rows
PortfolioOptimisers.carrier_rows
PortfolioOptimisers.partial_fit_transform
PortfolioOptimisers.PricesToReturnsState
PortfolioOptimisers.series_values
PortfolioOptimisers.advance_anchor
PortfolioOptimisers.block_gap_return!
PortfolioOptimisers.series_values_returns
PortfolioOptimisers.returns_with_series
fit_preprocessing(ptr::PricesToReturns)
PortfolioOptimisers.PriceGapFillState
fit_preprocessing(est::PriceGapFill)
PortfolioOptimisers.MissingDataFilterState
fit_preprocessing(mdf::MissingDataFilter)
PortfolioOptimisers.show_fields(est::Union{<:PricesToReturns, <:PriceGapFill, <:MissingDataFilter})
```
