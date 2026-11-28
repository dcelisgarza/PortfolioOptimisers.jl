```@meta
Description = "Preprocessing partial fit, private API of PortfolioOptimisers.jl: PricesToReturnsState, PriceGapFillState, MissingDataFilterState, PriceJumpFilterState, …"
```

# Preprocessing partial fit: private API

## Types

```@docs
PortfolioOptimisers.PricesToReturnsState
PortfolioOptimisers.PriceGapFillState
PortfolioOptimisers.MissingDataFilterState
PortfolioOptimisers.PriceJumpFilterState
```

## Functions

```@docs
PortfolioOptimisers.vcat_observations
PortfolioOptimisers.vcat_returns_panel
PortfolioOptimisers.finite_universe_panel
PortfolioOptimisers.returns_with_universe
PortfolioOptimisers.assert_pinned_value
PortfolioOptimisers.vcat_optional
PortfolioOptimisers.data_row_count
PortfolioOptimisers.partial_fit_transform
PortfolioOptimisers.series_values
PortfolioOptimisers.advance_anchor
PortfolioOptimisers.block_gap_return!
PortfolioOptimisers.series_values_returns
PortfolioOptimisers.returns_with_series
PortfolioOptimisers.end_price
PortfolioOptimisers.show_fields(est::Union{PricesToReturns, PriceGapFill, MissingDataFilter, PriceJumpFilter})
```
