```@meta
Description = "Preprocessing partial fit, private API of PortfolioOptimisers.jl: PricesToReturnsState, PriceGapFillState, MissingDataFilterState, vcat_carrier_rows, …"
```

# Preprocessing partial fit: private API

## Types

```@docs
PortfolioOptimisers.PricesToReturnsState
PortfolioOptimisers.PriceGapFillState
PortfolioOptimisers.MissingDataFilterState
```

## Functions

```@docs
PortfolioOptimisers.vcat_carrier_rows
PortfolioOptimisers.assert_pinned_carrier
PortfolioOptimisers.vcat_optional
PortfolioOptimisers.vcat_panel_rows
PortfolioOptimisers.carrier_rows
PortfolioOptimisers.partial_fit_transform
PortfolioOptimisers.series_values
PortfolioOptimisers.advance_anchor
PortfolioOptimisers.block_gap_return!
PortfolioOptimisers.series_values_returns
PortfolioOptimisers.returns_with_series
PortfolioOptimisers.show_fields(est::Union{<:PricesToReturns, <:PriceGapFill, <:MissingDataFilter})
```
