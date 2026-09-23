```@meta
Description = "Returns result, public API of PortfolioOptimisers.jl: ReturnsResult, asset_panel, port_opt_view, returns_result_picker."
```

# Returns result

## Prices to returns

Every optimiser works on returns, except a [`FiniteAllocationOptimisationEstimator`](@ref), which works on prices. The types and functions below hold returns and compute them from prices.

## Types

```@docs
ReturnsResult
```

## Functions

```@docs
PortfolioOptimisers.asset_panel(::Nothing, ::Any, ::ReturnsResult, ::Any)
port_opt_view(::ReturnsResult, ::Any)
returns_result_picker
```
