```@meta
Description = "Returns result, public API of PortfolioOptimisers.jl: ReturnsResult, asset_panel, port_opt_view, returns_result_picker."
```

# Returns result

## Prices to returns

Other than [`FiniteAllocationOptimisationEstimator`](@ref), all optimisations work based off returns data rather than price data. These functions and types are involved in computing returns.

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
