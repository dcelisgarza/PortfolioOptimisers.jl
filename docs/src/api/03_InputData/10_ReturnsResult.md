# Returns result

## Prices to returns

Other than [`FiniteAllocationOptimisationEstimator`](@ref), all optimisations work based off returns data rather than price data. These functions and types are involved in computing returns.

```@docs
AbstractReturnsResult
ReturnsResult
check_names_and_returns_matrix
PortfolioOptimisers.asset_panel(::Nothing, ::Any, ::ReturnsResult, ::Any)
PortfolioOptimisers.assert_asset_panel_supplied
port_opt_view(::ReturnsResult, ::Any)
returns_result_picker
Prices_RR
```
