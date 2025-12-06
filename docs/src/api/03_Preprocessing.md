# Pre-processing

## Prices to returns

Other than [`FiniteAllocationOptimisationEstimator`](@ref), all optimisations work based off returns data rather than price data. These functions and types are involved in computing returns.

```@docs
AbstractReturnsResult
ReturnsResult
_check_names_and_returns_matrix
prices_to_returns
```

## Pre-filtering

Price data is often incomplete or noisy, so it can be worthwhile having some pre-filtering steps to remove data that does not contribute meaningful information and may pollute calculations.

```@docs
find_complete_indices
select_k_extremes
```
