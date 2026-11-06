```@meta
Description = "Base finite allocation, public API of PortfolioOptimisers.jl: FiniteAllocationOptimisationEstimator, AbstractCollateralAlgorithm, ProceedsCollateral, …"
```

# Base finite allocation

A new finite allocation estimator subtypes `FiniteAllocationOptimisationEstimator` and adds the methods that its docstring states. A new collateral algorithm subtypes `AbstractCollateralAlgorithm` and adds the two call methods that its docstring states. A new rule for the target weight of an asset with no price on the allocation date subtypes `AbstractUnpricedWeightAlgorithm` and adds the call method that its docstring states.

```@docs
PortfolioOptimisers.FiniteAllocationOptimisationEstimator
PortfolioOptimisers.AbstractCollateralAlgorithm
ProceedsCollateral
CashCollateral
PortfolioOptimisers.AbstractUnpricedWeightAlgorithm
ProportionalRedistribution
EvenRedistribution
AssetRedistribution
SideRedistribution
PortfolioOptimisers.unpriced_weights
FiniteAllocationInput
factory(res::FiniteAllocationOptimisationResult, fb::Option{<:FOptE_FOpt_FbChain})
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
